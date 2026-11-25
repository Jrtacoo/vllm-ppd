"""Observe default vLLM scheduling without changing its decisions.

Load via --scheduler-cls phase10_scheduler.TraceScheduler from a separate mount.
The record covers engine scheduling/model wall time, not CUDA kernel time.
"""
import json
import hashlib
import os
from pathlib import Path
import threading
import time

from vllm.v1.core.sched.scheduler import Scheduler


class TraceScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._figure2_path = Path(os.environ["FIGURE2_TRACE"])
        self._figure2_records = []
        self._figure2_step = 0
        self._figure2_lock = threading.Lock()
        self._figure2_pending = {}
        # Export only when the external runner explicitly requests it AFTER
        # measured requests have finished. No serialization/I/O in engine steps.
        threading.Thread(target=self._figure2_export, daemon=True).start()

    def _figure2_export(self):
        flag = self._figure2_path.parent / "export-request"
        seen = None
        while True:
            time.sleep(0.5)
            if not flag.exists():
                continue
            current = flag.read_text().strip()
            # A truncated/partially-written control file is not an export
            # request. Older controllers could briefly expose an empty file.
            if not current.isdigit() or current == seen:
                continue
            try:
                self._figure2_export_token(current)
                seen = current
            except Exception as exc:
                # Never drain memory after a failed export. Expose the error
                # to the controller instead of losing a daemon thread silently.
                error = self._figure2_path.parent / "export-error.json"
                temporary = error.with_suffix(".tmp")
                temporary.write_text(json.dumps(dict(token=current, error=repr(exc))))
                temporary.replace(error)

    def _figure2_export_token(self, token):
        """Commit a uniquely named snapshot and receipt before draining RAM.

        Only the external controller calls this protocol, between idle groups.
        A repeated token reuses its immutable snapshot; an empty later export
        cannot overwrite any earlier nonempty evidence.
        """
        if not token.isdigit():
            return None
        directory = self._figure2_path.parent / "exports"
        directory.mkdir(exist_ok=True)
        target = directory / (token + ".jsonl")
        receipt_path = directory / (token + ".json")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["token"] != token or not target.is_file():
                raise RuntimeError("Invalid existing export receipt")
            return receipt
        with self._figure2_lock:
            records = list(self._figure2_records)
        temporary = target.with_suffix(".tmp")
        digest = hashlib.sha256()
        with temporary.open("wb") as f:
            for record in records:
                data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                f.write(data)
                digest.update(data)
            f.flush()
            os.fsync(f.fileno())
        if target.exists():
            with target.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != digest.hexdigest():
                    raise RuntimeError("Existing immutable snapshot differs")
            temporary.unlink()
        else:
            temporary.replace(target)
        receipt = dict(protocol=2, token=token, file="exports/" + target.name,
                       records=len(records), bytes=target.stat().st_size,
                       sha256=digest.hexdigest())
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt))
        temporary.replace(receipt_path)
        # The receipt acknowledges a committed snapshot, never an empty flag.
        with self._figure2_lock:
            del self._figure2_records[:len(records)]
        ack = self._figure2_path.with_suffix(".exported")
        temporary = ack.with_suffix(".tmp")
        temporary.write_text(token)
        temporary.replace(ack)
        return receipt

    def schedule(self):
        started = time.perf_counter_ns()
        wall_ns = time.time_ns()
        output = super().schedule()
        observer_start = time.perf_counter_ns()
        requests = []
        for request_id, n in output.num_scheduled_tokens.items():
            r = self.requests[request_id]
            requests.append({
                "id": request_id, "scheduled_tokens": n,
                "computed_before": r.num_computed_tokens - n,
                "prompt_tokens": r.num_prompt_tokens,
                "output_tokens_before": r.num_output_tokens,
                "cached_tokens": r.num_cached_tokens,
                "arrival_wall_s": r.arrival_time,
                "preemptions": r.num_preemptions,
            })
        self._figure2_step += 1
        self._figure2_pending[id(output)] = {
            "step": self._figure2_step, "schedule_start_ns": started,
            "schedule_wall_ns": wall_ns,
            "schedule_return_ns": None, "requests": requests,
            "waiting": len(self.waiting), "running": len(self.running),
            "preempted_ids": list(output.preempted_req_ids),
        }
        record = self._figure2_pending[id(output)]
        record["capture_observer_us"] = (time.perf_counter_ns() - observer_start) / 1000
        record["schedule_return_ns"] = time.perf_counter_ns()
        return output

    def update_from_output(self, scheduler_output, model_runner_output):
        entered = time.perf_counter_ns()
        result = super().update_from_output(scheduler_output, model_runner_output)
        record = self._figure2_pending.pop(id(scheduler_output))
        record["output_received_ns"] = entered
        record["update_return_ns"] = time.perf_counter_ns()
        record["model_wall_ms"] = (entered - record["schedule_return_ns"]) / 1e6
        observer_start = time.perf_counter_ns()
        with self._figure2_lock:
            self._figure2_records.append(record)
        record["observer_us"] = record["capture_observer_us"] + (time.perf_counter_ns() - observer_start) / 1000
        return result

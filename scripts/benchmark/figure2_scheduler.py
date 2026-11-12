"""Observe default vLLM scheduling without changing its decisions.

Load via --scheduler-cls figure2_scheduler.TraceScheduler from a separate mount.
The record covers engine scheduling/model wall time, not CUDA kernel time.
"""
import json
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
            current = flag.read_text()
            if current == seen:
                continue
            temporary = self._figure2_path.with_suffix(".tmp")
            with temporary.open("w") as f:
                for record in self._figure2_records.copy():
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
            temporary.replace(self._figure2_path)
            self._figure2_path.with_suffix(".exported").write_text(current)
            seen = current

    def schedule(self):
        started = time.perf_counter_ns()
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
                "preemptions": r.num_preemptions,
            })
        self._figure2_step += 1
        self._figure2_pending[id(output)] = {
            "step": self._figure2_step, "schedule_start_ns": started,
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
        self._figure2_records.append(record)
        record["observer_us"] = record["capture_observer_us"] + (time.perf_counter_ns() - observer_start) / 1000
        return result

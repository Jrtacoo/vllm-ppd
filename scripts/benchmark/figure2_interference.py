#!/usr/bin/env python3
"""Single-service Figure 2 adaptation with request/SSE and overlap evidence.

Never starts/stops services. Exact prompt token IDs avoid token-count drift.
Every trial warms independent cache salts before measuring; all three scenarios
warm the same number and size of prefixes. No inference request is retried.
"""
import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import aiohttp

MODEL = "Qwen2.5-7B-Instruct"
SCENARIOS = ("decode_only", "full_prefill", "append_prefill")
FIELDS = ("trial", "scenario", "batch_size", "repeat", "role", "request_id", "success",
          "prompt_tokens", "cached_tokens", "completion_tokens", "ttft_ms", "tpot_ms",
          "content_tpot_ms", "token_tpot_ms", "e2e_ms", "first_token_ms", "last_token_ms",
          "start_ns", "end_ns", "finish_reason", "raw_stem", "error")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


async def stream(session, url, payload, stem, on_token=None, artifact_sink=None):
    if artifact_sink is None:
        write_json(Path(str(stem) + ".request.json"), payload)
    row = {k: None for k in FIELDS}
    row.update(success=False, error=None, request_id=payload["request_id"],
               prompt_tokens=len(payload["prompt"]), text="", token_events=0, observed_tokens=0)
    row["start_ns"] = time.perf_counter_ns()
    raw, events, data_lines = bytearray(), [], []
    last_content, done = None, False

    def parse_event():
        nonlocal last_content, done
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        now = time.perf_counter_ns()
        ms = (now - row["start_ns"]) / 1e6
        if done:
            raise ValueError("SSE data after DONE")
        if data == "[DONE]":
            done = True
            events.append({"elapsed_ms": ms, "perf_ns": now, "data": data})
            return
        obj = json.loads(data)
        events.append({"elapsed_ms": ms, "perf_ns": now, "data": obj})
        if obj.get("error"):
            raise ValueError("SSE error: " + str(obj["error"]))
        if obj.get("usage"):
            row["usage"] = obj["usage"]
        for choice in obj.get("choices", []):
            content = choice.get("text") or ""
            if content:
                if row["ttft_ms"] is None:
                    row["ttft_ms"] = ms
                last_content = ms
                row["text"] += content
            token_ids = choice.get("token_ids") or []
            if token_ids:
                row["observed_tokens"] += len(token_ids)
                row["token_events"] += 1
                if row["first_token_ms"] is None:
                    row["first_token_ms"] = ms
                row["last_token_ms"] = ms
                if on_token:
                    on_token(row["observed_tokens"])
            if choice.get("finish_reason") is not None:
                row["finish_reason"] = choice["finish_reason"]

    try:
        async with session.post(url + "/v1/completions", json=payload) as response:
            row["http_status"] = response.status
            if response.status != 200 or "text/event-stream" not in response.headers.get("Content-Type", ""):
                raw.extend(await response.read())
                raise ValueError(f"HTTP {response.status}: {bytes(raw)[:500]!r}")
            async for line in response.content:
                raw.extend(line)
                text = line.decode("utf-8").rstrip("\r\n")
                if not text:
                    parse_event()
                elif text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
            if data_lines:
                raise ValueError("Unterminated SSE event")
        row["end_ns"] = time.perf_counter_ns()
        row["e2e_ms"] = (row["end_ns"] - row["start_ns"]) / 1e6
        usage = row.get("usage") or {}
        row["prompt_tokens"] = usage.get("prompt_tokens")
        row["completion_tokens"] = usage.get("completion_tokens")
        row["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if not done or row["ttft_ms"] is None or row["first_token_ms"] is None:
            raise ValueError("Missing DONE, content, or token-ID events")
        if row["prompt_tokens"] != len(payload["prompt"]) or row["completion_tokens"] != payload["max_tokens"]:
            raise ValueError("Usage differs from exact requested token lengths")
        if row["observed_tokens"] != row["completion_tokens"] or row["finish_reason"] != "length":
            raise ValueError("Missing tokens or unexpected finish reason")
        n = row["completion_tokens"]
        if n > 1:
            row["tpot_ms"] = (row["e2e_ms"] - row["ttft_ms"]) / (n - 1)
            row["content_tpot_ms"] = (last_content - row["ttft_ms"]) / (n - 1)
            row["token_tpot_ms"] = (row["last_token_ms"] - row["first_token_ms"]) / (n - 1)
        row["success"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["end_ns"] = row["end_ns"] or time.perf_counter_ns()
        row["e2e_ms"] = row["e2e_ms"] or (row["end_ns"] - row["start_ns"]) / 1e6
        if artifact_sink is None:
            Path(str(stem) + ".sse").write_bytes(raw)
            write_json(Path(str(stem) + ".events.json"), events)
            write_json(Path(str(stem) + ".response.json"), row)
        else:
            artifact_sink.append((stem, payload, raw, events, dict(row)))
    return row


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir()
    rng = random.Random(args.seed)
    schedule = []
    for repeat in range(args.repeats):
        batches = args.batches.copy()
        rng.shuffle(batches)
        for batch in batches:
            scenarios = list(SCENARIOS[1:] if batch == 0 else SCENARIOS)
            rng.shuffle(scenarios)
            for scenario in scenarios:
                schedule.append({"scenario": scenario, "batch_size": batch, "repeat": repeat})
    write_json(args.output / "run.json", {
        "model": MODEL, "server": args.server, "seed": args.seed, "schedule": schedule,
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "decode_cached_tokens": args.decode_history_tokens, "decode_new_tokens": 4, "decode_output_tokens": args.output_tokens,
        "full_prefill_cached_tokens": 0, "full_prefill_new_tokens": args.new_input_tokens,
        "full_prefill_computed_tokens": args.new_input_tokens + (args.prefill_history_tokens if args.full_prefill_mode == "same-context" else 0),
        "full_prefill_mode": args.full_prefill_mode,
        "append_prefill_cached_tokens": args.prefill_history_tokens, "append_prefill_new_tokens": args.new_input_tokens,
        "full_prefill_prompt_tokens": args.new_input_tokens + (args.prefill_history_tokens if args.full_prefill_mode == "same-context" else 0),
        "append_prefill_prompt_tokens": args.prefill_history_tokens + args.new_input_tokens,
        "max_prefill_chunks": args.max_prefill_chunks,
        "batch_zero_policy": "prefill-only full/append; no decode baseline or decode TPOT",
        "interferer_output_tokens": 1, "inject_after_each_decode_output_tokens": 8,
        "primary_tpot": "(last token-ID event - first token-ID event) / (completion_tokens - 1)",
        "replicate_aggregation": "mean TPOT across background decode requests within each trial",
        "cache_isolation": "independent cache_salt for each role/trial; same logical prompts across three scenarios within a repeat/batch",
        "clock": "time.perf_counter_ns; same dcn10 kernel as scheduler trace",
        "failure_policy": "retain all raw evidence; finish in-flight requests, stop on failed trial; no retry",
        "http_connections": "force_close=True; a fresh TCP connection for every request avoids stale keep-alive reuse",
        "trial_prefix": args.trial_prefix or args.output.name,
        "prefix_mode": args.prefix_mode,
        "warmup_policy": "two prefixes (shared background + separate interferer)" if args.prefix_mode == "shared" else "batch+1 independent prefixes",
        "connection_limit": max(100, max(args.batches) + 16),
        "shared_prefix_caveat": "Background histories share physical KV; new 4-token suffixes differ per sequence. Not independent-KV scaling." if args.prefix_mode == "shared" else None,
        "artifact_persistence": "buffer during active inference; flush after all in-flight requests finish" if args.defer_artifacts else "per-request synchronous persistence",
    })
    connection_limit = max(100, max(args.batches) + 16)
    connector = aiohttp.TCPConnector(limit=connection_limit, limit_per_host=connection_limit, force_close=True)
    timeout = aiohttp.ClientTimeout(total=120)
    with (args.output / "requests.jsonl").open("x") as records, (args.output / "requests.csv").open("x", newline="") as csvfile, (args.output / "trials.jsonl").open("x") as trialfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
            async with session.get(args.server + "/health") as r:
                r.raise_for_status()
            async with session.post(args.server + "/tokenize", json={"model": MODEL, "prompt": "Reference notes: data model serving cache memory inference request token sequence analysis performance. " * 16, "add_special_tokens": False}) as r:
                r.raise_for_status()
                pool = (await r.json())["tokens"]
            write_json(args.output / "token-pool.json", pool)

            for seq, meta in enumerate(schedule):
                trial = f"{args.trial_prefix or args.output.name}-b{meta['batch_size']}-r{meta['repeat']}-{meta['scenario']}"
                batch = meta["batch_size"]
                inputs_rng = random.Random(f"figure2/{args.seed}/{meta['repeat']}/{batch}")
                prefixes = [[inputs_rng.choice(pool) for _ in range(args.decode_history_tokens)] for _ in range(batch)]
                prefixes.append([inputs_rng.choice(pool) for _ in range(args.prefill_history_tokens)])
                # Identical new input token IDs for the cold and cached interference paths.
                extra = [inputs_rng.choice(pool) for _ in range(args.new_input_tokens)]
                if args.prefix_mode == "shared" and batch:
                    prefixes = [prefixes[0]] * batch + [prefixes[-1]]
                    vocabulary = sorted(set(pool))
                    if batch > len(vocabulary) ** 4:
                        raise ValueError("Token pool cannot supply unique four-token suffixes")
                    decode_suffixes = [[vocabulary[(i // len(vocabulary) ** j) % len(vocabulary)]
                                        for j in range(4)] for i in range(batch)]
                else:
                    decode_suffixes = [extra[:4]] * batch

                def payload(prompt, name, salt, output):
                    return {"model": MODEL, "prompt": prompt, "max_tokens": output,
                            "temperature": 0, "seed": args.seed, "ignore_eos": True,
                            "stream": True, "stream_options": {"include_usage": True},
                            "return_token_ids": True, "request_id": name, "cache_salt": salt}

                def retain(row, role):
                    row.update(meta, trial=trial, role=role,
                               raw_stem="raw/" + row["request_id"])
                    if args.defer_artifacts:
                        deferred_rows.append(dict(row))
                    else:
                        records.write(json.dumps(row, ensure_ascii=False) + "\n")
                        records.flush()
                        writer.writerow({k: row.get(k) for k in FIELDS})
                        csvfile.flush()

                deferred_artifacts, deferred_rows = [], []

                def flush_artifacts():
                    # No inference is in flight at either call site.
                    for stem, data, raw, events, response in deferred_artifacts:
                        write_json(Path(str(stem) + ".request.json"), data)
                        Path(str(stem) + ".sse").write_bytes(raw)
                        write_json(Path(str(stem) + ".events.json"), events)
                        write_json(Path(str(stem) + ".response.json"), response)
                    deferred_artifacts.clear()
                    for row in deferred_rows:
                        records.write(json.dumps(row, ensure_ascii=False) + "\n")
                        writer.writerow({k: row.get(k) for k in FIELDS})
                    records.flush()
                    csvfile.flush()
                    deferred_rows.clear()

                async def request(prompt, role, suffix, output, salt, callback=None):
                    name = trial + "_" + role + "_" + suffix
                    row = await stream(session, args.server, payload(prompt, name, salt, output),
                                       args.output / "raw" / name, callback,
                                       deferred_artifacts if args.defer_artifacts else None)
                    retain(row, role)
                    return row

                salts = [trial + f"/prefix/{i}" for i in range(batch + 1)]
                if args.prefix_mode == "shared" and batch:
                    salts = [trial + "/shared-background"] * batch + [salts[-1]]
                semaphore = asyncio.Semaphore(4)

                async def warm(i):
                    async with semaphore:
                        return await request(prefixes[i], "warmup", str(i), 1, salts[i])

                warm_indices = (0, batch) if args.prefix_mode == "shared" and batch else range(batch + 1)
                warmups = await asyncio.gather(*(warm(i) for i in warm_indices))
                flush_artifacts()
                if not all(r["success"] for r in warmups):
                    raise RuntimeError(f"{trial}: prefix warmup failed; retained")
                async with session.get(args.server + "/metrics") as r:
                    r.raise_for_status()
                    (args.output / f"{trial}-before-metrics.txt").write_text(await r.text())
                gate = asyncio.Event()
                if batch == 0:
                    gate.set()
                progress = [0] * batch

                def callback(i, n):
                    progress[i] = n
                    if min(progress) >= 8:
                        gate.set()

                trial_start = time.perf_counter_ns()
                tasks = [asyncio.create_task(request(prefixes[i] + decode_suffixes[i], "decode", str(i),
                         args.output_tokens, salts[i], lambda n, i=i: callback(i, n))) for i in range(batch)]
                injection = None
                error = None
                try:
                    await asyncio.wait_for(gate.wait(), timeout=30)
                    injection = time.perf_counter_ns()
                    if meta["scenario"] != "decode_only":
                        if meta["scenario"] == "full_prefill":
                            prompt = prefixes[-1] + extra if args.full_prefill_mode == "same-context" else extra
                            salt = trial + "/cold"
                        else:
                            prompt, salt = prefixes[-1] + extra, salts[-1]
                        tasks.append(asyncio.create_task(request(prompt, "interferer", "0", 1, salt)))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                results = await asyncio.gather(*tasks, return_exceptions=True)
                flush_artifacts()
                async with session.get(args.server + "/metrics") as r:
                    r.raise_for_status()
                    (args.output / f"{trial}-after-metrics.txt").write_text(await r.text())
                decoded = [r for r in results if isinstance(r, dict) and r["role"] == "decode"]
                interferers = [r for r in results if isinstance(r, dict) and r["role"] == "interferer"]
                checks = {
                    "all_requests_succeeded": all(isinstance(r, dict) and r["success"] for r in results),
                    "decode_count": len(decoded) == batch,
                    "decode_cached_expected": all(r["cached_tokens"] == args.decode_history_tokens for r in decoded),
                    "injected_after_decode_started": injection is not None and (not progress or min(progress) >= 8),
                    "interferer_count": len(interferers) == int(meta["scenario"] != "decode_only"),
                }
                if interferers:
                    checks["interferer_cache"] = interferers[0]["cached_tokens"] == (args.prefill_history_tokens if meta["scenario"] == "append_prefill" else 0)
                    checks["client_overlap"] = all(r["start_ns"] < interferers[0]["start_ns"] < r["end_ns"] for r in decoded)
                good = [r["token_tpot_ms"] for r in decoded if r["success"]]
                summary = dict(meta, trial=trial, sequence=seq, start_ns=trial_start,
                               injection_ns=injection, end_ns=time.perf_counter_ns(),
                               decode_avg_tpot_ms=statistics.mean(good) if good else None,
                               decode_avg_api_tpot_ms=statistics.mean(r["tpot_ms"] for r in decoded) if good else None,
                               checks=checks, error=error,
                               exceptions=[str(r) for r in results if isinstance(r, BaseException)])
                trialfile.write(json.dumps(summary) + "\n")
                trialfile.flush()
                print(json.dumps(summary), flush=True)
                if error or not all(checks.values()):
                    raise RuntimeError(f"{trial}: checks failed; evidence retained, no retry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://192.168.1.11:8300")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-prefix", help="Unique request/cache namespace when reusing an idle engine")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix-mode", choices=("independent", "shared"), default="independent")
    parser.add_argument("--decode-history-tokens", type=int, default=128)
    parser.add_argument("--prefill-history-tokens", type=int, default=16384)
    parser.add_argument("--new-input-tokens", type=int, default=1024)
    parser.add_argument("--full-prefill-mode", choices=("same-context", "new-only"), default="same-context",
                        help="same-context recomputes history+new; new-only is the legacy workload")
    parser.add_argument("--max-prefill-chunks", type=int, default=1)
    parser.add_argument("--defer-artifacts", action="store_true",
                        help="Write raw evidence only after all in-flight requests finish")
    args = parser.parse_args()
    if min(args.batches) < 0 or args.repeats < 1 or args.output_tokens < 32:
        parser.error("Nonnegative batches, positive repeats and at least 32 output tokens required")
    if min(args.decode_history_tokens, args.prefill_history_tokens, args.new_input_tokens) < 16 or args.max_prefill_chunks < 1:
        parser.error("At least 16 history/new tokens and positive max-prefill-chunks required")
    if args.decode_history_tokens % 16 or args.prefill_history_tokens % 16:
        parser.error("History token lengths must align to 16-token KV blocks")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

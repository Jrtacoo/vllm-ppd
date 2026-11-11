#!/usr/bin/env python3
"""External-service, two-turn streaming baseline; never starts/stops servers.

Client TTFT includes Proxy/P/network time. TPOT follows the project's convention:
(stream EOF - first nonempty text) / (completion_tokens - 1).
Artifacts are append-only per run; use a new output directory for each routing.
"""
import argparse
import asyncio
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import aiohttp

MODEL = "Qwen2.5-7B-Instruct"
FIELDS = ["routing", "workload", "concurrency", "conversation", "turn", "success",
          "prompt_tokens", "completion_tokens", "requested_output_tokens", "ttft_ms",
          "tpot_ms", "e2e_ms", "last_content_ms", "content_tpot_ms", "done_ms",
          "tokenized_prompt_tokens", "response_id", "finish_reason", "payload_sha256",
          "raw_stem", "error"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


async def json_http(session, url, payload=None):
    async with session.request("POST" if payload is not None else "GET", url, json=payload) as r:
        r.raise_for_status()
        return await r.json()


async def token_count(session, backend, prompt):
    result = await json_http(session, backend + "/tokenize",
                             {"model": MODEL, "prompt": prompt, "add_special_tokens": True})
    return result["count"]


async def prepare(session, backend, path, conversations):
    cells = []
    for name, target, append, output in [("short", 256, 32, 64), ("long", 768, 256, 128)]:
        for concurrency in (1, 4):
            cases = []
            for i in range(conversations):
                case = f"{name}-c{concurrency}-{i:02d}"
                nonce = hashlib.sha256(("ppd-phase5-v1/" + case).encode()).hexdigest()[:40]
                head = f"User: Session {nonce}. Read these reference notes:"
                tail = "\nDescribe the notes in complete English sentences.\nAssistant:"
                # ' data' is one Qwen token; still query the real tokenizer.
                n = max(0, target - await token_count(session, backend, head + tail))
                for _ in range(4):
                    prompt = head + " data" * n + tail
                    count = await token_count(session, backend, prompt)
                    if count == target:
                        break
                    n += target - count
                if count != target:
                    raise RuntimeError(f"unexpected tokenizer length: {case}: {count} != {target}")
                suffix = "\nUser: Continue with these additional details:" + " detail" * append
                suffix += "\nWrite another complete explanation.\nAssistant:"
                cases.append({"conversation": case, "prompt": prompt, "prompt_tokens": count,
                              "suffix": suffix})
            cells.append({"workload": name, "concurrency": concurrency,
                          "first_input_tokens": target, "append_filler_tokens": append,
                          "output_tokens": output, "cases": cases})
    write_json(path, {"version": 1, "model": MODEL, "conversations_per_cell": conversations,
                      "seed": 42, "cells": cells})


async def stream_request(session, url, payload, stem):
    """Retain failures and partial raw evidence, including HTTP-200 SSE errors."""
    write_json(stem.with_suffix(".request.json"), payload)
    result = dict(success=False, error=None, response_id=None, finish_reason=None,
                  prompt_tokens=None, completion_tokens=None, ttft_ms=None, tpot_ms=None,
                  e2e_ms=None, last_content_ms=None, content_tpot_ms=None, done_ms=None,
                  text="", usage=None, http_status=None)
    raw, events, text_parts, data_lines = bytearray(), [], [], []
    start = time.perf_counter()
    done = False

    def event():
        nonlocal done
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        elapsed = (time.perf_counter() - start) * 1000
        if done:
            raise ValueError("data received after [DONE]")
        if data == "[DONE]":
            done = True
            result["done_ms"] = elapsed
            events.append({"elapsed_ms": elapsed, "data": "[DONE]"})
            return
        item = json.loads(data)
        events.append({"elapsed_ms": elapsed, "data": item})
        if "error" in item:
            raise ValueError("SSE error: " + json.dumps(item["error"]))
        if item.get("id"):
            result["response_id"] = item["id"]
        if item.get("usage"):
            result["usage"] = item["usage"]
        for choice in item.get("choices", []):
            content = choice.get("text") or choice.get("delta", {}).get("content") or ""
            if content:
                if result["ttft_ms"] is None:
                    result["ttft_ms"] = elapsed
                result["last_content_ms"] = elapsed
                text_parts.append(content)
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]

    try:
        async with session.post(url + "/v1/completions", json=payload) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raw.extend(await response.read())
                raise ValueError(f"HTTP {response.status}: {bytes(raw)[:300]!r}")
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raw.extend(await response.read())
                raise ValueError("expected SSE response, got " + bytes(raw)[:300].decode(errors="replace"))
            # aiohttp's line reader buffers TCP fragments; decode only complete UTF-8 lines.
            async for line in response.content:
                raw.extend(line)
                line = line.decode("utf-8").rstrip("\r\n")
                if not line:
                    event()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
            if data_lines:
                raise ValueError("unterminated SSE event")
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        if not done or result["ttft_ms"] is None or not result["usage"]:
            raise ValueError("missing DONE, generated text or usage")
        usage = result["usage"]
        result["prompt_tokens"] = usage["prompt_tokens"]
        result["completion_tokens"] = usage["completion_tokens"]
        n = result["completion_tokens"]
        if n != payload["max_tokens"] or n < 2 or result["finish_reason"] != "length":
            raise ValueError(f"unexpected completion length/finish: {n}, {result['finish_reason']}")
        result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / (n - 1)
        result["content_tpot_ms"] = (result["last_content_ms"] - result["ttft_ms"]) / (n - 1)
        if not all(math.isfinite(result[k]) and result[k] > 0 for k in ("ttft_ms", "tpot_ms", "e2e_ms")):
            raise ValueError("invalid timing")
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if result["e2e_ms"] is None:
            result["e2e_ms"] = (time.perf_counter() - start) * 1000
    result["text"] = "".join(text_parts)
    stem.with_suffix(".sse").write_bytes(raw)
    write_json(stem.with_suffix(".events.json"), events)
    write_json(stem.with_suffix(".response.json"), result)
    return result


async def snapshot(session, proxy, prefill, decode, output, label):
    status = await json_http(session, proxy + "/status")
    write_json(output / (label + "-status.json"), status)
    for role, endpoint in [("prefill", prefill), ("decode", decode)]:
        async with session.get(endpoint + "/health") as r:
            if r.status != 200:
                raise RuntimeError(f"{role} unhealthy: {r.status}")
        async with session.get(endpoint + "/metrics") as r:
            r.raise_for_status()
            (output / (label + "-" + role + "-metrics.txt")).write_text(await r.text())
    return status


def summarize(records):
    groups = []
    for workload in ("short", "long"):
        for concurrency in (1, 4):
            for turn in (1, 2):
                rows = [r for r in records if r["workload"] == workload and
                        r["concurrency"] == concurrency and r["turn"] == turn]
                good = [r for r in rows if r["success"]]
                group = dict(workload=workload, concurrency=concurrency, turn=turn,
                             requests=len(rows), successful=len(good))
                for field in ("ttft_ms", "tpot_ms", "e2e_ms", "prompt_tokens", "completion_tokens"):
                    values = [r[field] for r in good]
                    group[field] = {"mean": statistics.mean(values), "median": statistics.median(values),
                                    "min": min(values), "max": max(values)} if values else None
                groups.append(group)
    return groups


async def run(args, session):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    manifest = json.loads(args.manifest.read_text())
    write_json(output / "run.json", {"routing": args.routing, "model": MODEL, "proxy": args.proxy,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "client_aiohttp": aiohttp.__version__, "clock": "time.perf_counter",
        "ttft": "POST start to first nonempty text event", "tpot": "(stream EOF - first text)/(usage completion_tokens - 1)",
        "scheduling": "closed loop: at most concurrency two-turn conversations; no think time except tokenizer validation",
        "sampling": {"temperature": 0, "seed": 42, "ignore_eos": True},
        "warmup": "one separate two-turn conversation per cell; excluded from requests.jsonl/csv",
        "cells": [{k: v for k, v in c.items() if k != "cases"} for c in manifest["cells"]]})
    records = []
    with (output / "requests.jsonl").open("x") as jsonl, (output / "requests.csv").open("x", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
        writer.writeheader()

        async def conversation(case, cell, warmup=False):
            prompt = case["prompt"]
            for turn in (1, 2):
                count = await token_count(session, args.decode, prompt)
                if count + cell["output_tokens"] > 2048:
                    raise RuntimeError("context limit would be exceeded")
                payload = dict(model=MODEL, prompt=prompt, max_tokens=cell["output_tokens"],
                               temperature=0, seed=42, ignore_eos=True, stream=True,
                               stream_options={"include_usage": True})
                name = f"{case['conversation']}-t{turn}"
                stem = output / "raw" / name
                result = await stream_request(session, args.proxy, payload, stem)
                row = dict(result, routing=args.routing, workload=cell["workload"],
                           concurrency=cell["concurrency"], conversation=case["conversation"], turn=turn,
                           requested_output_tokens=cell["output_tokens"], tokenized_prompt_tokens=count,
                           raw_stem="raw/" + name,
                           payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
                if result["success"] and result["prompt_tokens"] != count:
                    row.update(success=False, error="API usage/tokenizer count mismatch")
                if not warmup:
                    records.append(row)
                    jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    writer.writerow({k: row.get(k) for k in FIELDS})
                    csvfile.flush()
                if not row["success"]:
                    raise RuntimeError(f"{name}: {row['error']}")
                prompt += result["text"] + case["suffix"]

        for cell in manifest["cells"]:
            label = f"{cell['workload']}-c{cell['concurrency']}"
            warm = dict(cell["cases"][0])
            nonce = hashlib.sha256(("warmup/" + args.routing + "/" + label).encode()).hexdigest()[:40]
            warm["conversation"] = "warmup-" + label
            original_nonce = warm["prompt"].split("Session ", 1)[1].split(".", 1)[0]
            warm["prompt"] = warm["prompt"].replace(original_nonce, nonce, 1)
            await conversation(warm, cell, warmup=True)
            before = await snapshot(session, args.proxy, args.prefill, args.decode, output, label + "-before")
            if before["config"] != args.routing or before["ppd_mode_enabled"]:
                raise RuntimeError("unexpected routing or dynamic mode enabled")
            semaphore = asyncio.Semaphore(cell["concurrency"])
            async def worker(case):
                async with semaphore:
                    await conversation(case, cell)
            start = time.perf_counter()
            # Gather all outcomes so other in-flight streams are retained even if one fails.
            outcomes = await asyncio.gather(*(worker(case) for case in cell["cases"]), return_exceptions=True)
            elapsed = time.perf_counter() - start
            after = await snapshot(session, args.proxy, args.prefill, args.decode, output, label + "-after")
            n = len(cell["cases"])
            delta = {k: after["stats"][k] - before["stats"][k] for k in before["stats"]
                     if k != "start_time"}
            expected = {"total_requests": 2*n, "turn1_requests": n, "turn2plus_requests": n,
                        "pd_requests": 2*n if args.routing == "1P_1D" else 0,
                        "ppd_requests": n if args.routing == "1P_1pD" else 0,
                        "ppd_direct_requests": n if args.routing == "1P_1pD" else 0}
            checks = {k: delta[k] == v for k, v in expected.items()}
            write_json(output / (label + "-checks.json"), {"delta": delta, "expected": expected,
                "checks": checks, "cell_wall_seconds": elapsed,
                "errors": [str(r) for r in outcomes if isinstance(r, BaseException)]})
            write_json(output / "summary.json", {"routing": args.routing, "groups": summarize(records)})
            print(json.dumps({"routing": args.routing, "cell": label, "requests": 2*n,
                              "seconds": round(elapsed, 2), "routing_checks": checks}), flush=True)
            if any(isinstance(r, BaseException) for r in outcomes) or not all(checks.values()):
                raise RuntimeError("cell failed; see checks and per-request results")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversations", type=int, default=12)
    parser.add_argument("--routing", choices=("1P_1D", "1P_1pD"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--proxy", default="http://192.168.1.9:10001")
    parser.add_argument("--prefill", default="http://192.168.1.9:8100")
    parser.add_argument("--decode", default="http://192.168.1.11:8200")
    args = parser.parse_args()
    if args.conversations < 1 or (not args.prepare and (not args.routing or not args.output)):
        parser.error("positive conversations and routing/output required for a run")
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=180)) as session:
        if args.prepare:
            if args.manifest.exists():
                parser.error("manifest already exists")
            await prepare(session, args.decode, args.manifest, args.conversations)
        else:
            await run(args, session)


if __name__ == "__main__":
    asyncio.run(main())

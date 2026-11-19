#!/usr/bin/env python3
"""Paired serial PD KV-transport experiment with exact prompt token counts."""
import argparse
import asyncio
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp

MODEL = "Qwen2.5-7B-Instruct"
SIZES = (256, 512, 1024, 2048, 4096, 8192)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


async def post_json(session, url, body):
    async with session.post(url, json=body) as response:
        response.raise_for_status()
        return await response.json()


async def tokenize(session, backend, prompt):
    return (await post_json(session, backend + "/tokenize",
                            {"model": MODEL, "prompt": prompt,
                             "add_special_tokens": True}))["count"]


async def make_manifest(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(trust_env=False,
            timeout=aiohttp.ClientTimeout(total=60)) as session:
        cells = []
        for size in args.input_sizes:
            cases = []
            for rep in range(args.repeats):
                nonce = hashlib.sha256(f"{args.namespace}/{size}/{rep}".encode()).hexdigest()[:32]
                head = f"Session {nonce}. Read and retain these reference notes:"
                tail = "\nSummarize the reference notes precisely.\nAssistant:"
                prompt = head + tail
                count = await tokenize(session, args.decode, prompt)
                fill = max(0, size - count)
                for _ in range(12):
                    prompt = head + " data" * fill + tail
                    count = await tokenize(session, args.decode, prompt)
                    if count == size:
                        break
                    fill = max(0, fill + size - count)
                if count != size:
                    raise RuntimeError(f"could not construct {size} tokens, got {count}")
                cases.append({"id": f"n{size}-r{rep + 1}", "repeat": rep + 1,
                              "prompt": prompt, "prompt_tokens": count,
                              "requested_output_tokens": 32})
            cells.append({"input_tokens": size, "cases": cases})
    write_json(args.output, {"version": 1, "model": MODEL, "seed": 42,
                             "completion_tokens": 32, "cells": cells})


async def request(session, endpoint, case, stem):
    body = {"model": MODEL, "prompt": case["prompt"], "max_tokens": 32,
            "temperature": 0, "seed": 42, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    write_json(stem.with_suffix(".request.json"), body)
    result = {"id": case["id"], "input_tokens": case["prompt_tokens"],
              "success": False, "error": None, "ttft_ms": None, "e2e_ms": None,
              "completion_tokens": None, "prompt_tokens_reported": None,
              "finish_reason": None}
    raw, events, data_lines, text_seen = bytearray(), [], [], False
    done = False
    start = time.perf_counter()

    def consume():
        nonlocal text_seen, done
        data = "\n".join(data_lines)
        data_lines.clear()
        if not data:
            return
        elapsed = (time.perf_counter() - start) * 1000
        if data == "[DONE]":
            done = True
            events.append({"elapsed_ms": elapsed, "data": "[DONE]"})
            return
        item = json.loads(data)
        if item.get("id"):
            result["response_id"] = item["id"]
        events.append({"elapsed_ms": elapsed, "data": item})
        if "error" in item:
            raise RuntimeError(json.dumps(item["error"]))
        usage = item.get("usage")
        if usage:
            result["prompt_tokens_reported"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
        for choice in item.get("choices", []):
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
            text = choice.get("text") or choice.get("delta", {}).get("content") or ""
            if text:
                text_seen = True
                if result["ttft_ms"] is None:
                    result["ttft_ms"] = elapsed

    try:
        async with session.post(endpoint + "/v1/completions", json=body) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raw.extend(await response.read())
                raise RuntimeError(f"HTTP {response.status}: {bytes(raw)[:500]!r}")
            async for line in response.content:
                raw.extend(line)
                line = line.decode("utf-8").rstrip("\r\n")
                if not line:
                    consume()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
        valid = (not data_lines and done and text_seen
                 and result["finish_reason"] == "length"
                 and result["completion_tokens"] == 32
                 and result["prompt_tokens_reported"] == case["prompt_tokens"])
        if not valid:
            raise RuntimeError(f"unexpected output/usage: {result}")
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["e2e_ms"] = (time.perf_counter() - start) * 1000
    stem.with_suffix(".sse").write_bytes(raw)
    write_json(stem.with_suffix(".events.json"), events)
    write_json(stem.with_suffix(".response.json"), result)
    return result


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir()
    manifest = json.loads(args.manifest.read_text())
    rows = []
    with (args.output / "requests.csv").open("x", newline="") as handle:
        fields = ["transport", "repeat", "id", "input_tokens", "kv_bytes_estimate",
                  "success", "prompt_tokens_reported", "completion_tokens", "ttft_ms",
                  "e2e_ms", "finish_reason", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        async with aiohttp.ClientSession(trust_env=False,
                timeout=aiohttp.ClientTimeout(total=1200)) as session:
            for cell in manifest["cells"]:
                for case in cell["cases"]:
                    stem = args.output / "raw" / case["id"]
                    row = await request(session, args.endpoint, case, stem)
                    row.update(transport=args.transport, repeat=case["repeat"],
                               kv_bytes_estimate=case["prompt_tokens"] * 57344)
                    rows.append(row)
                    writer.writerow({k: row.get(k) for k in fields})
                    handle.flush()
                    print(json.dumps({k: row.get(k) for k in fields}), flush=True)
    good = [r for r in rows if r["success"]]
    summary = []
    for n in (cell["input_tokens"] for cell in manifest["cells"]):
        values = [r["e2e_ms"] for r in good if r["input_tokens"] == n]
        summary.append({"input_tokens": n, "requests": len(values),
                        "e2e_median_ms": statistics.median(values) if values else None,
                        "e2e_mean_ms": statistics.mean(values) if values else None})
    write_json(args.output / "summary.json", {"transport": args.transport,
        "requests": len(rows), "successful": len(good), "groups": summary})
    if len(good) != len(rows):
        raise RuntimeError(f"{len(rows) - len(good)} request(s) failed; see retained raw evidence")


async def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--decode", default="http://192.168.1.11:8200")
    prepare.add_argument("--namespace", default="phase9-v1")
    prepare.add_argument("--input-sizes", type=int, nargs="+", default=list(SIZES))
    prepare.add_argument("--repeats", type=int, default=5)
    execute = sub.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--endpoint", default="http://192.168.1.9:10001")
    execute.add_argument("--transport", choices=("socket", "rdma"), required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.repeats < 1 or any(size < 64 for size in args.input_sizes):
            parser.error("repeats must be positive and input sizes must be at least 64")
        if len(args.input_sizes) != len(set(args.input_sizes)):
            parser.error("input sizes must be unique")
        await make_manifest(args)
    else:
        await run(args)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Small streaming client for either standalone vLLM or the PPD Proxy.

No server management or inference retry. Full request/SSE evidence is retained.
Requires aiohttp only; can run using the existing runtime image without a GPU.
"""
import argparse
import asyncio
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid

import aiohttp


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


async def stream(session, url, payload, stem):
    save(Path(str(stem) + ".request.json"), payload)
    row = dict(success=False, error=None, http_status=None, response_id=None,
               prompt_tokens=None, completion_tokens=None, finish_reason=None,
               ttft_ms=None, tpot_ms=None, content_tpot_ms=None, e2e_ms=None, text="")
    events, raw, data, texts = [], bytearray(), [], []
    done, last_content = False, None
    start = time.perf_counter()

    def event():
        nonlocal done, last_content
        if not data:
            return
        value = "\n".join(data)
        data.clear()
        elapsed = (time.perf_counter() - start) * 1000
        if done:
            raise ValueError("Data after DONE")
        obj = value if value == "[DONE]" else json.loads(value)
        events.append({"elapsed_ms": elapsed, "data": obj})
        if obj == "[DONE]":
            done = True
            return
        if "error" in obj:
            raise ValueError("SSE error: " + str(obj["error"]))
        if obj.get("id"):
            row["response_id"] = obj["id"]
        if obj.get("usage"):
            row["usage"] = obj["usage"]
        for choice in obj.get("choices", []):
            text = choice.get("text") or choice.get("delta", {}).get("content") or ""
            if text:
                if row["ttft_ms"] is None:
                    row["ttft_ms"] = elapsed
                last_content = elapsed
                texts.append(text)
            if choice.get("finish_reason") is not None:
                row["finish_reason"] = choice["finish_reason"]

    try:
        async with session.post(url.rstrip("/") + "/v1/completions", json=payload) as r:
            row["http_status"] = r.status
            if r.status != 200 or "text/event-stream" not in r.headers.get("Content-Type", ""):
                raw.extend(await r.read())
                raise ValueError(f"HTTP {r.status}: {bytes(raw)[:500]!r}")
            async for line in r.content:
                raw.extend(line)
                line = line.decode("utf-8").rstrip("\r\n")
                if not line:
                    event()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip(" "))
            if data:
                raise ValueError("Unterminated SSE event")
        row["e2e_ms"] = (time.perf_counter() - start) * 1000
        usage = row.get("usage") or {}
        row["prompt_tokens"] = usage.get("prompt_tokens")
        n = row["completion_tokens"] = usage.get("completion_tokens")
        if not done or row["ttft_ms"] is None or not isinstance(n, int) or n < 1 or row["finish_reason"] is None:
            raise ValueError("Missing DONE, text, usage or finish reason")
        if n != payload["max_tokens"] or row["finish_reason"] != "length":
            raise ValueError("Unexpected output length or finish reason")
        if n > 1:
            row["tpot_ms"] = (row["e2e_ms"] - row["ttft_ms"]) / (n - 1)
            row["content_tpot_ms"] = (last_content - row["ttft_ms"]) / (n - 1)
        row["success"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["e2e_ms"] = row["e2e_ms"] or (time.perf_counter() - start) * 1000
    row["text"] = "".join(texts)
    Path(str(stem) + ".sse").write_bytes(raw)
    save(Path(str(stem) + ".events.json"), events)
    save(Path(str(stem) + ".response.json"), row)
    return row


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "raw").mkdir()
    namespace = uuid.uuid4().hex
    save(args.output / "run.json", dict(url=args.url, model=args.model,
        prompt=args.prompt, append=args.append, namespace=namespace,
        conversations=args.conversations, concurrency=args.concurrency,
        max_tokens=args.max_tokens, warmup=False, scheduling="closed-loop conversations",
        sampling={"temperature": 0, "seed": 42, "ignore_eos": True},
        prompt_format="User: Session <unique ID>. <prompt>\\nAssistant:",
        client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tpot="(SSE EOF - first nonempty text)/(completion_tokens-1)"))
    rows = []
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(force_close=True, limit=max(100, args.concurrency))
    with (args.output / "requests.jsonl").open("x") as journal:
        async with aiohttp.ClientSession(trust_env=False, timeout=timeout, connector=connector) as session:
            async def conversation(i):
                async with semaphore:
                    prompt = f"User: Session {namespace}-{i}. {args.prompt}\nAssistant:"
                    for turn in range(1, 3 if args.append is not None else 2):
                        name = f"c{i:04d}-t{turn}"
                        payload = dict(model=args.model, prompt=prompt, max_tokens=args.max_tokens,
                            temperature=0, seed=42, ignore_eos=True, stream=True,
                            stream_options={"include_usage": True})
                        result = await stream(session, args.url, payload, args.output / "raw" / name)
                        row = dict(result, conversation=i, turn=turn, raw_stem="raw/" + name,
                            payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest())
                        rows.append(row)
                        journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                        journal.flush()
                        print(json.dumps({k: row[k] for k in ("conversation", "turn", "success", "ttft_ms", "tpot_ms", "error")}), flush=True)
                        if not row["success"]:
                            break
                        prompt += row["text"] + "\nUser: " + (args.append or "") + "\nAssistant:"
            outcomes = await asyncio.gather(*(conversation(i) for i in range(args.conversations)), return_exceptions=True)
    fields = ["conversation", "turn", "success", "prompt_tokens", "completion_tokens", "ttft_ms",
              "tpot_ms", "content_tpot_ms", "e2e_ms", "response_id", "finish_reason", "raw_stem", "error"]
    with (args.output / "requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["conversation"], r["turn"])))
    expected = args.conversations * (2 if args.append is not None else 1)
    errors = [str(r) for r in outcomes if isinstance(r, BaseException)]
    passed = len(rows) == expected and all(r["success"] for r in rows) and not errors
    groups = []
    for turn in sorted({r["turn"] for r in rows}):
        good = [r for r in rows if r["turn"] == turn and r["success"]]
        group = dict(turn=turn, successful=len(good))
        for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
            values = [r[key] for r in good if r[key] is not None]
            group[key] = {"mean": statistics.mean(values), "median": statistics.median(values)} if values else None
        groups.append(group)
    save(args.output / "summary.json", dict(passed=passed, expected_requests=expected,
        recorded_requests=len(rows), successful=sum(r["success"] for r in rows), errors=errors, groups=groups))
    return passed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True, help="Base URL, without /v1/completions")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", default="Explain prefix caching in simple terms.")
    prompt.add_argument("--prompt-file", type=Path)
    p.add_argument("--append", help="Second question; omit for single-turn experiments")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--conversations", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output", type=Path, required=True, help="Must not already exist")
    args = p.parse_args()
    if min(args.conversations, args.concurrency, args.max_tokens) < 1 or args.timeout <= 0:
        p.error("Counts and timeout must be positive")
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text()
    raise SystemExit(0 if asyncio.run(run(args)) else 1)


if __name__ == "__main__":
    main()

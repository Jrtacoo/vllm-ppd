#!/usr/bin/env python3
"""Two full-history HTTP turns and proxy routing checks against running nodes.

This sends real inference requests when invoked. It does not start/stop servers.
Routing/response checks alone do not prove KV transfer or a prefix-cache hit.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import ProxyHandler, Request, build_opener
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ppd.benchmark_options import registry_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=("1P_1D", "1P_1pD"))
    parser.add_argument("--proxy-url", default="http://192.168.1.9:10001")
    parser.add_argument("--api-model-name", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True, help="Save raw responses and routing evidence")
    args = parser.parse_args()
    # Internal HTTP must bypass inherited outbound proxies.
    opener = build_opener(ProxyHandler({}))
    report = {"config": args.config, "api_model_name": args.api_model_name,
              "proxy_url": args.proxy_url, "response_and_routing_checks": "failed",
              "kv_transfer": "unverified; inspect connector/NCCL evidence",
              "prefix_cache_hit": "unverified; inspect consumer metrics",
              "rdma": "unverified"}

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(args.proxy_url.rstrip("/") + path, data=data,
                      headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=180) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def turn(messages):
        response = request("/v1/chat/completions", {
            "model": args.api_model_name, "messages": messages,
            "temperature": 0, "max_tokens": 32, "stream": False})
        report.setdefault("responses", []).append(response)
        text = response["choices"][0]["message"]["content"]
        if not text or response.get("usage", {}).get("completion_tokens", 0) <= 0:
            raise RuntimeError("No generated content/token usage")
        return text

    try:
        initial = request("/status")
        error = registry_error(initial, args.config)
        if error:
            raise RuntimeError(error)
        # A separate conversation establishes connections before measurement.
        turn([{"role": "user", "content": f"Warmup {uuid.uuid4().hex}. Say ready."}])
        report["before"] = request("/status")
        history = [{"role": "user", "content":
                    f"Test {uuid.uuid4().hex}. Remember that the test color is blue. Reply briefly."}]
        history.append({"role": "assistant", "content": turn(history)})
        history.append({"role": "user", "content": "What test color did I give you? Reply briefly."})
        report["second_turn_full_history"] = list(history)
        turn(history)
        report["after"] = request("/status")
        error = registry_error(report["after"], args.config)
        if error:
            raise RuntimeError(error)
        delta = {key: report["after"]["stats"].get(key, 0) - value
                 for key, value in report["before"]["stats"].items()}
        report["stats_delta"] = delta
        expected = {"total_requests": 2, "turn1_requests": 1, "turn2plus_requests": 1,
                    "pd_requests": 2 if args.config == "1P_1D" else 0,
                    "ppd_requests": 1 if args.config == "1P_1pD" else 0,
                    "ppd_direct_requests": 1 if args.config == "1P_1pD" else 0}
        if any(delta.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Routing counters differ from {expected}; run without other clients")
        report["response_and_routing_checks"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["timestamp_unix"] = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("responses", "before", "after", "second_turn_full_history")},
                     ensure_ascii=False, indent=2))
    return 0 if report["response_and_routing_checks"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

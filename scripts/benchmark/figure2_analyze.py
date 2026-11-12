#!/usr/bin/env python3
"""Audit actual cache and mixed scheduling, then aggregate Figure 2 trials."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SCENARIOS = ("decode_only", "full_prefill", "append_prefill")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def describe(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values), "max": max(values), "n": len(values)} if values else None


def observer_summary(values):
    summary = describe(values)
    if summary is None:
        return None
    ordered = sorted(values)
    for label, fraction in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        index = (len(ordered) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        summary[label] = ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    summary["over_1000us_steps"] = sum(v > 1000 for v in values)
    return summary


def analyze(evidence, run):
    folder = evidence / run
    trials = read_jsonl(folder / "trials.jsonl")
    requests = read_jsonl(folder / "requests.jsonl")
    runmeta = json.loads((folder / "run.json").read_text())
    decode_history = runmeta["decode_cached_tokens"]
    prefill_history = runmeta["append_prefill_cached_tokens"]
    new_tokens = runmeta["append_prefill_new_tokens"]
    full_tokens = runmeta.get("full_prefill_computed_tokens", runmeta["full_prefill_new_tokens"])
    full_prompt_tokens = runmeta.get("full_prefill_prompt_tokens", full_tokens)
    append_prompt_tokens = prefill_history + new_tokens
    max_chunks = runmeta.get("max_prefill_chunks", 1)
    request_ids = {r["request_id"] for r in requests}
    by_trial = {t["trial"]: t for t in trials}
    observations = {k: {"decode_batch_counts": Counter(), "mixed_steps": [], "decode_step_ms": [],
                        "preemptions": 0, "interferer_tokens": 0, "observer_us": []} for k in by_trial}
    observer_costs, step_count, run_step_count = [], 0, 0
    trace = evidence / "scheduler-trace.jsonl"
    with trace.open() as f:
        for line in f:
            record = json.loads(line)
            if "observer_step" in record:
                observer_costs.append(record["observer_us"])
                continue
            # vLLM completions adds a sequence suffix (e.g. cmpl-<request_id>-0).
            belongs_to_run = any(
                r["id"].removeprefix("cmpl-") in request_ids
                or r["id"].removeprefix("cmpl-").rsplit("-", 1)[0] in request_ids
                for r in record["requests"])
            if belongs_to_run:
                run_step_count += 1
            if "observer_us" in record and belongs_to_run:
                observer_costs.append(record["observer_us"])
            step_count += 1
            relevant = defaultdict(list)
            for r in record["requests"]:
                for role in ("decode", "interferer"):
                    marker = "_" + role + "_"
                    if marker in r["id"]:
                        trial = r["id"].removeprefix("cmpl-").rsplit(marker, 1)[0]
                        if trial in observations:
                            relevant[trial].append(dict(r, role=role))
            for trial, rs in relevant.items():
                o = observations[trial]
                if "observer_us" in record:
                    o["observer_us"].append(record["observer_us"])
                decodes = [r for r in rs if r["role"] == "decode" and r["computed_before"] >= r["prompt_tokens"] and r["scheduled_tokens"] == 1]
                prefills = [r for r in rs if r["role"] == "interferer" and r["computed_before"] < r["prompt_tokens"]]
                o["preemptions"] += len(record["preempted_ids"])
                if decodes:
                    o["decode_batch_counts"][len(decodes)] += 1
                if prefills:
                    o["interferer_tokens"] += sum(r["scheduled_tokens"] for r in prefills)
                    o["mixed_steps"].append({"step": record["step"], "decode_count": len(decodes),
                        "decode_output_tokens_before": [r["output_tokens_before"] for r in decodes],
                        "interferers": prefills, "model_wall_ms": record["model_wall_ms"],
                        "observer_us": record.get("observer_us")})
                elif len(decodes) == by_trial[trial]["batch_size"]:
                    o["decode_step_ms"].append(record["model_wall_ms"])
    raw_ok = True
    raw_usage_ok, event_timings_ok = True, True
    prompt_groups = defaultdict(set)
    new_prefill_groups = defaultdict(set)
    complete_prefill_groups = defaultdict(set)
    background_histories = defaultdict(set)
    background_suffixes = defaultdict(set)
    warmup_prefixes, prefix_references = {}, []
    for r in requests:
        stem = folder / r["raw_stem"]
        raw_ok &= all(Path(str(stem) + s).is_file() for s in (".request.json", ".response.json", ".events.json", ".sse"))
        raw_ok &= b'data: [DONE]' in Path(str(stem) + '.sse').read_bytes()
        payload = json.loads(Path(str(stem) + '.request.json').read_text())
        response = json.loads(Path(str(stem) + '.response.json').read_text())
        raw_usage_ok &= (response["success"] and response["usage"]["prompt_tokens"] == len(payload["prompt"])
                         and response["usage"]["completion_tokens"] == payload["max_tokens"])
        if r["role"] == "warmup":
            expected_prompt = prefill_history if int(r["request_id"].rsplit("_", 1)[1]) == r["batch_size"] else decode_history
        elif r["role"] == "decode":
            expected_prompt = decode_history + runmeta["decode_new_tokens"]
        else:
            expected_prompt = full_prompt_tokens if r["scenario"] == "full_prefill" else append_prompt_tokens
        raw_usage_ok &= len(payload["prompt"]) == expected_prompt
        raw_usage_ok &= payload["max_tokens"] == (runmeta["decode_output_tokens"] if r["role"] == "decode" else 1)
        if r["role"] == "warmup":
            warmup_prefixes[(r["trial"], int(r["request_id"].rsplit("_", 1)[1]))] = (
                hashlib.sha256(json.dumps(payload["prompt"]).encode()).hexdigest(), payload["cache_salt"])
        elif r["role"] == "decode":
            seq = int(r["request_id"].rsplit("_", 1)[1])
            warm_index = 0 if runmeta.get("prefix_mode") == "shared" else seq
            prefix_references.append((r["trial"], warm_index,
                hashlib.sha256(json.dumps(payload["prompt"][:decode_history]).encode()).hexdigest(),
                payload["cache_salt"], True))
        elif r["role"] == "interferer" and (r["scenario"] == "append_prefill" or runmeta.get("full_prefill_mode") == "same-context"):
            prefix_references.append((r["trial"], r["batch_size"],
                hashlib.sha256(json.dumps(payload["prompt"][:prefill_history]).encode()).hexdigest(),
                payload["cache_salt"], r["scenario"] == "append_prefill"))
        if r["role"] == "interferer":
            new_prefill_groups[(r["batch_size"], r["repeat"])].add(
                hashlib.sha256(json.dumps(payload["prompt"][-new_tokens:]).encode()).hexdigest())
            complete_prefill_groups[(r["batch_size"], r["repeat"])].add(
                hashlib.sha256(json.dumps(payload["prompt"]).encode()).hexdigest())
        if r["role"] == "decode":
            background_histories[r["trial"]].add((payload["cache_salt"], tuple(payload["prompt"][:decode_history])))
            background_suffixes[r["trial"]].add(tuple(payload["prompt"][decode_history:]))
            events = json.loads(Path(str(stem) + '.events.json').read_text())
            token_events = [(e["elapsed_ms"], c["token_ids"]) for e in events if isinstance(e["data"], dict)
                            for c in e["data"].get("choices", []) if c.get("token_ids")]
            n = sum(len(ids) for _, ids in token_events)
            event_timings_ok &= n == r["completion_tokens"] and math.isclose(
                (token_events[-1][0] - token_events[0][0]) / (n - 1), r["token_tpot_ms"], abs_tol=1e-12)
            suffix = payload["request_id"].rsplit("_decode_", 1)[1]
            prompt_groups[(r["batch_size"], r["repeat"], suffix)].add(hashlib.sha256(json.dumps(payload["prompt"]).encode()).hexdigest())
    prefix_links_ok = all(
        warmup_prefixes[(trial, index)][0] == prompt_hash and
        ((warmup_prefixes[(trial, index)][1] == salt) == reuse)
        for trial, index, prompt_hash, salt, reuse in prefix_references)
    audits = []
    for trial, t in by_trial.items():
        o = observations[trial]
        baseline = t["scenario"] == "decode_only"
        decode_rows = [r for r in requests if r["trial"] == trial and r["role"] == "decode"]
        checks = dict(t["checks"])
        checks.update({
            "observed_full_decode_batch": t["batch_size"] == 0 or t["batch_size"] in o["decode_batch_counts"],
            "no_preemptions": o["preemptions"] == 0,
            "client_mean_recomputed": (t["batch_size"] == 0 and t["decode_avg_tpot_ms"] is None) or (bool(decode_rows) and math.isclose(
                statistics.mean(r["token_tpot_ms"] for r in decode_rows), t["decode_avg_tpot_ms"], abs_tol=1e-12)),
        })
        if baseline:
            checks["no_interferer"] = not o["mixed_steps"]
        else:
            checks.update({
                "exact_computed_prefill_tokens": o["interferer_tokens"] == (full_tokens if t["scenario"] == "full_prefill" else new_tokens),
                "prefill_chunk_budget": 1 <= len(o["mixed_steps"]) <= (max_chunks if t["scenario"] == "full_prefill" else 1),
                "actual_mixed_batch": bool(o["mixed_steps"]) and all(s["decode_count"] == t["batch_size"] for s in o["mixed_steps"]),
                "decode_already_generating": bool(o["mixed_steps"]) and all(n >= 8 for s in o["mixed_steps"] for n in s["decode_output_tokens_before"]),
                "prefill_actual_cache": all(r["cached_tokens"] == (prefill_history if t["scenario"] == "append_prefill" else 0)
                    for s in o["mixed_steps"] for r in s["interferers"]),
            })
        audits.append({"trial": trial, "scenario": t["scenario"], "batch_size": t["batch_size"],
            "checks": checks, "passed": all(checks.values()),
            "decode_batch_counts": dict(sorted(o["decode_batch_counts"].items())),
            "mixed_steps": o["mixed_steps"], "decode_only_step_wall_ms": describe(o["decode_step_ms"])})
        audits[-1]["observer_us"] = observer_summary(o["observer_us"])
    groups = []
    prefill_only_groups = []
    lookup = {(t["batch_size"], t["repeat"], t["scenario"]): t for t in trials}
    for batch in sorted({t["batch_size"] for t in trials}):
        if batch == 0:
            for scenario in SCENARIOS[1:]:
                rs = [r for r in requests if r["batch_size"] == 0 and r["scenario"] == scenario and r["role"] == "interferer"]
                aa = [a for a in audits if a["batch_size"] == 0 and a["scenario"] == scenario]
                prefill_only_groups.append({"batch_size": 0, "scenario": scenario,
                    "ttft_ms": describe([r["ttft_ms"] for r in rs]),
                    "model_wall_ms": describe([sum(s["model_wall_ms"] for s in a["mixed_steps"]) for a in aa]),
                    "note": "No background decode; decode TPOT and slowdown are undefined"})
            continue
        for scenario in SCENARIOS:
            ts = [t for t in trials if t["batch_size"] == batch and t["scenario"] == scenario]
            values = [t["decode_avg_tpot_ms"] for t in ts]
            relative = [(t["decode_avg_tpot_ms"] / lookup[(batch, t["repeat"], "decode_only")]["decode_avg_tpot_ms"] - 1) * 100 for t in ts]
            mixed_wall = [s["model_wall_ms"] for a in audits if a["batch_size"] == batch and a["scenario"] == scenario for s in a["mixed_steps"]]
            steady_wall = [a["decode_only_step_wall_ms"]["mean"] for a in audits if a["batch_size"] == batch and a["scenario"] == scenario and a["decode_only_step_wall_ms"]]
            groups.append({"batch_size": batch, "scenario": scenario, "decode_tpot_ms": describe(values),
                           "slowdown_percent_vs_same_repeat_baseline": describe(relative),
                           "mixed_step_wall_ms": describe(mixed_wall), "steady_decode_step_wall_ms": describe(steady_wall),
                           "trial_values_ms": values,
                           "observer_us": observer_summary([v for t in ts for v in observations[t["trial"]]["observer_us"]])})
    prefix = runmeta.get("trial_prefix", run)
    expected_ids = {f"{prefix}-b{t['batch_size']}-r{t['repeat']}-{t['scenario']}" for t in runmeta["schedule"]}
    expected_roles = Counter()
    for t in runmeta["schedule"]:
        expected_roles.update(warmup=2 if runmeta.get("prefix_mode") == "shared" and t["batch_size"] else t["batch_size"] + 1, decode=t["batch_size"],
                              interferer=int(t["scenario"] != "decode_only"))
    report = {"run": run, "requests": len(requests), "request_roles": dict(Counter(r["role"] for r in requests)),
        "trials": len(trials), "expected_trials": len(runmeta["schedule"]),
        "prefix_mode": runmeta.get("prefix_mode", "independent"),
        "checks": {"all_expected_trials_finished": len(trials) == len(runmeta["schedule"]),
                   "exact_trial_id_set": set(by_trial) == expected_ids,
                   "unique_request_ids": len(request_ids) == len(requests),
                   "exact_request_role_counts": Counter(r["role"] for r in requests) == expected_roles,
                   "all_requests_successful": all(r["success"] for r in requests),
                   "raw_files_complete": bool(raw_ok), "raw_usage_and_exact_prompts": bool(raw_usage_ok),
                   "token_event_timings_recomputed": bool(event_timings_ok),
                   "identical_decode_prompts_across_scenarios": all(len(v) == 1 for v in prompt_groups.values()),
                   "identical_new_prefill_across_scenarios": all(len(v) == 1 for v in new_prefill_groups.values()),
                   "identical_complete_prefill_prompts": runmeta.get("full_prefill_mode") != "same-context" or all(len(v) == 1 for v in complete_prefill_groups.values()),
                   "histories_match_warmups_and_cache_salts_control_reuse": prefix_links_ok,
                   "background_prefix_isolation_or_sharing": all(
                       len(background_histories[t["trial"]]) == (1 if runmeta.get("prefix_mode") == "shared" and t["batch_size"] else t["batch_size"])
                       for t in trials),
                   "shared_mode_unique_new_suffixes": runmeta.get("prefix_mode") != "shared" or all(
                       len(background_suffixes[t["trial"]]) == t["batch_size"] for t in trials),
                   "all_trial_checks": all(a["passed"] for a in audits)},
        "trial_audits": audits, "groups": groups, "prefill_only_groups": prefill_only_groups,
        "observer": {"trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                     "total_steps_so_far": step_count, "steps_for_this_run": run_step_count,
                     "cost_scope": "steps containing this run's request IDs, including warmup",
                     "capture_store_us": observer_summary(observer_costs)},
        "limitations": ["RTX 4090/Qwen 7B adaptation; original Figure 2 uses H100/Llama 8B.",
            f"One injected prefill, {runmeta['decode_output_tokens']} output tokens per background decode; full-trace average dilutes a one-step disturbance.",
            f"Full computes {full_tokens} tokens; append computes {new_tokens} tokens and reuses {prefill_history} cached tokens. Mode: {runmeta.get('full_prefill_mode', 'new-only')}.",
            "Error bars are sample standard deviations across independent repeated trials, not confidence intervals.",
            "Scheduler observer uses the default scheduler unchanged; observed engine-step wall time is not CUDA kernel time."]}
    if runmeta.get("prefix_mode") == "shared":
        report["limitations"].append("Background histories share physical KV, with distinct four-token suffixes. This is not independent-KV capacity or scaling.")
    report["passed"] = all(report["checks"].values())
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    with (folder / "requests.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        decode_rows = [r for r in reader if r["role"] == "decode"]
    with (folder / "decode-requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decode_rows)
    with (folder / "summary.csv").open("w", newline="") as f:
        fields = ["batch_size", "scenario", "repeats", "mean_tpot_ms", "std_tpot_ms", "mean_slowdown_percent", "std_slowdown_percent"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g in groups:
            a, b = g["decode_tpot_ms"], g["slowdown_percent_vs_same_repeat_baseline"]
            w.writerow(dict(batch_size=g["batch_size"], scenario=g["scenario"], repeats=a["n"],
                mean_tpot_ms=a["mean"], std_tpot_ms=a["std"], mean_slowdown_percent=b["mean"], std_slowdown_percent=b["std"]))
    print(json.dumps({k: v for k, v in report.items() if k not in ("trial_audits", "observer", "limitations")}, indent=2))
    return report["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--run", required=True)
    args = p.parse_args()
    raise SystemExit(not analyze(args.evidence, args.run))

#!/usr/bin/env python3
"""Plot the audited Phase 8 Figure 2 adaptation, or its portable snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCENARIOS = ("decode_only", "full_prefill", "append_prefill")
LABELS = ("Decode only", "+ Full prefill", "+ Append prefill")
COLORS = ("#6B7C93", "#D46A37", "#008577")
MARKERS = ("o", "s", "D")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-snapshot", action="store_true")
    p.add_argument("--evidence-dir", type=Path, default=ROOT / "artifacts/two-node-phase8-figure2-evidence")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("--run", default="formal")
    p.add_argument("--snapshot", type=Path, default=HERE / "plot-data.json")
    p.add_argument("--linear-x", action="store_true")
    p.add_argument("--figure-label", default="PHASE 8 / FIGURE 2 ADAPTATION")
    args = p.parse_args()
    if args.from_snapshot:
        data = json.loads(args.snapshot.read_text())
    else:
        path = args.evidence_dir / args.run / "verification.json"
        report = json.loads(path.read_text())
        if not report["passed"]:
            raise ValueError("Formal request/cache/scheduler audit has not passed")
        data = {"source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "groups": report["groups"], "checks": report["checks"],
                "formal_trials": report["trials"], "request_roles": report["request_roles"],
                "limitations": report["limitations"], "prefix_mode": report.get("prefix_mode", "independent"),
                "observer_us": report.get("observer", {}).get("capture_store_us")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.from_snapshot:
        (args.output_dir / "plot-data.json").write_text(json.dumps(data, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#172D46", "axes.labelcolor": "#41556C", "xtick.color": "#41556C",
        "ytick.color": "#41556C", "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#C7D3E0", "svg.fonttype": "none", "svg.hashsalt": "ppd-figure2",
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))
    fig.subplots_adjust(left=0.075, right=0.965, top=0.76, bottom=0.23, wspace=0.23)
    fig.text(0.055, 0.97, args.figure_label, color="#008577", fontsize=10, weight="bold", va="top")
    fig.text(0.055, 0.925, "Prefill interference on an active decode batch", fontsize=22, weight="bold", va="top")
    mode = data.get("prefix_mode", "independent")
    subtitle = "Single RTX 4090 | Qwen2.5-7B-Instruct | 5 repeats per point | " + (
        ("Shared background history KV" if mode == "shared" else "Independent background history KV")
        if args.linear_x or mode == "shared" else "1 injected prefill request")
    fig.text(0.055, 0.862, subtitle, color="#607286", va="top")
    handles = [Line2D([0], [0], color=c, marker=m, label=l, linewidth=2) for c, m, l in zip(COLORS, MARKERS, LABELS)]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.99), ncol=3, frameon=False, fontsize=9)
    batches = sorted({g["batch_size"] for g in data["groups"]})
    lookup = {(g["batch_size"], g["scenario"]): g for g in data["groups"]}
    for si, (scenario, color, marker) in enumerate(zip(SCENARIOS, COLORS, MARKERS)):
        groups = [lookup[(b, scenario)] for b in batches]
        for ax, field in zip(axes, ("decode_tpot_ms", "slowdown_percent_vs_same_repeat_baseline")):
            means = [g[field]["mean"] for g in groups]
            stds = [g[field]["std"] for g in groups]
            if any(g[field]["n"] != 5 for g in groups):
                raise ValueError("Expected five independently repeated trials per point")
            offset = (si - 1) * 0.03
            xs = np.asarray(batches, dtype=float) if args.linear_x else np.asarray(batches, dtype=float) * (2 ** offset)
            ax.errorbar(xs, means, yerr=stds, color=color, marker=marker, linewidth=2,
                        markersize=6, capsize=4, elinewidth=1.3, zorder=4)
            if field == "decode_tpot_ms":
                for x, g in zip(xs, groups):
                    values = g["trial_values_ms"]
                    jitter = x + np.linspace(-0.3, 0.3, len(values)) if args.linear_x else x * (2 ** np.linspace(-0.025, 0.025, len(values)))
                    ax.scatter(jitter, values,
                               color=color, s=15, alpha=0.35, linewidths=0, zorder=3)
    for ax in axes:
        if args.linear_x:
            ticks = [1, 50, 100, 150, 200, 250] if max(batches) >= 200 else batches
            ax.set_xticks(ticks, [str(b) for b in ticks])
            ax.set_xlim(min(batches) - max(batches)*0.02, max(batches)*1.025)
        else:
            ax.set_xscale("log", base=2)
            ax.set_xticks(batches, [str(b) for b in batches])
            ax.set_xlim(min(batches) / 1.18, max(batches) * 1.18)
        ax.grid(color="#E8EDF3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=8)
        ax.set_xlabel("Observed decode batch size" + ("" if args.linear_x else " (log2 scale)"), labelpad=12)
    axes[0].set_title("A   Mean decode TPOT", loc="left", weight="bold", pad=16)
    axes[0].set_ylabel("ms / output token", labelpad=12)
    axes[1].set_title("B   Change relative to decode only", loc="left", weight="bold", pad=16)
    axes[1].set_ylabel("TPOT change (%)", labelpad=12)
    axes[1].axhline(0, color="#99A8B8", linestyle="--", linewidth=1)
    notes = [
        "Mean of trial means; error bars = sample SD across 5 repeats. Faint dots = individual trials. Lines are slightly offset horizontally for readability.",
        "Full: 0 cached + 1024 new tokens. Append: 2048 cached + 1024 new tokens. Decode: 2048 cached + 4 new, then 128 output tokens.",
        "Injection occurs after every decode emits >=8 tokens. Scheduler evidence verifies actual batch size and mixed execution; no KV network transfer is involved.",
        "This is a hardware/model adaptation of Figure 2, not the original H100/Llama result. TPOT averages cover the full decode stream; one-step effects are diluted.",
    ]
    if args.linear_x:
        notes[0] = "Mean of trial means; error bars = sample SD across 5 repeats. Faint dots = individual trials. Curves use the actual measured batch sizes."
    if mode == "shared":
        notes.append("Background sequences share the same 2048-token history KV, with distinct 4-token suffixes and separate generation. This is not independent-KV scaling.")
    observer = data.get("observer_us")
    if args.linear_x and observer and observer["max"] > 5000:
        notes.append(f"Observer overhead: P99 {observer.get('p99', observer['max'])/1000:.2f} ms, max {observer['max']/1000:.2f} ms. Occasional stalls are retained; small differences are not significance claims.")
    for i, text in enumerate(reversed(notes)):
        fig.text(0.055, 0.025 + i * 0.022, text, fontsize=8.4, color="#607286")
    for ext in ("png", "svg", "pdf"):
        path = args.output_dir / ("figure2-interference." + ext)
        fig.savefig(path, dpi=220, metadata={"Creator": "PPD Figure 2 adaptation"})
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()

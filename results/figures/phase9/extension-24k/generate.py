#!/usr/bin/env python3
"""Render the audited 24K Phase 9 snapshot without accessing model services."""
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

HERE = Path(__file__).resolve().parent
TRANSPORTS = ("socket", "rdma")
COLORS = ("#426CB4", "#009B83")
METRICS = (
    ("tx_nccl_sync_sum_ms", "A", "Synchronized send calls", "Sum over 28 layers; includes synchronization"),
    ("rx_nccl_sync_sum_ms", "B", "Synchronized receive calls", "Sum over 28 layers; includes peer waiting"),
    ("rx_receive_span_ms", "C", "KV receive span", "First PUT metadata to last completed receive"),
    ("tx_send_span_ms", "D", "KV send span", "First send entry to last completed send"),
    ("ttft_ms", "E", "Time to first text", "Client POST to first nonempty text"),
    ("e2e_ms", "F", "End-to-end latency", "Client POST to SSE EOF; 32 output tokens"),
)


def validate(data):
    rows = data["requests"]
    if len(rows) != 10 or not data["verification"]["passed"]:
        raise ValueError("Expected 10 audited requests")
    for repeat in range(1, 6):
        pair = [r for r in rows if r["repeat"] == repeat]
        if len(pair) != 2 or {r["transport"] for r in pair} != set(TRANSPORTS):
            raise ValueError(f"Incomplete pair {repeat}")
        if pair[0]["payload_sha256"] != pair[1]["payload_sha256"]:
            raise ValueError(f"Mismatched input pair {repeat}")
    for row in rows:
        if not (row["success"] and row["input_tokens"] == 24576
                and row["completion_tokens"] == 32 and row["layers"] == 28
                and row["payload_bytes"] == 1409286144):
            raise ValueError("Unexpected request configuration or tensor payload")
        for key, *_ in METRICS:
            if not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"Invalid timing: {key}")
    return rows


def render(rows):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#20324B", "axes.labelcolor": "#53667C",
        "xtick.color": "#3F526A", "ytick.color": "#53667C",
        "axes.edgecolor": "#CBD5E1", "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "svg.fonttype": "none",
        "pdf.fonttype": 42, "svg.hashsalt": "phase9-24k",
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.4))
    fig.subplots_adjust(left=.065, right=.97, bottom=.17, top=.72,
                        wspace=.30, hspace=.64)
    fig.text(.045, .963, "PHASE 9  /  24K INPUT EXTENSION", fontsize=10,
             weight="bold", color="#008577")
    fig.text(.045, .916, "Large KV payload: Socket / TCP vs RoCE",
             fontsize=24, weight="bold")
    fig.text(.045, .877,
             "2 × RTX 4090  ·  Qwen2.5-7B-Instruct BF16  ·  100 Gb/s  ·  Serial PD  ·  RoCE with host staging",
             fontsize=11, color="#53667C")
    cards = ((.045, "24,576", "input tokens"), (.28, "1,344 MiB", "KV payload / request"),
             (.54, "5 matched pairs", "all 10 requests successful"),
             (.79, "32 tokens", "fixed output length"))
    for x, value, description in cards:
        fig.text(x, .826, value, fontsize=16, weight="bold")
        fig.text(x, .800, description, fontsize=10, color="#64748B")

    summary = {}
    for ax, (key, letter, title, subtitle) in zip(axes.flat, METRICS):
        groups = [[r for r in rows if r["transport"] == t] for t in TRANSPORTS]
        groups = [sorted(group, key=lambda r: r["repeat"]) for group in groups]
        values = [[r[key] for r in group] for group in groups]
        medians = [statistics.median(v) for v in values]
        maximum = max(max(v) for v in values)
        ceiling = maximum * 1.30
        for i in range(5):
            offset = (i - 2) * .036
            ax.plot([offset, 1 + offset], [values[0][i], values[1][i]],
                    color="#CBD5E1", alpha=.65, lw=.8, zorder=2)
        for x, (vals, median, color) in enumerate(zip(values, medians, COLORS)):
            ax.errorbar(x, median, yerr=[[median-min(vals)], [max(vals)-median]],
                        fmt="none", color=color, lw=1.4, capsize=5, zorder=3)
            ax.scatter([x+(i-2)*.036 for i in range(5)], vals,
                       color=color, s=34, alpha=.72, edgecolors="white",
                       linewidths=.6, zorder=4)
            ax.plot([x-.14, x+.14], [median, median], color=color,
                    lw=3.3, solid_capstyle="round", zorder=5)
            ax.text(x, max(vals)+ceiling*.043, f"{median:,.2f}",
                    ha="center", color=color, weight="bold", fontsize=12)
        reduction = 100 * (1 - medians[1]/medians[0])
        ax.text(.98, .95, f"RoCE median ↓ {reduction:.1f}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color="#007C6B", bbox=dict(boxstyle="round,pad=.32",
                facecolor="#EFF8F5", edgecolor="none"))
        ax.text(0, 1.145, f"{letter}   {title}", transform=ax.transAxes,
                fontsize=12, weight="bold")
        ax.text(0, 1.065, subtitle, transform=ax.transAxes,
                fontsize=8.8, color="#64748B")
        ax.set_xticks([0, 1], ["Socket / TCP", "RoCE"])
        ax.set_xlim(-.48, 1.48)
        ax.set_ylim(0, ceiling)
        ax.set_ylabel("Time (ms)")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#E6ECF3", lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=7)
        summary[key] = {t: {"median_ms": statistics.median(v),
                            "min_ms": min(v), "max_ms": max(v)}
                        for t, v in zip(TRANSPORTS, values)}

    handles = [
        Line2D([], [], color="#53667C", lw=3, label="Median (numeric labels)"),
        Line2D([], [], color="#53667C", marker="o", lw=0, markersize=5,
               label="All 5 samples per transport"),
        Line2D([], [], color="#CBD5E1", lw=1, label="Matched input pairs"),
        Line2D([], [], color="#53667C", marker="|", lw=1,
               label="Whiskers: min–max"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .095),
               ncol=4, frameon=False, fontsize=10, handlelength=1.6,
               columnspacing=2)
    fig.text(.045, .058,
             "All axes start at zero. Min–max ranges describe these five samples; they are not confidence intervals.",
             fontsize=9, color="#53667C")
    fig.text(.045, .036,
             "NCCL call sums include waiting; spans include layer gaps. Communication overlaps prefill, so these are not pure wire latencies.",
             fontsize=9, color="#53667C")
    fig.text(.045, .014,
             "24K uses a larger context, token budget, GPU memory fraction and KV buffer than the original six-size run; reported separately.",
             fontsize=9, color="#53667C")
    for extension in ("png", "svg", "pdf"):
        fig.savefig(HERE/f"kv-transport-24k.{extension}", dpi=210)
    plt.close(fig)
    return summary


def main():
    source = HERE/"measurements.json"
    rows = validate(json.loads(source.read_text()))
    summary = render(rows)
    (HERE/"figure-summary.json").write_text(json.dumps({
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "requests": len(rows), "metrics": summary,
    }, indent=2)+"\n")
    print("Validated 10 requests / 5 matched pairs; wrote PNG, SVG and PDF.")


if __name__ == "__main__":
    main()

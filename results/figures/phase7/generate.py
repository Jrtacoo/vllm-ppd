#!/usr/bin/env python3
"""Render three Phase 7 summary figures from evidence or a portable snapshot."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ("1P_1D", "1P_1pD")
COLORS = ("#3264C8", "#008577")
MODE_LABELS = ("PD", "Static PPD")
TRANSPORT_COLORS = ("#708399", "#8854C6")
CELLS = (("short", 1), ("short", 4), ("long", 1), ("long", 4))
METRICS = (("ttft_ms", "TTFT", "ms"),
           ("tpot_ms", "TPOT", "ms / token"),
           ("e2e_ms", "End-to-end latency", "ms"))
SOURCES = ("all-requests.csv", "matched-second-turn-comparison.json",
           "socket-rdma-comparison.json")
HARDWARE = "2 x RTX 4090 | Qwen2.5-7B-Instruct | 100Gbps RoCE, host staging | Dynamic routing disabled"


def read_evidence(path):
    rows = list(csv.DictReader((path / SOURCES[0]).open(newline="")))
    fields = ("routing", "workload", "concurrency", "conversation", "turn", "success",
              "prompt_tokens", "completion_tokens", "ttft_ms", "tpot_ms", "e2e_ms", "error")
    clean = []
    for row in rows:
        r = {k: row[k] for k in fields}
        for k in ("concurrency", "turn", "prompt_tokens", "completion_tokens"):
            r[k] = int(r[k])
        for k, _, _ in METRICS:
            r[k] = float(r[k]) if r[k] else None
        r["success"] = r["success"] == "True"
        clean.append(r)
    return {
        "description": "Plotting data only; original request/SSE/log evidence stays in artifacts/.",
        "source_directory": "artifacts/two-node-phase7-evidence",
        "source_sha256": {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in SOURCES},
        "aggregation": "Comparison JSON values are group medians, as defined by the evidence audit scripts.",
        "requests": clean,
        "matched_second_turn": json.loads((path / SOURCES[1]).read_text()),
        "socket_rdma": json.loads((path / SOURCES[2]).read_text()),
    }


def validate(data):
    rows = data["requests"]
    if len(rows) != 192 or not all(r["success"] and not r["error"] for r in rows):
        raise ValueError("Expected the 192 successful Phase 7 formal requests; inspect failures before plotting.")
    counts = Counter((r["routing"], r["workload"], r["concurrency"], r["turn"]) for r in rows)
    expected = {(m, w, c, t): 12 for m in MODES for w, c in CELLS for t in (1, 2)}
    if dict(counts) != expected:
        raise ValueError("Unexpected request grouping.")
    if not all(math.isfinite(r[k]) and r[k] > 0 for r in rows for k, _, _ in METRICS):
        raise ValueError("Non-finite or non-positive request latency.")
    matched = data["matched_second_turn"]
    if len(matched) != 4 or sum(g["matching_second_turn_pairs"] for g in matched) != 42:
        raise ValueError("Expected 42 matching Turn 2 pairs across four conditions.")
    transport = data["socket_rdma"]
    if (transport["pairs"], transport["same_payload_pairs"], len(transport["groups"])) != (192, 183, 16):
        raise ValueError("Unexpected transport comparison coverage.")
    if sum(g["same_payload_pairs"] for g in transport["groups"]) != 183:
        raise ValueError("Transport group sample counts disagree with total.")
    for g in transport["groups"]:
        if not g["same_payload_pairs"] or any(
            not math.isfinite(g["same_payload"][t][k]) or g["same_payload"][t][k] <= 0
            for t in ("socket", "rdma") for k, _, _ in METRICS
        ):
            raise ValueError("Invalid matched transport data.")


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": "#172D46", "axes.labelcolor": "#41556C",
        "xtick.color": "#41556C", "ytick.color": "#41556C",
        "axes.edgecolor": "#CAD4DF", "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
        "svg.hashsalt": "ppd-phase7", "pdf.fonttype": 42,
    })


def header(fig, kicker, title, subtitle, handles):
    fig.text(0.055, 0.975, kicker, color="#008577", weight="bold", fontsize=10, va="top")
    fig.text(0.055, 0.944, title, weight="bold", fontsize=23, va="top")
    subtitle_y = 0.944 - 0.50 / fig.get_figheight()
    fig.text(0.055, subtitle_y, subtitle, color="#5D6E80", fontsize=10.5, va="top")
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.965, 0.984),
               ncol=2, frameon=False, fontsize=10, handlelength=1.5, columnspacing=1.7)


def footer(fig, lines):
    for i, text in enumerate(reversed(lines)):
        fig.text(0.055, 0.019 + i * 0.019, text, fontsize=8.5, color="#617286")


def save(fig, output, name):
    for extension in ("png", "svg", "pdf"):
        target = output / f"{name}.{extension}"
        fig.savefig(target, dpi=220, metadata={"Creator": "PPD Phase 7 figure generator"})
        print(target)
    plt.close(fig)


def request_figure(data, output):
    rows = data["requests"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.805, bottom=0.15, hspace=0.48, wspace=0.24)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(COLORS, MODE_LABELS)]
    header(fig, "PHASE 7 / 01", "Latency across all formal requests",
           "192 requests | 12 per mode, workload, concurrency and turn | All requests succeeded", handles)
    rng = np.random.default_rng(7)
    for ti, turn in enumerate((1, 2)):
        for j, (field, title, unit) in enumerate(METRICS):
            ax = axes[ti, j]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E8EDF3", linewidth=0.8)
            for mi, (mode, color) in enumerate(zip(MODES, COLORS)):
                values = [[r[field] for r in rows if (r["routing"], r["workload"], r["concurrency"], r["turn"])
                           == (mode, w, c, turn)] for w, c in CELLS]
                positions = np.arange(4) + (-0.19 if mi == 0 else 0.19)
                ax.boxplot(values, positions=positions, widths=0.28, patch_artist=True,
                           showfliers=False, manage_ticks=False,
                           boxprops={"facecolor": color + "26", "edgecolor": color, "linewidth": 1.2},
                           medianprops={"color": color, "linewidth": 2.2},
                           whiskerprops={"color": color, "linewidth": 1},
                           capprops={"color": color, "linewidth": 1})
                for x, v in zip(positions, values):
                    ax.scatter(x + rng.uniform(-0.065, 0.065, len(v)), v,
                               s=17, color=color, alpha=0.63, zorder=3, linewidths=0)
            ax.set_title(f"Turn {turn}  /  {title}", loc="left", pad=13)
            ax.set_ylabel(unit)
            ax.set_xticks(range(4), [f"{w.title()}\nC{c}" for w, c in CELLS])
            ax.set_xlim(-0.6, 3.6)
            ax.set_ylim(0, max(r[field] for r in rows) * 1.13)
            ax.tick_params(axis="both", length=0, pad=7)
    footer(fig, [HARDWARE,
        "Dots: every request. Boxes: median and IQR; whiskers: 1.5 x IQR. All points, including outliers, remain visible. C = concurrent conversations.",
        "Source: all-requests.csv. Warmup/smoke excluded. Turn 2 histories can differ across modes; this figure uses all requests, not a matched subset."])
    save(fig, output, "all-requests-latency")


def matched_figure(data, output):
    by_cell = {(g["workload"], g["concurrency"]): g for g in data["matched_second_turn"]}
    groups = [by_cell[c] for c in CELLS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.78, bottom=0.24, wspace=0.26)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(COLORS, MODE_LABELS)]
    header(fig, "PHASE 7 / 02", "Turn 2 with identical request payloads",
           "PD vs static PPD | 42 matched pairs | Bars show group medians; lower latency is better", handles)
    for ax, (field, title, unit) in zip(axes, METRICS):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E8EDF3", linewidth=0.8)
        values = [[g[mode][field] for g in groups] for mode in MODES]
        highest = max(v for vs in values for v in vs)
        for mi, color in enumerate(COLORS):
            xs = np.arange(4) + (-0.19 if mi == 0 else 0.19)
            ax.bar(xs, values[mi], width=0.32, color=color, zorder=3)
            for x, v in zip(xs, values[mi]):
                label = f"{v:.2f}" if field == "tpot_ms" else f"{v:.1f}" if field == "ttft_ms" else f"{v:.0f}"
                ax.text(x, v + highest * 0.025, label, ha="center", va="bottom", fontsize=8.5, color=color)
        for i, (a, b) in enumerate(zip(*values)):
            delta = (b / a - 1) * 100
            ax.text(i, max(a, b) + highest * 0.15, f"{delta:+.1f}%", ha="center",
                    fontsize=10, weight="bold", color="#008577" if delta < 0 else "#9B5B23",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F2F6F9", "edgecolor": "none"})
        ax.set_title(title, loc="left", pad=14)
        ax.set_ylabel(unit)
        ax.set_ylim(0, highest * 1.39)
        ax.set_xticks(range(4), [f"{w.title()} / C{c}\nn = {g['matching_second_turn_pairs']} pairs"
                                 for (w, c), g in zip(CELLS, groups)])
        ax.tick_params(axis="both", length=0, pad=8)
    footer(fig, [HARDWARE,
        "Badges: relative change in group median = (static PPD / PD - 1) x 100%; these are not medians of per-pair percentage changes.",
        "Source: matched-second-turn-comparison.json. Identical Turn 2 payloads only. Single run; no confidence interval or significance claim."])
    save(fig, output, "matched-turn2-comparison")


def transport_figure(data, output):
    comparison = data["socket_rdma"]
    index = {(g["routing"], g["workload"], g["concurrency"], g["turn"]): g for g in comparison["groups"]}
    groups = [index[(mode, w, c, t)] for mode in MODES for w, c in CELLS for t in (1, 2)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 12))
    fig.subplots_adjust(left=0.205, right=0.97, top=0.815, bottom=0.12, wspace=0.24)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(TRANSPORT_COLORS, ("Socket / Phase 5", "RoCE / Phase 7"))]
    header(fig, "PHASE 7 / 03", "Socket and RoCE: a historical comparison",
           "183 / 192 request pairs have identical payloads | Matched subsets only | Bars show group medians", handles)
    ys = np.arange(len(groups), dtype=float)
    ys[8:] += 0.7
    for j, (ax, (field, title, unit)) in enumerate(zip(axes, METRICS)):
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#E8EDF3", linewidth=0.8)
        high = max(g["same_payload"][t][field] for g in groups for t in ("socket", "rdma"))
        for ti, (transport, color) in enumerate(zip(("socket", "rdma"), TRANSPORT_COLORS)):
            values = [g["same_payload"][transport][field] for g in groups]
            pos = ys + (-0.19 if ti == 0 else 0.19)
            ax.barh(pos, values, height=0.32, color=color, zorder=3)
            for y, v in zip(pos, values):
                label = f"{v:.2f}" if field == "tpot_ms" else f"{v:.1f}" if field == "ttft_ms" else f"{v:.0f}"
                ax.text(v + high * 0.018, y, label, va="center", fontsize=8, color=color)
        ax.axhline(7.85, color="#CBD6E2", linestyle=(0, (3, 3)), linewidth=1)
        ax.set_title(title, loc="left", pad=17)
        ax.set_xlabel(unit, labelpad=10)
        ax.set_xlim(0, high * 1.22)
        ax.set_ylim(ys[-1] + 0.75, -0.75)
        ax.set_yticks(ys)
        if j == 0:
            ax.set_yticklabels([f"{'PD' if g['routing'] == '1P_1D' else 'Static PPD'} | {g['workload'].title()} | C{g['concurrency']} | T{g['turn']}   n={g['same_payload_pairs']}"
                                for g in groups], fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0, pad=8)
    footer(fig, ["2 x RTX 4090 | Qwen2.5-7B-Instruct | Same runtime and workload | RoCE uses host staging; no GPUDirect RDMA",
        "C = concurrent conversations; T = turn; n = identical-payload pairs in that row. Row subsets can differ; unmatched requests stay in the source data.",
        "Source: socket-rdma-comparison.json, same_payload groups. Separate single runs; no causal protocol-speedup or statistical-significance claim."])
    save(fig, output, "socket-roce-comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "artifacts/two-node-phase7-evidence")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--from-snapshot", action="store_true", help="Use plot-data.json beside this script; no artifacts needed.")
    args = parser.parse_args()
    data = json.loads((HERE / "plot-data.json").read_text()) if args.from_snapshot else read_evidence(args.evidence_dir)
    validate(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.from_snapshot:
        (args.output_dir / "plot-data.json").write_text(json.dumps(data, indent=2) + "\n")
    style()
    request_figure(data, args.output_dir)
    matched_figure(data, args.output_dir)
    transport_figure(data, args.output_dir)


if __name__ == "__main__":
    main()

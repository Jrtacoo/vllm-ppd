#!/usr/bin/env python3
"""Amortize observed generation-span excess over all 128 output tokens."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st

from generate import COLORS, LABELS, GRAY, setup, layout, finish

HERE = Path(__file__).resolve().parent


def derive(data):
    assert data['validation']['passed']
    tokens = data['run_metadata']['formal']['decode_output_tokens']
    assert tokens == 128
    rows, groups = [], []
    for g in data['groups']:
        if g['run'] != 'formal':
            continue
        trials = [w for w in g['window_trials']
                  if w['metric'] == 'client_ms' and w['window'] == 'whole_request']
        assert len(trials) == 5 and {w['repeat'] for w in trials} == set(range(5))
        values = []
        for w in trials:
            assert w['steps'] == tokens - 1
            excess = (tokens - 1) * (w['mean_ms'] - w['baseline_mean_ms'])
            assert math.isclose(excess, w['cumulative_delta_ms'], abs_tol=1e-8)
            value = excess / tokens
            rows.append(dict(batch_size=g['batch_size'], scenario=g['scenario'],
                             repeat=w['repeat'], trial=w['trial'], output_tokens=tokens,
                             generation_span_excess_ms=excess,
                             amortized_excess_ms_per_output_token=value))
            values.append(value)
        groups.append(dict(batch_size=g['batch_size'], scenario=g['scenario'],
                           mean=st.mean(values), std=st.stdev(values), n=len(values)))
    assert len(groups) == 40 and len(rows) == 200
    return rows, groups


def draw(groups, out):
    setup()
    fig, axes = layout(1, 1, 'Extra delay spread across 128 output tokens',
        'Single RTX 4090 | Qwen2.5-7B | 16K history + 1K new input | Independent background KV | 5 repeats')
    ax = axes[0, 0]
    for scenario, color in COLORS.items():
        points = sorted((g for g in groups if g['scenario'] == scenario), key=lambda g:g['batch_size'])
        ax.errorbar([g['batch_size'] for g in points], [g['mean'] for g in points],
                    yerr=[g['std'] for g in points], color=color, lw=2.2,
                    marker='o' if scenario == 'full_prefill' else 's', ms=5, capsize=3)
        last = points[-1]
        ax.annotate(f"{LABELS[scenario]}: {last['mean']:.2f} ms/token",
                    xy=(last['batch_size'], last['mean']), xytext=(-18, 42),
                    textcoords='offset points', ha='right', color=color, weight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=3),
                    arrowprops=dict(arrowstyle='-', color=color, lw=1))
    ax.axhline(0, color=GRAY, ls='--', lw=1)
    ax.set_xlim(0, 205)
    ax.set_xticks(list(range(0, 201, 20)))
    ax.set_ylim(-1.2, 19)
    ax.set_xlabel('Observed decode batch size')
    ax.set_ylabel('Extra delay per output token (ms/token)')
    ax.set_title('Per-request generation-span excess / 128', loc='left', weight='bold', pad=12)
    finish(fig, out, 'amortized-128-token-delay', [
        'Per trial: [(last - first token arrival) with prefill - matched decode-only span] / 128 = 127/128 x (TPOT - baseline TPOT).',
        'Average background requests first, then five paired repeats; error bars = sample SD. Includes all 127 gaps and signed fluctuations; excludes TTFT.',
        'This is an amortized accounting metric, not a uniform delay at every token. No division by batch size; batch 0 has no background output.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, default=HERE/'plot-data.json')
    p.add_argument('--output-dir', type=Path, default=HERE)
    args = p.parse_args()
    rows, groups = derive(json.loads(args.snapshot.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/'amortized-128-token-trials.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    draw(groups, args.output_dir)
    print(json.dumps([g for g in groups if g['batch_size'] == 200], indent=2))


if __name__ == '__main__':
    main()

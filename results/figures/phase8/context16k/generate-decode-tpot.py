#!/usr/bin/env python3
"""Plot absolute background decode TPOT for all three experimental conditions."""
import argparse
import json
from pathlib import Path
import statistics as st

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from generate import setup, layout, finish

HERE = Path(__file__).resolve().parent
STYLES = {
    'decode_only': ('#347BB5', 'o', 'Decode only'),
    'append_prefill': ('#008577', 's', 'Decode + 1 append prefill'),
    'full_prefill': ('#D46A37', '^', 'Decode + 1 full prefill'),
}


def derive(data):
    assert data['validation']['passed']
    assert data['run_metadata']['formal']['decode_output_tokens'] == 128
    values = {}
    for group in data['groups']:
        if group['run'] != 'formal':
            continue
        batch, scenario = group['batch_size'], group['scenario']
        for w in group['window_trials']:
            if w['window'] != 'whole_request' or w['metric'] != 'client_ms':
                continue
            assert w['steps'] == 127
            values[batch, scenario, w['repeat']] = w['mean_ms']
            key = batch, 'decode_only', w['repeat']
            if key in values:
                assert values[key] == w['baseline_mean_ms']
            values[key] = w['baseline_mean_ms']
    assert len(values) == 300
    result = []
    for batch in sorted({k[0] for k in values}):
        for scenario in STYLES:
            xs = [values[batch, scenario, r] for r in range(5)]
            result.append(dict(batch_size=batch, scenario=scenario, mean_ms=st.mean(xs),
                               std_ms=st.stdev(xs), trial_tpot_ms=xs, repeats=5))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=HERE/'plot-data.json')
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    groups = derive(json.loads(args.snapshot.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup()
    fig, axes = layout(1, 1, 'Background decode TPOT under prefill interference',
        'Single RTX 4090 | Qwen2.5-7B | 128 output tokens per background request | Mean and sample SD, 5 repeats')
    for legend in list(fig.legends):
        legend.remove()
    fig.legend(handles=[Line2D([0], [0], color=c, marker=m, lw=2, label=label)
                        for c, m, label in STYLES.values()],
               loc='upper right', bbox_to_anchor=(.975, 1-.17/fig.get_figheight()),
               ncol=3, frameon=False, fontsize=9)
    ax = axes[0, 0]
    for scenario, (color, marker, label) in STYLES.items():
        points = [g for g in groups if g['scenario'] == scenario]
        ax.errorbar([g['batch_size'] for g in points], [g['mean_ms'] for g in points],
                    yerr=[g['std_ms'] for g in points], color=color, marker=marker,
                    lw=2.2, ms=5, capsize=3)
    ax.set_xlabel('Observed decode batch size')
    ax.set_ylabel('Average background decode TPOT (ms)')
    ax.set_xticks(list(range(0, 201, 20)))
    ax.set_xlim(0, 205)
    ax.set_ylim(bottom=0)
    finish(fig, args.output_dir, 'background-decode-tpot', [
        'For each background request: TPOT = (last token arrival - first token arrival) / 127. Average requests within each trial, then five repeats.',
        'Only background decode requests contribute to these curves. Interferer TTFT and prefill model wall time are separate measurements.',
        'Adapted workload: independent 128-token histories; full recomputes 17,408 tokens, append computes 1,024 with 16,384 cached. Batch 0 has no TPOT.'])
    (args.output_dir/'background-decode-tpot-data.json').write_text(json.dumps({
        'metric': 'Absolute background decode TPOT; 127 inter-token arrival gaps, excludes TTFT',
        'groups': groups}, indent=2)+'\n')
    print(json.dumps([g for g in groups if g['batch_size'] == 200], indent=2))


if __name__ == '__main__':
    main()

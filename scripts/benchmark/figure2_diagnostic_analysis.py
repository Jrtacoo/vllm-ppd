#!/usr/bin/env python3
"""Analyze the bounded history/budget diagnostic, including multi-chunk prefills."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics as st


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def describe(xs):
    return dict(mean=st.mean(xs), std=st.stdev(xs) if len(xs) > 1 else 0, n=len(xs), min=min(xs), max=max(xs))


def digest(data):
    return hashlib.sha256(json.dumps(data, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', help='Analyze a completed subset for diagnostic validation')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [('budget2048', 'h16k-b2048'), ('budget2048', 'h2k-b2048'),
             ('budget2048', 'literal-h2k-b2048'), ('budget17664', 'h16k-b17664'), ('budget17664', 'h2k-b17664')]
    if args.cases:
        assert set(args.cases) <= {case for _,case in cases}
        cases = [(engine,case) for engine,case in cases if case in args.cases]
    decoded_hashes, suffix_hashes, history_hashes = defaultdict(set), defaultdict(set), defaultdict(set)
    trial_rows, profiles, exclusions, source_hashes, metadata, bundled_events = [], [], [], {}, {}, []
    trace_index, gpu_samples = {}, {}
    for engine in {x[0] for x in cases}:
        path = args.evidence / engine / 'scheduler-trace.jsonl'
        index = defaultdict(list)
        trace_hash = hashlib.sha256()
        with path.open('rb') as f:
            for line in f:
                trace_hash.update(line)
                record = json.loads(line)
                relevant = defaultdict(list)
                for r in record['requests']:
                    for role in ('decode', 'interferer'):
                        if '_' + role + '_' in r['id']:
                            trial = r['id'].removeprefix('cmpl-').rsplit('_' + role + '_', 1)[0]
                            relevant[trial].append((role, r))
                for trial, rr in relevant.items():
                    decodes = [r for role, r in rr if role == 'decode' and r['computed_before'] >= r['prompt_tokens'] and r['scheduled_tokens'] == 1]
                    prefills = [r for role, r in rr if role == 'interferer' and r['computed_before'] < r['prompt_tokens']]
                    if decodes:
                        index[trial].append(dict(step=record['step'], model_ms=record['model_wall_ms'],
                            prefill_tokens=sum(r['scheduled_tokens'] for r in prefills),
                            positions={int(r['id'].rsplit('_decode_', 1)[1].split('-')[0]): r['output_tokens_before'] for r in decodes}))
                assert sum(r['scheduled_tokens'] for r in record['requests']) <= int(engine.removeprefix('budget'))
        trace_index[engine] = index
        source_hashes[str(path.relative_to(args.evidence))] = trace_hash.hexdigest()
        gpu_samples[engine] = rows(args.evidence / engine / 'gpu-samples.jsonl')
    for engine, case in cases:
        folder = args.evidence / engine / case
        report = json.loads((folder / 'verification.json').read_text())
        assert report['passed'], case
        meta = json.loads((folder / 'run.json').read_text())
        metadata[case] = meta
        trials = rows(folder / 'trials.jsonl')
        requests = rows(folder / 'requests.jsonl')
        audit = {r['trial']: r for r in report['trial_audits']}
        lookup = {(t['batch_size'], t['repeat'], t['scenario']): t for t in trials}
        times, singleton, interferers, active_spans = {}, {}, {}, {}
        for filename in ('verification.json', 'run.json', 'requests.jsonl', 'trials.jsonl'):
            source_hashes[f'{engine}/{case}/{filename}'] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        for r in requests:
            if r['role'] == 'warmup':
                continue
            span = active_spans.setdefault(r['trial'], [r['start_ns'], r['end_ns']])
            span[0] = min(span[0], r['start_ns'])
            span[1] = max(span[1], r['end_ns'])
            stem = folder / r['raw_stem']
            payload = json.loads(Path(str(stem) + '.request.json').read_text())
            key = (r['batch_size'], r['repeat'])
            if r['role'] == 'interferer':
                interferers[r['trial']] = r
                suffix_hashes[key].add(digest(payload['prompt'][-1024:]))
                if r['scenario'] == 'append_prefill':
                    history_hashes[key].add(digest(payload['prompt'][:2048]))
                continue
            seq = int(r['request_id'].rsplit('_', 1)[1])
            decoded_hashes[key + (seq,)].add(digest(payload['prompt']))
            ts, singles = [], []
            for event in json.loads(Path(str(stem) + '.events.json').read_text()):
                if not isinstance(event['data'], dict):
                    continue
                for choice in event['data'].get('choices', []):
                    ids = choice.get('token_ids') or []
                    if ids:
                        if len(ids) > 1:
                            bundled_events.append(dict(case=case,trial=r['trial'],request_id=r['request_id'],
                                first_token_ordinal=len(ts)+1,token_count=len(ids),perf_ns=event['perf_ns']))
                        # These tokens arrive in the same SSE event. Their individual
                        # model-step arrival gaps are not resolved; exclude affected
                        # local intervals, but retain whole-request client TPOT.
                        ts.extend([event['perf_ns']] * len(ids))
                        singles.extend([len(ids) == 1] * len(ids))
            assert len(ts) == 128
            assert math.isclose((ts[-1] - ts[0]) / 1e6 / 127, r['token_tpot_ms'], abs_tol=1e-8)
            times[r['trial'], seq] = ts
            singleton[r['trial'], seq] = singles
        for t in trials:
            tid, batch = t['trial'], t['batch_size']
            base = lookup[batch, t['repeat'], 'decode_only']
            base_id = base['trial']
            mixed = audit[tid]['mixed_steps']
            row = dict(case=case, budget=int(engine.removeprefix('budget')), history=meta['append_prefill_cached_tokens'],
                mode=meta['full_prefill_mode'], batch=batch, repeat=t['repeat'], scenario=t['scenario'], trial=tid,
                tpot_ms=t['decode_avg_tpot_ms'], baseline_tpot_ms=base['decode_avg_tpot_ms'],
                excess_tpot_ms=t['decode_avg_tpot_ms'] - base['decode_avg_tpot_ms'],
                slowdown_pct=100 * (t['decode_avg_tpot_ms'] / base['decode_avg_tpot_ms'] - 1),
                total_span_excess_ms=127 * (t['decode_avg_tpot_ms'] - base['decode_avg_tpot_ms']),
                prefill_chunks=len(mixed), mixed_model_sum_ms=sum(s['model_wall_ms'] for s in mixed),
                mixed_model_max_ms=max((s['model_wall_ms'] for s in mixed), default=0),
                prefill_tokens_by_chunk=[sum(r['scheduled_tokens'] for r in s['interferers']) for s in mixed])
            all_steps = trace_index[engine][tid]
            baseline_index = {(seq,k):s for s in trace_index[engine][base_id] for seq,k in s['positions'].items()}
            mixed_excess = []
            for s in all_steps:
                if s['prefill_tokens']:
                    refs = [baseline_index[seq,k] for seq,k in s['positions'].items()]
                    assert len(s['positions']) == batch and all(len(b['positions']) == batch for b in refs)
                    mixed_excess.append(s['model_ms'] - st.mean(b['model_ms'] for b in refs))
            row['mixed_model_excess_ms'] = sum(mixed_excess)
            row['max_client_gap_mean_ms'] = st.mean(max((b-a)/1e6 for a,b in zip(times[tid,i], times[tid,i][1:])) for i in range(batch))
            samples = []
            for sample in gpu_samples[engine]:
                if active_spans[tid][0] <= sample['perf_ns'] <= active_spans[tid][1] and sample['rc'] == 0:
                    v = [x.strip() for x in sample['values'].split(',')]
                    samples.append(dict(gpu_util_pct=float(v[1]), memory_util_pct=float(v[2]),
                        memory_used_mib=float(v[3]), power_w=float(v[5]), temperature_c=float(v[6])))
            row['gpu_sampling'] = {k:describe([s[k] for s in samples]) for k in samples[0]} if samples else {}
            if mixed:
                intf = interferers[tid]
                row['interferer_ttft_ms'] = intf['ttft_ms']
                row['interferer_e2e_ms'] = intf['e2e_ms']
                first = next(i for i,s in enumerate(all_steps) if s['prefill_tokens'])
                expected = meta['full_prefill_computed_tokens'] if t['scenario'] == 'full_prefill' else 1024
                assert sum(s['prefill_tokens'] for s in all_steps) == expected
                for j, s in enumerate(all_steps):
                    offset = j - first
                    if len(s['positions']) != batch:
                        exclusions.append(dict(case=case, trial=tid, offset=offset, reason='partial treatment batch'))
                        continue
                    if not all(singleton[trial_key,seq][k] and singleton[trial_key,seq][k-1]
                               for seq,k in s['positions'].items() for trial_key in (tid,base_id)):
                        exclusions.append(dict(case=case, trial=tid, offset=offset, reason='bundled SSE obscures individual token interval in treatment or matched baseline'))
                        continue
                    gaps, base_gaps = [], []
                    for seq, k in s['positions'].items():
                        assert 1 <= k < 128
                        a, b = times[tid, seq], times[base_id, seq]
                        gaps.append((a[k]-a[k-1])/1e6)
                        base_gaps.append((b[k]-b[k-1])/1e6)
                    profiles.append(dict(case=case, trial=tid, batch=batch, repeat=t['repeat'], scenario=t['scenario'],
                        offset=offset, prefill_tokens=s['prefill_tokens'], model_ms=s['model_ms'],
                        client_ms=st.mean(gaps), baseline_client_ms=st.mean(base_gaps),
                        client_delta_ms=st.mean(gaps)-st.mean(base_gaps)))
            trial_rows.append(row)
        print('Analyzed', case, flush=True)
    assert all(len(v) == 1 for v in decoded_hashes.values()), 'Cross-case background input changed'
    assert all(len(v) == 1 for v in suffix_hashes.values()), 'Cross-case new input changed'
    assert all(len(v) == 1 for v in history_hashes.values()), '2K history not prefix of 16K'
    metrics = ('tpot_ms', 'excess_tpot_ms', 'slowdown_pct', 'total_span_excess_ms', 'prefill_chunks',
               'mixed_model_sum_ms', 'mixed_model_max_ms', 'mixed_model_excess_ms', 'max_client_gap_mean_ms')
    groups = []
    for case, batch, scenario in sorted({(r['case'],r['batch'],r['scenario']) for r in trial_rows}):
        rr = [r for r in trial_rows if (r['case'],r['batch'],r['scenario']) == (case,batch,scenario)]
        assert len(rr) == 3
        groups.append(dict(case=case, batch=batch, scenario=scenario, **{m:describe([r[m] for r in rr]) for m in metrics}))
    # Report client gaps on exact matched output ordinals; never discard negative excess.
    profile_groups = []
    grouped = defaultdict(list)
    for r in profiles:
        grouped[r['case'], r['batch'], r['scenario'], r['offset']].append(r)
    for (case,batch,scenario,offset), rr in sorted(grouped.items()):
        profile_groups.append(dict(case=case,batch=batch,scenario=scenario,offset=offset,
            **{m:describe([r[m] for r in rr]) for m in ('client_ms','baseline_client_ms','client_delta_ms','model_ms','prefill_tokens')}))
    comparisons = []
    row_index = {(r['case'],r['batch'],r['scenario'],r['repeat']): r for r in trial_rows}
    for old, new, label in [('h16k-b17664','h16k-b2048','reduce_budget_at_16k'),
                            ('h2k-b17664','h2k-b2048','reduce_budget_at_2k'),
                            ('h16k-b17664','h2k-b17664','reduce_history_at_large_budget'),
                            ('h16k-b2048','h2k-b2048','reduce_history_at_small_budget')]:
        if old not in metadata or new not in metadata:
            continue
        for batch in (10,100,200):
            for scenario in ('full_prefill','append_prefill'):
                pairs = [(row_index[old,batch,scenario,r],row_index[new,batch,scenario,r]) for r in range(3)]
                comparisons.append(dict(change=label,batch=batch,scenario=scenario,
                    excess_tpot_change_ms=describe([b['excess_tpot_ms']-a['excess_tpot_ms'] for a,b in pairs]),
                    peak_model_change_ms=describe([b['mixed_model_max_ms']-a['mixed_model_max_ms'] for a,b in pairs])))
    out = dict(validation={'passed': True, 'trials': len(trial_rows), 'cross_case_background_and_suffix_identical': True,
               'history_prefix_identical': True, 'raw_sse_tpot_recomputed': True, 'scheduler_token_budgets_checked': True,
               'excluded_local_steps': len(exclusions),
               'excluded_partial_steps': sum(x['reason']=='partial treatment batch' for x in exclusions),
               'excluded_bundled_steps': sum(x['reason'].startswith('bundled SSE') for x in exclusions),
               'bundled_sse_events': len(bundled_events)},
        method={'primary': 'All 127 client token gaps per background request, then requests, then 3 repeats; SD across repeats.',
                'event': 'Offset zero is first prefill chunk. Exact background output-token ordinal matched to same-repeat decode-only. Exclude local client intervals touching bundled SSE; retain all chunks in model metrics and all requests in whole-request TPOT.',
                'max_client_gap': 'Mean across requests of the largest observed SSE token-event arrival interval; bundled events do not resolve individual model-step gaps.',
                'limitations': 'Two sequential budget blocks; 3 repeats; single 4090/Qwen; caption control is a literal interpretation, not confirmed paper workload.'},
        metadata=metadata, groups=groups, trial_rows=trial_rows, profile_groups=profile_groups,
        paired_configuration_changes=comparisons, source_hashes=source_hashes)
    for name, data in [('plot-data.json',out), ('event-trials.json',profiles), ('excluded-local-steps.json',exclusions), ('bundled-sse-events.json',bundled_events)]:
        (args.output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(out['validation'],indent=2))


if __name__ == '__main__':
    main()

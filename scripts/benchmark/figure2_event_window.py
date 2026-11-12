#!/usr/bin/env python3
"""Read-only event-aligned analysis of audited Figure 2 traces and token SSE.

Writes a NEW derived-artifact directory; never modifies source evidence or runs inference.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st

SCENARIOS = ('full_prefill', 'append_prefill')
METRICS = ('model_ms', 'cycle_ms', 'client_ms')


def summary(xs):
    return {'mean': st.mean(xs), 'std': st.stdev(xs) if len(xs) > 1 else 0.0,
            'n': len(xs), 'min': min(xs), 'max': max(xs)} if xs else None


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def first_quiet_window(rows, threshold, metric, consecutive=5):
    """A descriptive local return, not a causal/recovery significance test."""
    delta = metric + '_delta'
    for i in range(1, len(rows) - consecutive + 1):
        if all(abs(r[delta]) <= threshold for r in rows[i:i+consecutive]):
            later = rows[i+consecutive:]
            return {'quiet_start': rows[i]['offset'], 'last_initial_step': rows[i-1]['offset'],
                    'censored': False, 'later_exceedances': sum(abs(r[delta]) > threshold for r in later),
                    'later_steps': len(later),
                    'later_max_abs_delta_ms': max((abs(r[delta]) for r in later), default=0.0)}
    return {'quiet_start': None, 'last_initial_step': rows[-1]['offset'], 'censored': True,
            'later_exceedances': None, 'later_steps': 0, 'later_max_abs_delta_ms': None}


def load_inputs(evidence, runs):
    trials, requests, reports, sources = {}, {}, {}, {}
    for run in runs:
        folder = evidence / run
        report = json.loads((folder / 'verification.json').read_text())
        assert report['passed'], run
        reports[run] = report
        for name in ('verification.json', 'run.json', 'trials.jsonl', 'requests.jsonl'):
            sources[run + '/' + name] = sha(folder / name)
        runmeta = json.loads((folder / 'run.json').read_text())
        for t in jsonl(folder / 'trials.jsonl'):
            if t['batch_size'] == 0: continue
            t['expected_prefill_tokens'] = (runmeta.get('full_prefill_computed_tokens', runmeta['full_prefill_new_tokens'])
                if t['scenario'] == 'full_prefill' else runmeta['append_prefill_new_tokens'])
            t['run'] = run; t['prefix_mode'] = report['prefix_mode']
            trials[t['trial']] = t
        for r in jsonl(folder / 'requests.jsonl'):
            if r['role'] == 'decode': requests[r['request_id']] = (run, r)
    return trials, requests, reports, sources


def extract_trace(evidence, trials, requests):
    steps = defaultdict(list); trace_hash = hashlib.sha256(); previous = None
    with (evidence / 'scheduler-trace.jsonl').open('rb') as f:
        for line in f:
            trace_hash.update(line); record = json.loads(line)
            if 'requests' not in record: continue
            relevant = defaultdict(list); mixed = set()
            for r in record['requests']:
                rid = r['id'].removeprefix('cmpl-')
                if rid not in requests: rid = rid.rsplit('-', 1)[0]
                if rid in requests:
                    if r['computed_before'] >= r['prompt_tokens'] and r['scheduled_tokens'] == 1:
                        trial = requests[rid][1]['trial']
                        relevant[trial].append((int(rid.rsplit('_decode_', 1)[1]), r['output_tokens_before']))
                elif '_interferer_' in rid and r['computed_before'] < r['prompt_tokens']:
                    trial = rid.rsplit('_interferer_', 1)[0]
                    if trial in trials:
                        assert r['scheduled_tokens'] == trials[trial]['expected_prefill_tokens']
                        mixed.add(trial)
            for trial, pairs in relevant.items():
                t = trials[trial]
                steps[trial].append({'step': record['step'], 'positions': dict(pairs),
                    'model_ms': record['model_wall_ms'],
                    'cycle_ms': (record['update_return_ns'] - previous['update_return_ns']) / 1e6 if previous else None,
                    'update_ns': record['update_return_ns'], 'observer_us': record['observer_us'],
                    'full_batch': len(pairs) == t['batch_size'], 'mixed': trial in mixed})
            previous = record
    return steps, trace_hash.hexdigest()


def extract_sse(evidence, requests):
    times, hashes, stats = {}, {}, {'requests': 0, 'tokens': 0, 'multi_token_events': 0}
    for i, (rid, (run, r)) in enumerate(requests.items(), 1):
        path = Path(str(evidence / run / r['raw_stem']) + '.events.json')
        raw = path.read_bytes(); hashes[str(path.relative_to(evidence))] = hashlib.sha256(raw).hexdigest()
        token_times = []
        for e in json.loads(raw):
            if not isinstance(e['data'], dict): continue
            for c in e['data'].get('choices', []):
                ids = c.get('token_ids') or []
                if len(ids) > 1: stats['multi_token_events'] += 1
                # Never invent per-token times within a bundled SSE event.
                if ids:
                    assert len(ids) == 1, 'Bundled token event needs interval-censored analysis: ' + rid
                    token_times.append(e['perf_ns'])
        assert len(token_times) == r['completion_tokens'] == 128
        assert all(b >= a for a, b in zip(token_times, token_times[1:]))
        gaps = [(b-a)/1e6 for a, b in zip(token_times, token_times[1:])]
        assert math.isclose(st.mean(gaps), r['token_tpot_ms'], abs_tol=1e-9), rid
        times[rid] = token_times; stats['requests'] += 1; stats['tokens'] += len(token_times)
        if i % 2500 == 0: print('Read SSE', i, '/', len(requests), flush=True)
    return times, hashes, stats


def compare(trials, steps, times):
    lookup = {(t['run'], t['batch_size'], t['repeat'], t['scenario']): t for t in trials.values()}
    aligned, windows, recovery, exclusions = [], [], [], []
    for trial, t in trials.items():
        if t['scenario'] == 'decode_only': continue
        baseline = lookup[t['run'], t['batch_size'], t['repeat'], 'decode_only']
        base_id = baseline['trial']; base_steps = steps[base_id]
        index = {(seq, k): s for s in base_steps for seq, k in s['positions'].items()}
        mixed = [i for i, s in enumerate(steps[trial]) if s['mixed']]
        assert len(mixed) == 1 and steps[trial][mixed[0]]['full_batch'], trial
        audit = []; mixed_i = mixed[0]
        for i, s in enumerate(steps[trial]):
            offset = i - mixed_i
            refs = [index.get((seq, k)) for seq, k in s['positions'].items()]
            valid = s['full_batch'] and all(r and r['full_batch'] for r in refs)
            if not valid:
                exclusions.append({'trial': trial, 'offset': offset, 'step': s['step'],
                    'reason': 'treatment or token-position-matched baseline has partial batch'})
                continue
            assert s['cycle_ms'] is not None
            row = {k: t[k] for k in ('run', 'prefix_mode', 'batch_size', 'repeat', 'scenario')}
            row.update(trial=trial, baseline_trial=base_id, offset=offset, step=s['step'],
                       output_tokens_before_min=min(s['positions'].values()),
                       output_tokens_before_max=max(s['positions'].values()),
                       observer_us=s['observer_us'], baseline_step_count=len({r['step'] for r in refs}))
            client, base_client = [], []
            for seq, k in s['positions'].items():
                a = times[trial + '_decode_' + str(seq)]; b = times[base_id + '_decode_' + str(seq)]
                assert 1 <= k < len(a)
                client.append((a[k] - a[k-1])/1e6); base_client.append((b[k] - b[k-1])/1e6)
            row['client_ms'] = st.mean(client); row['client_ms_baseline'] = st.mean(base_client)
            for metric in ('model_ms', 'cycle_ms'):
                row[metric] = s[metric]; row[metric + '_baseline'] = st.mean(r[metric] for r in refs)
            for metric in METRICS:
                row[metric + '_delta'] = row[metric] - row[metric + '_baseline']
                row[metric + '_pct'] = 100 * row[metric + '_delta'] / row[metric + '_baseline']
            audit.append(row)
        post = [r for r in audit if r['offset'] >= 0]
        assert post and post[0]['offset'] == 0
        assert [r['offset'] for r in post] == list(range(post[-1]['offset'] + 1)), trial
        assert [r['offset'] for r in audit if -5 <= r['offset'] <= 0] == list(range(-5, 1)), trial
        # Full consecutive steps are needed for sums of request token intervals.
        subset = steps[trial][mixed_i:mixed_i+len(post)]
        for a, b in zip(subset, subset[1:]):
            assert b['step'] == a['step'] + 1
            assert all(b['positions'][seq] == k+1 for seq, k in a['positions'].items())
        for metric in METRICS:
            cumulative = 0.0
            for r in audit:
                if r['offset'] >= 0: cumulative += r[metric + '_delta']
                r[metric + '_cumulative_delta'] = cumulative if r['offset'] >= 0 else None
        # Check that per-step client sums equal elapsed endpoint differences.
        for scenario_id, field in ((trial, 'client_ms'), (base_id, 'client_ms_baseline')):
            elapsed = []
            for seq, first_k in subset[0]['positions'].items():
                last_k = subset[-1]['positions'][seq]; a = times[scenario_id + '_decode_' + str(seq)]
                elapsed.append((a[last_k] - a[first_k-1])/1e6)
            assert math.isclose(sum(r[field] for r in post), st.mean(elapsed), abs_tol=1e-8)
        base_full = [s for s in base_steps if s['full_batch']]
        for metric in METRICS:
            noise = ([s[metric] for s in base_full] if metric != 'client_ms'
                     else [r['client_ms_baseline'] for r in audit])
            median = st.median(noise); mad = st.median(abs(x-median) for x in noise)
            # Save sensitivity to the relative floor; this is descriptive only.
            for relative in (0.03, 0.05, 0.10):
                threshold = max(1.0, relative * median, 3 * 1.4826 * mad)
                quiet = first_quiet_window(post, threshold, metric)
                end = quiet['last_initial_step']; initial = post[:end+1]
                recovery.append(dict(trial=trial, run=t['run'], prefix_mode=t['prefix_mode'],
                    batch_size=t['batch_size'], repeat=t['repeat'], scenario=t['scenario'], metric=metric,
                    relative_floor=relative, threshold_ms=threshold, baseline_median_ms=median,
                    baseline_mad_ms=mad, initial_cumulative_delta_ms=sum(r[metric+'_delta'] for r in initial),
                    available_post_steps=len(post), **quiet))
        for length in (1, 5, 16, 32, 64, len(post)):
            assert len(post) >= length
            rs = post[:length]
            label = 'available' if length == len(post) else str(length)
            for metric in METRICS:
                value = st.mean(r[metric] for r in rs); base = st.mean(r[metric+'_baseline'] for r in rs)
                windows.append(dict(trial=trial, run=t['run'], prefix_mode=t['prefix_mode'], batch_size=t['batch_size'],
                    repeat=t['repeat'], scenario=t['scenario'], window=label, steps=length, metric=metric,
                    mean_ms=value, baseline_mean_ms=base, change_percent=100*(value/base-1),
                    cumulative_delta_ms=sum(r[metric+'_delta'] for r in rs)))
        windows.append(dict(trial=trial, run=t['run'], prefix_mode=t['prefix_mode'], batch_size=t['batch_size'],
                    repeat=t['repeat'], scenario=t['scenario'], window='whole_request', steps=127, metric='client_ms',
                    mean_ms=t['decode_avg_tpot_ms'], baseline_mean_ms=baseline['decode_avg_tpot_ms'],
                    change_percent=100*(t['decode_avg_tpot_ms']/baseline['decode_avg_tpot_ms']-1),
                    cumulative_delta_ms=127*(t['decode_avg_tpot_ms']-baseline['decode_avg_tpot_ms'])))
        aligned.extend(audit)
    return aligned, windows, recovery, exclusions


def aggregate(aligned, windows, recovery):
    output = []
    for run, batch, scenario in sorted({(r['run'], r['batch_size'], r['scenario']) for r in aligned}):
        rs = [r for r in aligned if (r['run'], r['batch_size'], r['scenario']) == (run, batch, scenario)]
        profile = []
        for offset in sorted({r['offset'] for r in rs}):
            point = [r for r in rs if r['offset'] == offset]
            fields = [m+s for m in METRICS for s in ('', '_baseline', '_delta', '_pct', '_cumulative_delta')]
            profile.append(dict(offset=offset, n=len(point), **{
                field: summary([r[field] for r in point if r[field] is not None]) for field in fields}))
        ws = [r for r in windows if (r['run'], r['batch_size'], r['scenario']) == (run,batch,scenario)]
        window_summary = []
        for metric, window in sorted({(r['metric'], r['window']) for r in ws}):
            v = [r for r in ws if (r['metric'],r['window']) == (metric,window)]
            assert len(v) == 5
            window_summary.append(dict(metric=metric, window=window,
                **{field:summary([r[field] for r in v]) for field in ('mean_ms','baseline_mean_ms','change_percent','cumulative_delta_ms','steps')}))
        output.append(dict(run=run, prefix_mode=rs[0]['prefix_mode'], batch_size=batch, scenario=scenario,
            profile=profile, windows=window_summary,
            recovery_trials=[r for r in recovery if (r['run'],r['batch_size'],r['scenario']) == (run,batch,scenario)],
            window_trials=ws))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', nargs='+', default=['formal','formal-independent'])
    args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    trials, requests, reports, sources = load_inputs(args.evidence,args.runs)
    steps, trace_hash = extract_trace(args.evidence,trials,requests)
    assert all(trace_hash == r['observer']['trace_sha256'] for r in reports.values())
    sources['scheduler-trace.jsonl'] = trace_hash
    print('Trace matched original audits;',len(trials),'trials',flush=True)
    times, event_hashes, event_stats = extract_sse(args.evidence,requests)
    aligned, windows, recovery, exclusions = compare(trials,steps,times)
    groups = aggregate(aligned,windows,recovery)
    original_checks = dict(line.split('  ', 1)[::-1] for line in (args.evidence/'evidence.sha256').read_text().splitlines())
    for name, digest in {**sources, **event_hashes}.items():
        assert original_checks[name] == digest, 'Source changed since experiment checksum: ' + name
    manifest={'source_evidence':str(args.evidence), 'files':sources, 'token_event_files':event_hashes,
              'analysis_script_sha256':sha(Path(__file__))}
    write_json(args.output/'source-manifest.json',manifest)
    write_csv(args.output/'aligned-iterations.csv',aligned)
    write_csv(args.output/'window-trials.csv',windows)
    write_csv(args.output/'recovery-trials.csv',recovery)
    write_csv(args.output/'excluded-partial-steps.csv',exclusions)
    method={
        'alignment':'Offset 0 is the scheduler step executing the injected prefill with the full decode batch. Scheduled token count must match run metadata. Negative offsets precede it.',
        'pairing':'Same run, batch, repeat and decode request suffix; match the exact output_tokens_before ordinal per request. Average matched baseline step times over requests.',
        'model_ms':'schedule_return_ns to output_received_ns; host/engine wall time, not CUDA kernel time.',
        'cycle_ms':'Consecutive update_return_ns difference; includes model work, intervening scheduling and host/observer overhead.',
        'client_ms':'Mean across background requests of the singleton token-ID SSE gap for the output produced by this step. It is client arrival time, not pure GPU time.',
        'aggregation':'Within each trial average requests; across five repeats mean and sample SD. Steps and requests are not independent replicates.',
        'windows':'Fixed post-injection windows of 1, 5, 16, 32 and 64 steps, plus available full-batch suffix; whole_request is original 127-gap client TPOT. No trimming of outliers.',
        'recovery':'First run of five consecutive post-injection steps whose absolute paired excess is <= max(1 ms, 5% baseline median, 3*1.4826*MAD). Also save 3% and 10% sensitivity. The initial window ends immediately before that run. No return => right-censored. Later threshold exceedances are retained.',
        'recovery_caveat':'Descriptive local-return rule, not statistical significance or causal proof of recovery. A later spike can occur. Positive and negative excesses are both included in cumulative delay.',
        'coverage':'Partial treatment/baseline batches are excluded and listed. Post-injection retained steps must be consecutive. Every plotted point reports its repeat count.',
        'scope':'Event-window diagnostic of audited RTX4090/Qwen measurements; not a claim about the paper Figure 2 measurement window.'}
    validation={'passed':True,'source_audits_passed':True,'trace_matches_source_audits':True,
        'all_input_files_match_original_evidence_checksums':True,
        'formal_trials':sum(r['trials'] for r in reports.values()),
        'background_trials':len(trials),'prefill_only_trials':sum(a['batch_size']==0 for r in reports.values() for a in r['trial_audits']),
        'interference_trials':len({r['trial'] for r in aligned}),
        'aligned_iteration_rows':len(aligned),'event_statistics':event_stats,
        'exact_token_position_pairing':True,'post_steps_contiguous':True,
        'singleton_sse_events_only':True,'whole_request_tpot_recomputed':True,
        'client_window_sums_equal_endpoint_elapsed':True,'five_trials_per_window_group':True,
        'all_trials_have_minus5_to_plus63':True,'excluded_partial_steps':len(exclusions),
        'original_artifacts_modified':False}
    write_json(args.output/'verification.json',validation)
    snapshot={'schema':1,'method':method,'source_manifest_sha256':sha(args.output/'source-manifest.json'),
              'source_files':sources,'analysis_script_sha256':manifest['analysis_script_sha256'],
              'validation':validation,'groups':groups,
              'run_metadata':{run:json.loads((args.evidence/run/'run.json').read_text()) for run in args.runs},
              'prefill_only_groups':{run:reports[run].get('prefill_only_groups', []) for run in args.runs}}
    write_json(args.output/'plot-data.json',snapshot)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()

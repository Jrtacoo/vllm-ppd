#!/usr/bin/env python3
"""Audit Phase 10 raw requests and scheduler trace; derive paired event metrics."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics as st

from phase10_dense_interference import construct, triggers


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def describe(values):
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values)>1 else 0,
                n=len(values), min=min(values), max=max(values))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def analyze(evidence, name):
    folder=evidence/name
    meta=read(folder/'run.json'); trials=list(rows(folder/'trials.jsonl'))
    requests=list(rows(folder/'requests.jsonl')); pool=read(folder/'token-pool.json')
    check(len(trials)==len(meta['schedule']), 'Missing trials')
    check(all(all(t['checks'].values()) for t in trials), 'Client trial failed')
    check([{k:t[k] for k in ('batch_size','repeat','regime','scenario')} for t in trials]
          ==meta['schedule'], 'Trial schedule differs')
    by_trial=defaultdict(list)
    for r in requests: by_trial[r['trial']].append(r)
    request_map={r['request_id']:r for r in requests}
    check(len(request_map)==len(requests), 'Duplicate request ID')
    times={}; singles={}; bundled=[]; inputs={}; total_output=0
    for t in trials:
        tid=t['trial']; b=t['batch_size']; repeat=t['repeat']
        bg,intf=construct(pool,b,repeat,meta['seed']); k=len(intf)
        rr=by_trial[tid]
        expected_events=0 if t['regime']=='baseline' else len(triggers(b,t['regime'],repeat,meta['seed']))
        check(len(rr)==b+k+b+expected_events, tid+': request count')
        check(len(t['injections'])==expected_events, tid+': injection count')
        for role,count in [('warmup',b+k),('decode',b),('interferer',expected_events)]:
            check(len([r for r in rr if r['role']==role])==count, tid+': role count '+role)
        for r in rr:
            check(r['success'], r['request_id']+': request failed')
            stem=folder/r['raw_stem']; payload=read(Path(str(stem)+'.request.json'))
            response=read(Path(str(stem)+'.response.json'))
            check(response['success'] and response['request_id']==r['request_id'], 'Raw response mismatch')
            check(Path(str(stem)+'.sse').stat().st_size>0, 'Empty raw SSE')
            i=int(r['request_id'].rsplit('_',1)[1]); role=r['role']
            expected=(bg[i][:256] if i<b else intf[i-b][:2048]) if role=='warmup' else bg[i] if role=='decode' else intf[i]
            check(payload['prompt']==expected, r['request_id']+': input not reproducible')
            salt=f'{tid}/history/{i if role!="interferer" else b+i}'
            if role=='interferer' and t['scenario']=='full_prefill': salt=f'{tid}/cold/{i}'
            check(payload['cache_salt']==salt, 'Cache isolation mismatch')
            output=128 if role=='decode' else 1
            cached=256 if role=='decode' else 2048 if role=='interferer' and t['scenario']=='append_prefill' else 0
            check(r['prompt_tokens']==len(expected) and r['completion_tokens']==output and r['cached_tokens']==cached,
                  r['request_id']+': usage/cache mismatch')
            check(payload['max_tokens']==output and payload['ignore_eos'] and payload['temperature']==0,
                  'Generation controls differ')
            ts=[]; ss=[]; token_ids=[]
            for ev in read(Path(str(stem)+'.events.json')):
                if not isinstance(ev['data'],dict): continue
                for c in ev['data'].get('choices',[]):
                    ids=c.get('token_ids') or []
                    ts.extend([ev['perf_ns']]*len(ids));ss.extend([len(ids)==1]*len(ids));token_ids.extend(ids)
                    if len(ids)>1: bundled.append(dict(request=r['request_id'],count=len(ids),first=len(ts)-len(ids)+1))
            check(len(ts)==output, 'SSE token count mismatch')
            total_output+=len(ts)
            if role=='decode':
                check(abs((ts[-1]-ts[0])/1e6/127-r['token_tpot_ms'])<1e-7, 'TPOT recomputation failed')
                times[tid,i]=ts; singles[tid,i]=ss
            # Output equality is reported, not required: mixed prefill batch shapes can change rounding.
            inputs[r['request_id']]=token_ids
    steps=defaultdict(list); event_chunks=defaultdict(list); observed=set(); preemptions=[]
    observer=[]; max_tokens=0; max_seqs=0
    for record in rows(evidence/'scheduler-trace.jsonl'):
        rr=record['requests']; budget=sum(r['scheduled_tokens'] for r in rr)
        check(budget<=2048 and len(rr)<=251, 'Scheduler capacity exceeded')
        max_tokens=max(max_tokens,budget);max_seqs=max(max_seqs,len(rr))
        observer.append(record['observer_us'])
        if record['preempted_ids']: preemptions.append(record['step'])
        grouped=defaultdict(list)
        for r in rr:
            check(not r['preemptions'], 'Request preempted')
            rid=r['id'].removeprefix('cmpl-')
            if rid.endswith('-0'): rid=rid[:-2]
            if rid not in request_map: continue  # trace includes earlier pilot, not this run
            observed.add(rid); row=request_map[rid]
            grouped[row['trial']].append((row,r))
        for tid,items in grouped.items():
            pos={int(a['request_id'].rsplit('_',1)[1]):r['output_tokens_before'] for a,r in items
                 if a['role']=='decode' and r['computed_before']>=r['prompt_tokens'] and r['scheduled_tokens']==1}
            intfs=[(a,r) for a,r in items if a['role']=='interferer' and r['computed_before']<r['prompt_tokens']]
            s=dict(step=record['step'],model_ms=record['model_wall_ms'],positions=pos,
                   prefill_tokens=sum(r['scheduled_tokens'] for _,r in intfs),
                   start_ns=record['schedule_start_ns'],end_ns=record['update_return_ns'])
            if pos: steps[tid].append(s)
            for a,r in intfs:
                event_chunks[tid,a['event_index']].append(dict(step=s['step'],tokens=r['scheduled_tokens'],
                    computed_before=r['computed_before'],model_ms=s['model_ms'],background_count=len(pos),
                    min_background_output=min(pos.values(),default=-1),
                    server_queue_ms=(record['schedule_wall_ns']/1e9-r['arrival_wall_s'])*1000))
    check(observed==set(request_map), 'Requests missing from scheduler trace')
    check(not preemptions, 'Preemptions in trace')
    audits=[];trial_rows=[];windows=[];profiles=[];exclusions=[];events=[];trajectories=[]
    lookup={(t['batch_size'],t['repeat']):t for t in trials if t['regime']=='baseline'}
    gpu=list(rows(evidence/'gpu-samples.jsonl')) if (evidence/'gpu-samples.jsonl').exists() else []
    for t in trials:
        tid=t['trial']; b=t['batch_size']; allsteps=steps[tid]
        check(allsteps, 'No background decode steps')
        for seq in range(b):
            check(sorted(s['positions'][seq] for s in allsteps if seq in s['positions'])==list(range(1,128)),
                  tid+': missing/duplicate background decode steps')
        measured=[r for r in by_trial[tid] if r['role']=='decode']
        mean=st.mean(r['token_tpot_ms'] for r in measured)
        check(abs(mean-t['decode_avg_tpot_ms'])<1e-7, 'Trial TPOT mismatch')
        base=lookup[b,t['repeat']]; baseid=base['trial']
        baseindex={(seq,k):s for s in steps[baseid] for seq,k in s['positions'].items()}
        event_count=len(t['injections'])
        row=dict(trial=tid,batch=b,repeat=t['repeat'],regime=t['regime'],scenario=t['scenario'],
                 events=event_count,target_events=b/50 if t['regime']=='scaled' else event_count,tpot_ms=mean,baseline_tpot_ms=base['decode_avg_tpot_ms'],
                 excess_tpot_ms=mean-base['decode_avg_tpot_ms'],
                 total_span_excess_ms=127*(mean-base['decode_avg_tpot_ms']))
        # GPU timestamps and scheduler perf counters originate on the SAME server.
        lo=min(s['start_ns'] for s in allsteps);hi=max(s['end_ns'] for s in allsteps)
        samples=[]
        for g in gpu:
            if g['rc']==0 and lo<=g['perf_ns']<=hi:
                v=[x.strip() for x in g['values'].split(',')]
                samples.append(dict(util_pct=float(v[1]),memory_used_mib=float(v[3]),power_w=float(v[5]),temperature_c=float(v[6])))
        row['gpu']={k:describe([g[k] for g in samples]) for k in samples[0]} if samples else {}
        local={}
        for index,s in enumerate(allsteps):
            reason=None
            if len(s['positions'])!=b: reason='partial treatment batch'
            elif any(len(baseindex[seq,k]['positions'])!=b for seq,k in s['positions'].items()): reason='partial matched baseline batch'
            elif any(not(singles[q,seq][k] and singles[q,seq][k-1]) for seq,k in s['positions'].items() for q in (tid,baseid)):
                reason='bundled SSE hides individual gap'
            if reason:
                exclusions.append(dict(trial=tid,step=s['step'],reason=reason));continue
            gaps=[(times[tid,seq][k]-times[tid,seq][k-1])/1e6 for seq,k in s['positions'].items()]
            bases=[(times[baseid,seq][k]-times[baseid,seq][k-1])/1e6 for seq,k in s['positions'].items()]
            local[index]=dict(step=s['step'],mean_output_ordinal=st.mean(s['positions'].values()),
                client_ms=st.mean(gaps),baseline_client_ms=st.mean(bases),
                delta_ms=st.mean(gaps)-st.mean(bases),model_ms=s['model_ms'],
                model_delta_ms=s['model_ms']-st.mean(baseindex[seq,k]['model_ms'] for seq,k in s['positions'].items()),
                prefill_tokens=s['prefill_tokens'])
        trajectories.extend(dict(trial=tid,batch=b,repeat=t['repeat'],regime=t['regime'],scenario=t['scenario'],**s)
                            for s in local.values())
        event_positions=[]
        for injection in t['injections']:
            e=injection['event_index'];chunks=event_chunks[tid,e]
            check(chunks, tid+': missing interference event')
            check(all(c['background_count']==b for c in chunks), tid+': partial decode batch during prefill')
            check(chunks[0]['min_background_output']>=injection['trigger_tokens'], 'Actual injection too early')
            check(injection['trigger_tokens'] in triggers(b,t['regime'],repeat,meta['seed']), 'Unplanned injection checkpoint')
            expected=128 if t['scenario']=='append_prefill' else 2176
            check(sum(c['tokens'] for c in chunks)==expected, tid+': wrong prefill computation')
            computed=2048 if t['scenario']=='append_prefill' else 0
            for c in chunks:
                check(c['computed_before']==computed, 'Non-contiguous prefill chunks')
                computed+=c['tokens']
            first=next(j for j,s in enumerate(allsteps) if s['step']==chunks[0]['step'])
            last=next(j for j,s in enumerate(allsteps) if s['step']==chunks[-1]['step'])
            event_positions.append((first,last))
            er=dict(trial=tid,batch=b,repeat=t['repeat'],regime=t['regime'],scenario=t['scenario'],event=e,
                    trigger=injection['trigger_tokens'],chunks=chunks,first_step=chunks[0]['step'],last_step=chunks[-1]['step'],
                    server_queue_ms=chunks[0]['server_queue_ms'])
            events.append(er)
            for j in range(max(0,first-4),min(len(allsteps),first+21)):
                if j in local: profiles.append(dict(trial=tid,batch=b,repeat=t['repeat'],regime=t['regime'],
                    scenario=t['scenario'],event=e,offset=j-first,**local[j]))
            for width in (1,5,16,'chunks'):
                stop=last+1 if width=='chunks' else first+width
                ids=set(range(first,stop));valid=[local[j] for j in sorted(ids) if j in local]
                complete=len(valid)==len(ids)
                windows.append(dict(trial=tid,batch=b,repeat=t['repeat'],regime=t['regime'],scenario=t['scenario'],
                    event=e,width=width,complete=complete,expected_steps=len(ids),observed_steps=len(valid),
                    **({k:st.mean(s[k] for s in valid) for k in ('client_ms','baseline_client_ms','delta_ms','model_ms','model_delta_ms')} if complete else {})))
        # Merge overlapping windows by scheduled step before accumulating the event effect.
        unions={}
        for width in (1,5,16,'chunks'):
            sets=[set(range(first,last+1 if width=='chunks' else first+width)) for first,last in event_positions]
            union=set().union(*sets) if sets else set()
            valid=[local[j] for j in sorted(union) if j in local]
            unions[str(width)]=dict(steps=len(union),valid_steps=len(valid),
                duplicate_steps_removed=sum(map(len,sets))-len(union),complete=len(valid)==len(union),
                **({k:st.mean(s[k] for s in valid) for k in ('client_ms','baseline_client_ms','delta_ms','model_ms','model_delta_ms')} if valid else {}),
                cumulative_excess_ms=sum(s['delta_ms'] for s in valid))
        row['event_window_unions']=unions
        row['background_schedule_steps']=len(allsteps)
        row['background_decode_work']=sum(len(s['positions']) for s in allsteps)
        check(row['background_decode_work']==b*127, 'Background work does not match output length')
        row['prefill_chunks']=sum(len(event_chunks[tid,e]) for e in range(event_count))
        row['output_equal_to_baseline']=all(inputs[r['request_id']]==inputs[f'{baseid}_decode_{i}'] for i,r in
            ((int(r['request_id'].rsplit('_',1)[1]),r) for r in measured))
        trial_rows.append(row)
        audits.append(dict(trial=tid,background=b,events=event_count,chunks=row['prefill_chunks'],passed=True))
    grouped=defaultdict(list)
    for r in trial_rows: grouped[r['batch'],r['regime'],r['scenario']].append(r)
    groups=[]
    for (b,regime,scenario),rr in sorted(grouped.items()):
        groups.append(dict(batch=b,regime=regime,scenario=scenario,events=describe([r['events'] for r in rr]),
            **{k:describe([r[k] for r in rr]) for k in ('tpot_ms','baseline_tpot_ms','excess_tpot_ms','total_span_excess_ms')},
            first5={k:describe([r['event_window_unions']['5'][k] for r in rr if r['events'] and r['event_window_unions']['5']['complete']])
                    for k in ('client_ms','baseline_client_ms','delta_ms')}
                    if any(r['events'] and r['event_window_unions']['5']['complete'] for r in rr) else {}))
    report=dict(passed=True,trials=len(trials),requests=len(requests),output_tokens=total_output,
        max_scheduled_tokens=max_tokens,max_scheduled_sequences=max_seqs,preemptions=preemptions,
        observer_us=describe(observer),bundled_events=len(bundled),excluded_local_steps=len(exclusions),
        trial_audits=audits,source_sha256={n:sha(folder/n) for n in ('run.json','requests.jsonl','trials.jsonl','token-pool.json')},
        scheduler_sha256=sha(evidence/'scheduler-trace.jsonl'))
    save(folder/'verification.json',report)
    data=dict(method=dict(primary='Per request (last-first token arrival)/127; mean over background requests, then repeat mean ± sample SD.',
        event='Offset 0 is actual first prefill chunk. Compare identical background sequence/output ordinal in same-repeat baseline. Windows 1/5/16 plus complete prefill span. Exclude unresolved bundled intervals/partial batches; union overlapping windows before averaging.',
        queue='Server arrival to first prefill schedule wall time; not end-to-end network latency.',
        limitations='Single RTX 4090 adaptation; workload changes interference count with batch. A rising gap alone does not establish a batch-dependent hardware interference mechanism. GPU samples every 0.5 s do not resolve individual steps.'),
        verification=report,run=meta,groups=groups,trials=trial_rows,events=events)
    save(folder/'plot-data.json',data)
    for file,value in [('event-profiles.json',profiles),('step-trajectories.json',trajectories),('event-windows.json',windows),('excluded-local-steps.json',exclusions),('bundled-sse-events.json',bundled)]:save(folder/file,value)
    with (folder/'trial-summary.csv').open('w',newline='') as f:
        keys=['trial','batch','repeat','regime','scenario','events','tpot_ms','baseline_tpot_ms','excess_tpot_ms','total_span_excess_ms','prefill_chunks']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in trial_rows)
    print(json.dumps({k:report[k] for k in ('passed','trials','requests','output_tokens','max_scheduled_tokens','max_scheduled_sequences','bundled_events','excluded_local_steps')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path,required=True);p.add_argument('--run',required=True)
    args=p.parse_args()
    try:analyze(args.evidence,args.run)
    except Exception as exc:
        save(args.evidence/args.run/'verification.json',dict(passed=False,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':main()

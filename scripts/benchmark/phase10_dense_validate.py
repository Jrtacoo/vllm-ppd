#!/usr/bin/env python3
"""Independent synthetic checks for dense event allocation and raw-data audit.

These fixtures never enter experiment results and require no model/GPU.
"""
import copy
import json
from pathlib import Path
import statistics as st
import tempfile

from phase10_dense_analyze import analyze
from phase10_dense_interference import construct, triggers
from phase10_dense_run import verify_design


def save(p,x):p.write_text(json.dumps(x)+'\n')


def fixture(root,repeat):
    run=root/'run';(run/'raw').mkdir(parents=True)
    pool=[11,22,33];b=10;bg,inter=construct(pool,b,repeat,42)
    plans=[dict(batch_size=b,repeat=repeat,regime=regime,scenario=scenario) for regime,scenario in
           [('baseline','decode_only'),('scaled','append_prefill'),('scaled','full_prefill')]]
    save(run/'run.json',dict(seed=42,schedule=plans));save(run/'token-pool.json',pool)
    allrows=[];trials=[];trace=[];counter=0
    for trialno,meta in enumerate(plans):
        tid=f'synthetic-b{b}-r{repeat}-{meta["scenario"]}';sc=meta['scenario']
        goals=triggers(b,meta['regime'],repeat,42)
        first=(trialno+1)*10_000_000_000
        gaps=[20.]*127
        if goals:
            gaps[goals[0]-1]+=4 if sc=='append_prefill' else 70
            if sc=='full_prefill':gaps[goals[0]]+=10
        ts=[first]
        for g in gaps:ts.append(ts[-1]+int(g*1e6))
        def record(requests,start):
            nonlocal counter
            counter+=1
            trace.append(dict(step=counter,schedule_start_ns=start,schedule_wall_ns=start,
                update_return_ns=start+1_000_000,model_wall_ms=1.,observer_us=0.,preempted_ids=[],requests=requests))
        def entry(rid,tokens,before,prompt,output):
            return dict(id='cmpl-'+rid+'-0',scheduled_tokens=tokens,computed_before=before,
                        prompt_tokens=prompt,output_tokens_before=output,preemptions=0,arrival_wall_s=0.)
        def request(role,index,prompt,salt,tt,cache):
            rid=f'{tid}_{role}_{index}';stem=run/'raw'/rid
            row=dict(**meta,trial=tid,role=role,request_id=rid,success=True,prompt_tokens=len(prompt),
                     completion_tokens=len(tt),cached_tokens=cache,raw_stem='raw/'+rid,event_index=index if role=='interferer' else None)
            if role=='decode':row['token_tpot_ms']=(tt[-1]-tt[0])/1e6/127
            payload=dict(prompt=prompt,cache_salt=salt,max_tokens=len(tt),ignore_eos=True,temperature=0)
            save(Path(str(stem)+'.request.json'),payload);save(Path(str(stem)+'.response.json'),row)
            save(Path(str(stem)+'.events.json'),[dict(perf_ns=t,data={'choices':[{'token_ids':[7]}]}) for t in tt])
            Path(str(stem)+'.sse').write_text('data: {"choices":[{"token_ids":[7]}]}\n\n')
            allrows.append(row);return rid
        for i,prompt in enumerate([p[:256] for p in bg]+[p[:2048] for p in inter]):
            rid=request('warmup',i,prompt,f'{tid}/history/{i}',[first-2_000_000_000+i],0)
            record([entry(rid,len(prompt),0,len(prompt),0)],first-2_000_000_000+i)
        ids=[request('decode',i,p,f'{tid}/history/{i}',ts,256) for i,p in enumerate(bg)]
        record([entry(rid,32,256,288,0) for rid in ids],first-1_000_000)
        intid=None
        if goals:
            salt=f'{tid}/history/{b}' if sc=='append_prefill' else f'{tid}/cold/0'
            endk=goals[0] if sc=='append_prefill' else goals[0]+1
            intid=request('interferer',0,inter[0],salt,[ts[endk]],2048 if sc=='append_prefill' else 0)
        for k in range(1,128):
            rr=[entry(rid,1,288+k-1,288,k) for rid in ids]
            if goals and k==goals[0]:rr.append(entry(intid,128 if sc=='append_prefill' else 2038,2048 if sc=='append_prefill' else 0,2176,0))
            if goals and sc=='full_prefill' and k==goals[0]+1:rr.append(entry(intid,138,2038,2176,0))
            record(rr,ts[k]-2_000_000)
        trials.append(dict(**meta,trial=tid,checks={'synthetic':True},
            injections=[dict(event_index=0,trigger_tokens=goals[0])] if goals else [],decode_avg_tpot_ms=st.mean(gaps)))
    for name,rr in [('requests',allrows),('trials',trials)]:
        (run/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rr))
    (root/'scheduler-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trace))
    return trace


def main():
    verify_design();checks=['all 241 batches: exact stratified event counts, bounded checkpoints, zero-event repeats']
    with tempfile.TemporaryDirectory(prefix='phase10-dense-validation-') as td:
        root=Path(td)
        # A real dense-plan event at repeat5, and no event at repeat1.
        for repeat in (5,1):
            p=root/str(repeat);p.mkdir();trace=fixture(p,repeat);analyze(p,'run')
            data=json.loads((p/'run/plot-data.json').read_text())
            if repeat==5:
                full=next(t for t in data['trials'] if t['scenario']=='full_prefill')
                assert full['prefill_chunks']==2 and abs(full['event_window_unions']['5']['delta_ms']-16)<1e-9
                checks.append('full two chunks: exact 2176 computed tokens; matched five-step excess =16ms')
                bad=copy.deepcopy(trace);bad[0]['requests'][0]['scheduled_tokens']=2049
                (p/'scheduler-trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                try:analyze(p,'run')
                except ValueError as exc:assert 'capacity exceeded' in str(exc)
                else:raise AssertionError('Over-budget trace accepted')
                checks.append('reject scheduler token-budget violation')
            else:
                assert all(t['events']==0 and t['background_decode_work']==1270 for t in data['trials'])
                assert all(not t['event_window_unions']['5'].get('client_ms') for t in data['trials'])
                checks.append('zero-event treatments: retained whole-output TPOT; no invented event-window samples')
    print(json.dumps(dict(passed=True,synthetic_only=True,checks=checks),indent=2))


if __name__=='__main__':main()

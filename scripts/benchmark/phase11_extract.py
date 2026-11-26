#!/usr/bin/env python3
"""Extract genuine GPU boundaries and 10 kHz counter samples from read-only Nsight SQLite."""
import argparse
from bisect import bisect_left,bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics as st
import numpy as np

METRICS={'sm':'SMs Active [Throughput %]','read':'DRAM Read Bandwidth [Throughput %]',
         'write':'DRAM Write Bandwidth [Throughput %]'}

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def rid(x):
    x=x.removeprefix('cmpl-')
    return x[:-2] if x.endswith('-0') else x

def extract(db,worker,trace,trial,destination):
    c=sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True)
    names=dict(c.execute('select metricName,metricId from TARGET_INFO_GPU_METRICS'))
    metric_ids={k:names[v] for k,v in METRICS.items()}
    data={k:np.array(c.execute('select timestamp,value from GPU_METRICS where metricId=? order by timestamp',(mid,)).fetchall(),dtype=np.float64) for k,mid in metric_ids.items()}
    for a in data.values():assert len(a)>100 and np.all(np.diff(a[:,0])>0)
    ts=data['sm'][:,0].astype(np.int64)
    for a in data.values():assert np.array_equal(a[:,0],data['sm'][:,0])
    diagnostics=[dict(timestamp=r[0],severity=r[1],text=r[2]) for r in c.execute('select timestamp,severity,text from DIAGNOSTIC_EVENT')]
    bad=[x for x in diagnostics if x['severity']>=3 or any(t in x['text'].lower() for t in ('overflow','dropped','missing gpu','insufficient'))]
    assert not bad,bad
    assert all(np.all((a[:,1]>=0)&(a[:,1]<=100)) for a in data.values()),'Invalid percentage samples'
    gaps=np.diff(ts)
    intervals=dict(samples=len(ts),median_ns=float(np.median(gaps)),max_ns=int(max(gaps)),gaps_over_150us=int(sum(gaps>150000)),
                   diagnostics=diagnostics,invalid_samples=0)
    assert 95000<=np.median(gaps)<=105000,'Unexpected sample cadence'
    assert intervals['gaps_over_150us']==0,'GPU metric sampling gap detected'
    kernels=list(c.execute('select start,end,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    by_correlation=defaultdict(list)
    for a,b,cor in kernels:by_correlation[cor].append((a,b))
    runtimes=defaultdict(list)
    for a,b,tid,cor in c.execute('select start,end,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME order by start'):
        runtimes[tid].append((a,b,cor))
    runtime_ts={k:[x[0] for x in v] for k,v in runtimes.items()}
    runtime_correlations={x[2] for vv in runtimes.values() for x in vv}
    assert all(cor in runtime_correlations for _,_,cor in kernels),'Kernel without launch correlation'
    ranges=list(c.execute("select start,end,globalTid,text from NVTX_EVENTS where text like 'P11_%' order by start"))
    boundaries={}
    unmatched=[]
    for a,b,tid,label in ranges:
        if b is None:unmatched.append(label);continue
        kind,trial_id,index=label.split('|');assert trial_id==trial['trial']
        api=runtimes.get(tid,[]);times=runtime_ts.get(tid,[])
        kk=[k for _,_,cor in api[bisect_left(times,a):bisect_right(times,b)] for k in by_correlation.get(cor,[])]
        if not kk:unmatched.append(label);continue
        boundaries[kind,int(index)]=dict(start_ns=min(k[0] for k in kk),end_ns=max(k[1] for k in kk),kernels=len(kk),cpu_start_ns=a,cpu_end_ns=b)
    w=read(worker);assert w['trial']==trial['trial'] and w['enabled']
    empty_steps=[x['step'] for x in w['steps'] if not x['scheduled']]
    w['steps']=[x for x in w['steps'] if x['scheduled']]
    empty_boundaries={i:boundaries.pop(('P11_STEP',i),None) for i in empty_steps}
    unmatched=[x for x in unmatched if not (x.startswith('P11_STEP|') and int(x.rsplit('|',1)[1]) in empty_steps)]
    scheduler=[r for r in trace if any(trial['trial']+'_decode_' in rid(x['id']) or trial['trial']+'_interferer_' in rid(x['id']) for x in r['requests'])]
    assert len(scheduler)==len(w['steps']),(len(scheduler),len(w['steps']))
    steps=[]
    for ww,ss in zip(w['steps'],scheduler):
        assert {rid(k):v for k,v in ww['scheduled'].items()}=={rid(x['id']):x['scheduled_tokens'] for x in ss['requests']}
        i=ww['step'];body=boundaries['P11_FORWARD',i];outer=boundaries['P11_STEP',i]
        bg=[x for x in ss['requests'] if '_decode_' in rid(x['id'])]
        intr=[x for x in ss['requests'] if '_interferer_' in rid(x['id']) and x['computed_before']<x['prompt_tokens']]
        s=dict(step=i,scheduler_step=ss['step'],body=body,execute=outer,
             background=len(bg),background_outputs=[x['output_tokens_before'] for x in bg],
             scheduled_tokens=sum(ww['scheduled'].values()),prefill_tokens=sum(x['scheduled_tokens'] for x in intr),
             interference_computed_before=[x['computed_before'] for x in intr],
             background_prefill=any(x['computed_before']<x['prompt_tokens'] for x in bg))
        lo,hi=outer['start_ns'],outer['end_ns']
        # Metrics are interval averages: sample at timestamp T describes the preceding interval.
        ends=ts;starts=np.concatenate(([ts[0]-int(np.median(gaps))],ts[:-1]))
        weights=np.maximum(0,np.minimum(ends,hi)-np.maximum(starts,lo))
        assert weights.sum()>=.98*(hi-lo),'Incomplete metric coverage of a forward'
        s['gpu_execute_ms']=(hi-lo)/1e6;s['gpu_body_ms']=(body['end_ns']-body['start_ns'])/1e6
        s['metrics']={k:float(np.average(a[:,1],weights=weights)) for k,a in data.items()}
        steps.append(s)
    assert not unmatched,unmatched
    assert len(boundaries)==2*len(steps)
    intr=[s for s in steps if s['prefill_tokens']]
    assert sum(s['prefill_tokens'] for s in intr)==(0 if trial['scenario']=='decode_only' else 128 if trial['scenario']=='append_prefill' else 2176)
    assert all(s['background']==trial['batch_size'] for s in intr)
    lo=min(s['execute']['start_ns'] for s in steps);hi=max(s['execute']['end_ns'] for s in steps)
    mask=(ts>=lo-200000)&(ts<=hi+200000)
    destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/(trial['trial']+'.npz'),timestamp_ns=ts[mask],**{k:a[mask,1] for k,a in data.items()})
    report=dict(trial=trial['trial'],batch=trial['batch_size'],repeat=trial['repeat'],scenario=trial['scenario'],
        tpot_ms=trial['decode_avg_tpot_ms'],capture=dict(start_ns=lo,end_ns=hi),sample_quality=intervals,empty_cleanup_steps=empty_steps,empty_cleanup_gpu_bounds=empty_boundaries,
        metric_definitions=METRICS,steps=steps,missing_nvtx=unmatched)
    save(destination/(trial['trial']+'.json'),report)
    c.close();return report

def group(root):
    import phase10_dense_analyze as audit
    from phase11_resource_client import triggers
    audit.triggers=triggers
    audit.analyze(root,'on')
    derived=read(root/'on/plot-data.json')
    derived['method']['limitations']='Single RTX 4090; one fixed interference at client output 32; Nsight 100 us captures analyzed separately. Hardware curves use GPU kernel boundaries; CPU model_wall_ms remains a host span.'
    save(root/'on/plot-data.json',derived)
    trace=rows(root/'scheduler-trace.jsonl');trials=rows(root/'on/trials.jsonl')
    out=root/'extracted';out.mkdir(exist_ok=True)
    reports=[]
    for i,t in enumerate(trials,1):
        r=extract(root/f'profile.{i}.sqlite',root/(t['trial']+'.worker.json'),trace,t,out)
        reports.append({k:r[k] for k in ('trial','batch','repeat','scenario','tpot_ms','sample_quality')})
        print('EXTRACTED',t['trial'],len(r['steps']),flush=True)
    save(root/'resource-verification.json',dict(passed=True,trials=reports))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();group(a.root)

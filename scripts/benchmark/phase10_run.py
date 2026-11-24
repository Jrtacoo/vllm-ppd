#!/usr/bin/env python3
"""Collect the authorized Phase 10 pilot and formal matrix with isolated service."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import urllib.request

import phase10_operate as op

ROOT=Path(__file__).resolve().parents[2]
PY='/workspace/ppd-validation-env/bin/python'
NAMES=('phase10_interference.py','phase10_scheduler.py','phase10_analyze.py','phase10_operate.py','phase10_plot.py',
       'phase10_run.py','figure2_interference.py')


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def command(args,path):
    with path.open('x') as f:
        p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Command failed ({p.returncode}); retained {path}')


def export_trace(node,tag):
    token=str(time.time_ns());op.signal(node,tag,'export-request',token)
    ack_path=op.paths(node,tag)[2]/'runtime/scheduler-trace.exported'
    for _ in range(120):
        ack=op.host(node,f"from pathlib import Path;p=Path({str(ack_path)!r});print(p.read_text() if p.exists() else '')").strip()
        if ack==token:return
        time.sleep(.5)
    raise RuntimeError('Trace export timed out')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node',choices=('dcn9','dcn10'),required=True)
    p.add_argument('--evidence',type=Path,default=ROOT/'artifacts/phase10-multi-interference-evidence')
    args=p.parse_args();e=args.evidence.resolve();e.mkdir(exist_ok=True,parents=True)
    check=e/'runner-start.json'
    if check.exists():raise RuntimeError('Existing attempt retained; select a new evidence directory explicitly')
    source=e/'source-code';source.mkdir(exist_ok=True)
    for n in NAMES:shutil.copy2(ROOT/'scripts/benchmark'/n,source/n)
    shutil.copy2(ROOT/'artifacts/phase8-small-prefill-evidence/source-code/gpu-monitor.py',source/'gpu-monitor.py')
    captured=e/'captured';captured.mkdir(exist_ok=False)
    ip,name,_=op.paths(args.node,e.name);server='http://'+ip+':8300'
    save(check,dict(node=args.node,wall_ns=time.time_ns(),source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in source.glob('*.py')}))
    save(e/'plan.json',dict(background={'history':256,'new':32,'output':128,'independent':True},
        interference={'history':2048,'new':128,'full_computes':2176,'append_computes':128,'output':1},
        batches=[10,50,100,150,200,250],repeats=5,formal_trials=130,pilot_trials=3,
        interference_triggers={'10':[8],'50':[8],'100':[8,104],'150':[8,56,104],
                               '200':[8,40,72,104],'250':[8,32,56,80,104]},
        single_controls='B>=100; Append/Full at output 8 only; reuse first scaled event input',
        scheduler={'budget':2048,'max_seqs':251,'max_len':18432,'memory_utilization':.94,'chunked_prefill':True,'policy':'FCFS'},
        cache='Reset per trial; warm all independent histories identically in every condition; Full uses a cold salt.',
        primary='Background mean of 127 intertoken arrival gaps; repeat mean ± sample SD.',
        event='Actual schedule anchored; widths 1/5/16 and all prefill chunks; ordinal-matched baseline; deduplicate overlapping windows.',
        failures='Preserve and stop; no silent inference retries or batch reduction',
        machine=args.node,seed=42))
    save(e/'inventory-before.json',op.inventory())
    if args.node=='dcn10':
        # This CLI and the host share files, but may have different UIDs.
        # Give the host user ownership of ONLY this experiment's signal directory.
        identity=json.loads(op.host(args.node,"import os,json;print(json.dumps([os.getuid(),os.getgid()]))"))
        runtime=e/'runtime';runtime.mkdir(exist_ok=True)
        os.chown(runtime,identity[0],identity[1])
    launched=False
    try:
        (e/'stage.log').write_text(op.stage(args.node,e.name,source))
        # A busy GPU causes a preflight failure before any model is started.
        save(e/'container-command.json',op.launch(args.node,e.name));launched=True
        print('LAUNCHED',name,flush=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        ready=False
        for _ in range(240):
            try:
                with opener.open(server+'/health',timeout=2) as response:ready=response.status==200
            except Exception:ready=False
            if ready:break
            time.sleep(2)
        state=op.state(args.node,e.name);save(e/'ready-state.json',state)
        logs=state['logs']['out']+'\n'+state['logs']['err'];(e/'ready-server.log').write_text(logs)
        if not ready:raise RuntimeError('Server readiness timeout')
        capacities=re.findall(r'GPU KV cache size: ([\d,]+) tokens',logs)
        if not capacities:raise RuntimeError('Actual GPU KV cache capacity missing')
        capacity=int(capacities[-1].replace(',',''))
        save(e/'capacity-check.json',dict(actual_tokens=capacity,required_conservative_tokens=114240,
            note='250*ceil((288+127)/16)*16 + 5*2048; pilot also verifies no eviction of append histories, preemption or partial mixed batches.'))
        if capacity<114240:raise RuntimeError('KV capacity below pilot requirement; do not shrink batch')
        op.host(args.node,f"import subprocess;subprocess.run(['docker','exec','-d',{name!r},'python','/opt/figure2-tools/gpu-monitor.py'],check=True)")
        for run,repeats,batches,pilot in [('pilot-b250',1,[250],True),('formal',5,[10,50,100,150,200,250],False)]:
            client=[PY,str(source/'phase10_interference.py'),'--server',server,'--output',str(captured/run),
                '--namespace','phase10-v1-'+run,'--repeats',str(repeats),'--batches',*map(str,batches)]
            if pilot:client+=['--pilot']
            save(e/(run+'-command.json'),client)
            print('START',run,flush=True)
            error=None
            try:command(client,e/(run+'-client.log'))
            except Exception as exc:error=exc
            export_trace(args.node,e.name)
            op.fetch(args.node,e.name,captured)
            if pilot:shutil.copy2(captured/'scheduler-trace.jsonl',captured/'pilot-scheduler-trace.jsonl')
            if error:raise error
            command([PY,str(source/'phase10_analyze.py'),'--evidence',str(captured),'--run',run],e/(run+'-audit.log'))
            report=json.loads((captured/run/'verification.json').read_text())
            if not report['passed']:raise RuntimeError('Audit failed')
            print('PASS',run,report['trials'],report['requests'],flush=True)
        (e/'collection-complete').write_text(str(time.time_ns()))
    except Exception as exc:
        save(e/'failure.json',dict(error=f'{type(exc).__name__}: {exc}',wall_ns=time.time_ns()))
        raise
    finally:
        try:
            if launched:
                try:
                    op.signal(args.node,e.name,'monitor-stop','stop')
                    # Wait for atomic export completion, including for large traces.
                    export_trace(args.node,e.name)
                    op.fetch(args.node,e.name,captured)
                finally:
                    save(e/'before-cleanup-state.json',op.state(args.node,e.name))
                    (e/'cleanup.log').write_text(op.cleanup(args.node,e.name))
        finally:
            save(e/'inventory-after.json',op.inventory())
    print('COLLECTION COMPLETE; model removed; ready for plots/report',flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Phase11 scoped collector; immutable outputs, no inference retries."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import phase11_profile_operate as o
import phase11_session as sessions
import phase11_resource_client as client

ROOT=sessions.ROOT

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2)

def wait_health():
    deadline=time.monotonic()+360
    while time.monotonic()<deadline:
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open('http://192.168.1.9:8300/health',timeout=3) as r:
                if r.status==200:return
        except Exception:pass
        time.sleep(2)
    raise RuntimeError('Server not healthy; evidence retained')

def export_trace(label):
    token=str(time.time_ns())
    o.host(f'''import pathlib,os
p=pathlib.Path({str(o.REMOTE/'runtime/export-request')!r});t=p.with_suffix('.tmp');t.write_text({token!r});t.replace(p)''')
    for _ in range(30):
        ready=o.host(f'from pathlib import Path;print(Path({str(o.REMOTE/"runtime/exports"/(token+".json"))!r}).is_file())').strip()
        if ready=='True':break
        time.sleep(1)
    else:raise RuntimeError('Scheduler export timeout')
    dest=ROOT/label
    o.pull('exports/'+token+'.jsonl',dest/'scheduler-trace.jsonl')
    o.pull('exports/'+token+'.json',dest/'scheduler-trace-receipt.json')
    receipt=json.loads((dest/'scheduler-trace-receipt.json').read_text())
    data=(dest/'scheduler-trace.jsonl').read_bytes()
    assert hashlib.sha256(data).hexdigest()==receipt['sha256'] and len(data)==receipt['bytes']
    return token

def wait_reports(label,count):
    for _ in range(180):
        state=json.loads(o.host(f'''import pathlib,json,subprocess
p=pathlib.Path({str(o.REMOTE/'runtime'/label)!r})
t=subprocess.check_output(['docker','top',{o.NAME!r},'-eo','pid,args'],text=True)
print(json.dumps(dict(sql=[f.name for f in p.glob('*.sqlite')],active='--output=/evidence/{label}/profile' in t)))'''))
        if not state['active']:
            if len(state['sql'])!=count:raise RuntimeError('Report count mismatch '+str(state))
            return
        time.sleep(2)
    raise RuntimeError('Report export timeout; no files removed')

def fetch_reports(label,trials):
    dest=ROOT/label
    manifest=[]
    for i,t in enumerate(trials,1):
        for suffix in ('nsys-rep','sqlite'):
            name=f'{label}/profile.{i}.{suffix}'
            manifest.append(dict(trial=t['trial'],**o.pull(name,dest/f'profile.{i}.{suffix}')))
        manifest.append(dict(trial=t['trial'],**o.pull('worker/'+t['trial']+'.json',dest/(t['trial']+'.worker.json'))))
    for t in [x for x in (dest/'off'/'trials.jsonl').read_text().splitlines()] if (dest/'off'/'trials.jsonl').exists() else []:
        t=json.loads(t);manifest.append(o.pull('worker/'+t['trial']+'.json',dest/(t['trial']+'.worker.json')))
    manifest.append(o.pull(label+'/server.log',dest/'server.log'))
    save(dest/'transfer-manifest.json',manifest)

def run_client(label,batches,repeats,repeat_start=1,capture=True,sub='on'):
    args=argparse.Namespace(server='http://192.168.1.9:8300',output=ROOT/label/sub,
         namespace=label+'-'+sub,batches=batches,repeats=repeats,repeat_start=repeat_start,
         seed=42,pilot=label.startswith('pilot'),no_capture=not capture)
    asyncio.run(client.run(args))

def group(label,batches,repeats,pilot=False):
    if not (ROOT/label).exists():sessions.start(label)
    wait_health()
    # Equal worker hooks, with capture disabled, isolate additional profiler overhead.
    if pilot:run_client(label,batches,1,capture=False,sub='off')
    run_client(label,batches,repeats)
    trials=[json.loads(x) for x in (ROOT/label/'on/trials.jsonl').read_text().splitlines()]
    export_trace(label)
    print('STOP',sessions.stop(),flush=True)
    wait_reports(label,len(trials))
    fetch_reports(label,trials)
    print('GROUP COMPLETE',label,len(trials),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['pilot','formal']);a=p.parse_args()
    if a.mode=='pilot':group('pilot-01',[10,250],1,True)
    else:
        for b in (10,50,100,150,200,250):group(f'formal-b{b}',[b],5)

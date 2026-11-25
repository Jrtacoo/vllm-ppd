#!/usr/bin/env python3
"""Run the authorized dense Phase10 sweep, with bounded evidence and paired audits.

No inference retries. Failed shards remain unpacked. Successful raw evidence is
removed from scratch ONLY after a complete, content-verified archive is saved.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

import phase10_dense_operate as op
from phase10_dense_interference import CONTROL_BATCHES, phase_numerator, triggers

ROOT=Path(__file__).resolve().parents[2]
PY='/workspace/ppd-validation-env/bin/python'
PLOT_PY='/workspace/ppd-paper-tools/bin/python'
NAMES=('phase10_dense_run.py','phase10_dense_interference.py','phase10_dense_scheduler.py','phase10_dense_validate.py',
       'phase10_dense_operate.py','phase10_dense_analyze.py','phase10_dense_plot.py','figure2_interference.py')


def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def command(args,path):
    with path.open('x') as f:p=subprocess.run(args,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise RuntimeError(f'Command failed ({p.returncode}); retained {path}')


def export_trace(node,tag,allow_empty=False):
    token=str(time.time_ns());op.signal(node,tag,'export-request',token)
    runtime=op.paths(node,tag)[2]/'runtime'
    for _ in range(120):
        result=json.loads(op.host(node,f"import json;from pathlib import Path;p=Path({str(runtime/'exports'/(token+'.json'))!r});e=Path({str(runtime/'export-error.json')!r});print(json.dumps(dict(receipt=json.loads(p.read_text()) if p.exists() else None,error=json.loads(e.read_text()) if e.exists() else None)))"))
        if result['error'] and result['error']['token']==token:raise RuntimeError('Trace export failed: '+str(result['error']))
        receipt=result['receipt']
        if receipt:
            if receipt['token']!=token or receipt['protocol']!=2:raise RuntimeError('Invalid trace export receipt')
            if not allow_empty and receipt['records']==0:raise RuntimeError('Unexpected empty measured trace; snapshot retained')
            return token
        time.sleep(.5)
    raise RuntimeError('Trace export timed out; raw in-memory evidence not discarded before successful export')


def gpu_slice(node,tag,offset):
    p=op.paths(node,tag)[2]/'runtime/gpu-samples.jsonl'
    code=f'''import json
from pathlib import Path
p=Path({str(p)!r})
if not p.exists():print(json.dumps(dict(offset={offset},text='')))
else:
 with p.open('rb') as f:
  f.seek({offset});data=f.read()
 cut=data.rfind(b'\\n')+1
 print(json.dumps(dict(offset={offset}+cut,text=data[:cut].decode())))'''
    return json.loads(op.host(node,code))


def copy_remote(node,path,relative,evidence):
    """Copy one immutable file, verify host SHA, refuse conflicting evidence."""
    remote=op.HOST/'artifacts'/evidence.name/relative
    digest=sha(path)
    code=f'''import hashlib,pathlib,shutil,sys
p=pathlib.Path({str(remote)!r});p.parent.mkdir(parents=True,exist_ok=True)
def digest(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if p.exists():
 assert digest(p)=={digest!r},'Remote evidence differs: '+str(p)
 # Consume stdin even when already synchronized.
 for b in iter(lambda:sys.stdin.buffer.read(1048576),b''):pass
else:
 t=p.with_name(p.name+'.sync-tmp')
 with t.open('wb') as f:shutil.copyfileobj(sys.stdin.buffer,f,1048576)
 assert digest(t)=={digest!r},'Transport hash mismatch'
 t.replace(p)
print({digest!r})'''
    with path.open('rb') as f:
        p=subprocess.run([op.PY,op.HOST_EXEC,node,'python3 -c '+shlex.quote(code)],stdin=f,capture_output=True,text=True)
    if p.returncode or p.stdout.strip()!=digest:raise RuntimeError('Sync failed: '+str(relative)+' '+p.stderr[-500:])
    return dict(path=str(relative),sha256=digest,bytes=path.stat().st_size)


def archive_group(work,archive):
    files={str(p.relative_to(work)):dict(sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(work.rglob('*')) if p.is_file()}
    temporary=archive.with_name(archive.name+'.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=2) as tf:
        for rel in files:tf.add(work/rel,arcname=rel,recursive=False)
    observed={}
    with tarfile.open(temporary,'r:gz') as tf:
        for m in tf:
            if not m.isfile():raise RuntimeError('Unexpected archive member')
            f=tf.extractfile(m)
            observed[m.name]=dict(sha256=hashlib.file_digest(f,'sha256').hexdigest(),bytes=m.size)
    if files!=observed:raise RuntimeError('Raw evidence archive verification failed')
    temporary.replace(archive)
    return dict(passed=True,files=files,archive_sha256=sha(archive),bytes=archive.stat().st_size)


def verify_design():
    checks=[];plan={}
    for b in range(10,251):
        goals=[triggers(b,'scaled',r) for r in range(1,6)]
        assert sorted(phase_numerator(b,r) for r in range(1,6))==[1,3,5,7,9]
        for rr in goals:
            assert rr==sorted(set(rr)) and all(8<x<=104 for x in rr)
            assert len(rr)<=max(1,(b+49)//50)
        # Exact midpoint-grid rounding, including ties upward.
        assert sum(map(len,goals))==(b+5)//10
        plan[str(b)]=dict(target_events=b/50,actual_total_events=sum(map(len,goals)),
                          actual_mean_events=sum(map(len,goals))/5,
                          triggers_by_repeat={str(r):goals[r-1] for r in range(1,6)})
    assert sum(len(triggers(10,'scaled',r)) for r in range(1,6))==1
    assert all(len(triggers(250,'scaled',r))==5 for r in range(1,6))
    # Main output baseline and treatment keep paired independent inputs and cache warmup counts.
    checks=['all 241 batches; five stratified phases; 8<goal<=104; unique increasing checkpoints',
            'exact event rounding across repeats; zero-event repeats retained',
            'batch10 one total event/5 repeats; batch250 exactly five events/repeat']
    return dict(passed=True,checks=checks,trigger_plan=plan)


def make_schedule():
    out=[]
    for repeat in range(1,6):
        batches=list(range(10,251));random.Random(f'phase10-dense/batch-order/42/{repeat}').shuffle(batches)
        out.extend(dict(batch=b,repeat=repeat,kind='formal') for b in batches)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node',choices=('dcn9','dcn10'),default='dcn9')
    p.add_argument('--evidence',type=Path,default=ROOT/'artifacts/phase10-dense-workload-evidence')
    args=p.parse_args();e=args.evidence.resolve();e.mkdir(parents=True,exist_ok=True)
    if (e/'runner-start.json').exists():raise RuntimeError('Existing attempt retained; do not overwrite or rerun requests')
    design=verify_design();save(e/'offline-design-verification.json',design)
    source=e/'source-code';source.mkdir()
    for n in NAMES:shutil.copy2(ROOT/'scripts/benchmark'/n,source/n)
    shutil.copy2(ROOT/'artifacts/phase8-small-prefill-evidence/source-code/gpu-monitor.py',source/'gpu-monitor.py')
    command([PY,str(source/'phase10_dense_validate.py')],e/'offline-validation.log')
    frozen={x.name:sha(x) for x in source.glob('*.py')}
    formal=make_schedule()
    # Capacity/calibration at the six original batches, plus a zero-event low-batch case.
    event_repeat=next(r for r in range(1,6) if triggers(10,'scaled',r))
    zero_repeat=next(r for r in range(1,6) if not triggers(10,'scaled',r))
    pilot=[dict(batch=250,repeat=1,kind='pilot')]+[
        dict(batch=b,repeat=event_repeat if b==10 else 1,kind='pilot') for b in (10,50,100,150,200)]
    pilot.append(dict(batch=10,repeat=zero_repeat,kind='pilot'))
    schedule=pilot+formal
    save(e/'runner-start.json',dict(node=args.node,pid=os.getpid(),wall_ns=time.time_ns(),source_sha256=frozen))
    save(e/'plan.json',dict(background={'history':256,'new':32,'output':128,'independent':True},
        interference={'history':2048,'new':128,'full_computes':2176,'append_computes':128,'output':1},
        batches=list(range(10,251)),repeats=5,main_trials=3615,single_control_trials=60,total_formal_trials=3675,
        pilot_groups=len(pilot),work_per_event=4800,active_output_range=[8,104],
        phase_method='Five midpoint phases, deterministically permuted per batch; fractional event count allocated over repeats.',
        rounding='Across five repeats: nearest integer to B/10, ties upward; maximum mean-count error 0.1.',
        trigger_plan=design['trigger_plan'],schedule=schedule,single_control_batches=list(CONTROL_BATCHES),
        scheduler={'budget':2048,'max_seqs':251,'max_len':18432,'memory_utilization':.94,'chunked_prefill':True,'policy':'FCFS'},
        cache='Reset per trial; warm B backgrounds and ceil(B/50) independent interferer histories in EVERY condition.',
        primary='Mean request (last-first token arrival)/127, then mean over five repeats; includes zero-event repeats.',
        event='Actual first prefill schedule anchored; widths 1/5/16 and chunks, exact ordinal-matched baseline; report event/sample counts.',
        interpretation='Work-normalized injection, not fixed arrivals/second; dense sampling does not establish a fixed-event hardware slope.',
        failures='Retain and stop; no inference retries, no batch reduction, no selection by expected trend',
        evidence='Verified compressed raw archives per paired group; export and drain observer between idle groups only.',
        machine=args.node,seed=42))
    save(e/'inventory-before.json',op.inventory())
    for n in ('archives','summaries','archive-manifests','logs','scratch'):(e/n).mkdir()
    ip,name,_=op.paths(args.node,e.name);server='http://'+ip+':8300'
    launched=False;gpu_offset=0;completed=[];syncs=[];start=time.monotonic()
    status=dict(state='preparing',completed_groups=0,total_groups=len(schedule),formal_trials=0,
                total_formal_trials=3675,wall_ns=time.time_ns())
    save(e/'status.json',status)
    try:
        (e/'stage.log').write_text(op.stage(args.node,e.name,source))
        save(e/'container-command.json',op.launch(args.node,e.name));launched=True
        print('LAUNCHED',name,flush=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));ready=False
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
        if not capacities:raise RuntimeError('Actual GPU KV pool missing')
        capacity=int(capacities[-1].replace(',',''))
        save(e/'capacity-check.json',dict(actual_tokens=capacity,required_conservative_tokens=114240))
        if capacity<114240:raise RuntimeError('KV pool too small; no automatic batch reduction')
        op.host(args.node,f"import subprocess;subprocess.run(['docker','exec','-d',{name!r},'python','/opt/figure2-tools/gpu-monitor.py'],check=True)")
        # Sync frozen design now, all transfers happen outside measured inference.
        for q in [e/'plan.json',e/'offline-design-verification.json',e/'runner-start.json',*source.glob('*.py')]:
            syncs.append(copy_remote(args.node,q,q.relative_to(e),e))
        for seq,item in enumerate(schedule):
            b=item['batch'];r=item['repeat'];kind=item['kind'];gid=f'{kind}-b{b:03d}-r{r}'
            status.update(state=kind,current_group=gid,wall_ns=time.time_ns());save(e/'status.json',status)
            work=e/'scratch'/gid;work.mkdir()
            cmd=[PY,str(source/'phase10_dense_interference.py'),'--server',server,'--output',str(work/'run'),
                 '--namespace','p10dense-'+kind,'--repeats','1','--repeat-start',str(r),'--batches',str(b)]
            if kind=='pilot':cmd+=['--pilot']
            save(work/'client-command.json',cmd)
            print('START',seq+1,len(schedule),gid,flush=True)
            err=None
            try:command(cmd,work/'client.log')
            except Exception as exc:err=exc
            token=export_trace(args.node,e.name)
            op.fetch(args.node,e.name,work,token)
            # Avoid keeping a duplicate compressed transport copy inside the evidence archive.
            (work/'runtime-export.tar.gz').unlink()
            samples=gpu_slice(args.node,e.name,gpu_offset);gpu_offset=samples['offset']
            (work/'gpu-samples.jsonl').write_text(samples['text'])
            if err:raise err
            command([PY,str(source/'phase10_dense_analyze.py'),'--evidence',str(work),'--run','run'],work/'audit.log')
            summary=json.loads((work/'run/plot-data.json').read_text())
            if not summary['verification']['passed']:raise RuntimeError('Shard audit failed')
            summary['group']=item
            if b in CONTROL_BATCHES:
                summary['trajectories']=json.loads((work/'run/step-trajectories.json').read_text())
            save(e/'summaries'/(gid+'.json'),summary)
            arc=e/'archives'/(gid+'.tar.gz')
            manifest=archive_group(work,arc);save(e/'archive-manifests'/(gid+'.json'),manifest)
            syncs.append(copy_remote(args.node,arc,arc.relative_to(e),e))
            for q in [e/'summaries'/(gid+'.json'),e/'archive-manifests'/(gid+'.json')]:
                syncs.append(copy_remote(args.node,q,q.relative_to(e),e))
            # Content-verified local archive + remote verified copy now protect every raw file.
            shutil.rmtree(work)
            v=summary['verification']
            completed.append(dict(**item,id=gid,trials=v['trials'],requests=v['requests'],
                                  output_tokens=v['output_tokens'],archive_sha256=manifest['archive_sha256']))
            save(e/'completed-groups.json',completed)
            status.update(completed_groups=len(completed),formal_trials=sum(x['trials'] for x in completed if x['kind']=='formal'),
                          completed_requests=sum(x['requests'] for x in completed),elapsed_seconds=time.monotonic()-start,
                          wall_ns=time.time_ns())
            formal_done=sum(x['kind']=='formal' for x in completed)
            status['estimated_remaining_seconds']=((time.monotonic()-start)/len(completed))*(len(schedule)-len(completed))
            save(e/'status.json',status)
            print('PASS',gid,v['trials'],v['requests'],'formal',status['formal_trials'],'/3675',flush=True)
            if seq==len(pilot)-1:
                save(e/'pilot-verification.json',dict(passed=True,groups=completed.copy(),
                    note='Includes capacity B250, six representative baseline scheduling measurements, and zero-event treatment.'))
                print('PILOT PASS; BEGIN FULL SWEEP',flush=True)
        save(e/'collection-complete.json',dict(wall_ns=time.time_ns(),groups=len(completed),formal_trials=status['formal_trials']))
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',wall_ns=time.time_ns())
        save(e/'status.json',status);save(e/'failure.json',status)
        print('FAILED',status['error'],flush=True)
        raise
    finally:
        try:
            if launched:
                try:
                    op.signal(args.node,e.name,'monitor-stop','stop')
                    token=export_trace(args.node,e.name,allow_empty=True)
                    final=e/'final-runtime';final.mkdir(exist_ok=True)
                    op.fetch(args.node,e.name,final,token)
                    samples=gpu_slice(args.node,e.name,gpu_offset)
                    (final/'gpu-samples-tail.jsonl').write_text(samples['text'])
                finally:
                    save(e/'before-cleanup-state.json',op.state(args.node,e.name))
                    (e/'cleanup.log').write_text(op.cleanup(args.node,e.name))
        finally:save(e/'inventory-after.json',op.inventory())
    # Collection complete, release GPU before all final plotting and synchronization.
    command([PLOT_PY,str(source/'phase10_dense_plot.py'),'--evidence',str(e)],e/'analysis.log')
    # Immutable evidence + published report and plots, without overwriting existing remote changes.
    already={x['path'] for x in syncs}
    for q in sorted(e.rglob('*')):
        if not q.is_file() or '/scratch/' in str(q) or q.name in ('status.json','sync-verification.json'):continue
        rel=q.relative_to(e)
        if str(rel) not in already:syncs.append(copy_remote(args.node,q,rel,e))
    publication=[ROOT/'docs/TWO_NODE_PHASE10_DENSE.md',
                 *[ROOT/'scripts/benchmark'/n for n in NAMES if n.startswith('phase10_dense_')],
                 *sorted((ROOT/'results/figures/phase10/dense-workload').glob('*'))]
    for q in publication:
        if not q.is_file():continue
        # Preserve project-relative paths under a separate publication tree on host, then install with SHA checks.
        rel=Path('published')/q.relative_to(ROOT)
        syncs.append(copy_remote(args.node,q,rel,e))
    install_code=f'''import pathlib,shutil,hashlib
root=pathlib.Path({str(op.HOST)!r});pub=root/'artifacts'/{e.name!r}/'published'
for p in pub.rglob('*'):
 if not p.is_file():continue
 q=root/p.relative_to(pub);q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():assert q.read_bytes()==p.read_bytes(),'Remote publication differs: '+str(q)
 else:shutil.copy2(str(p),str(q))
print('Installed verified publications')'''
    op.host(args.node,install_code)
    save(e/'sync-verification.json',dict(passed=True,node=args.node,files=syncs,count=len(syncs)))
    status.update(state='complete',sync='complete',wall_ns=time.time_ns());save(e/'status.json',status)
    for q in (e/'sync-verification.json',e/'status.json'):
        copy_remote(args.node,q,q.relative_to(e),e)
    print('COMPLETE; figures generated, evidence audited and synchronized; no next stage started',flush=True)


if __name__=='__main__':main()

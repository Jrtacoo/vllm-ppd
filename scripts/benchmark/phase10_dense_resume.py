#!/usr/bin/env python3
"""Explicit audited resume; accepted shards never rerun, failed evidence retained."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import urllib.request

import phase10_dense_operate as op
import phase10_dense_run as run
from phase10_dense_interference import CONTROL_BATCHES

save=run.save
sha=run.sha


def gid(item):return f"{item['kind']}-b{item['batch']:03d}-r{item['repeat']}"


def remote_digest(node,path):
    return op.host(node,f'''import hashlib,pathlib
p=pathlib.Path({str(path)!r});h=hashlib.sha256()
if p.is_file():
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 print(h.hexdigest())
else:print('absent')''').strip()


def install(node,local,remote,expected):
    """Atomic mutable publication guarded by its observed previous hash."""
    digest=sha(local)
    code=f'''import pathlib,hashlib,sys,shutil
p=pathlib.Path({str(remote)!r});p.parent.mkdir(parents=True,exist_ok=True)
def digest(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
old=digest(p) if p.exists() else 'absent'
assert old in ({expected!r},{digest!r}),'Remote file changed; refusing overwrite: '+str(p)
t=p.with_name(p.name+'.resume-sync-tmp')
with t.open('wb') as f:shutil.copyfileobj(sys.stdin.buffer,f,1048576)
assert digest(t)=={digest!r},'Sync hash mismatch'
t.replace(p)
print({digest!r})'''
    with local.open('rb') as f:
        p=subprocess.run([op.PY,op.HOST_EXEC,node,'python3 -c '+shlex.quote(code)],stdin=f,capture_output=True,text=True)
    if p.returncode or p.stdout.strip()!=digest:raise RuntimeError('Guarded publication failed: '+str(remote)+' '+p.stderr[-600:])
    return dict(path=str(remote),sha256=digest,bytes=local.stat().st_size)


def preflight(root,e,a,node):
    plan=json.loads((e/'plan.json').read_text());completed=json.loads((e/'completed-groups.json').read_text())
    start=json.loads((e/'runner-start.json').read_text())
    for n,h in start['source_sha256'].items():assert sha(e/'source-code'/n)==h,'Original source snapshot changed'
    for n in ('phase10_dense_interference.py','phase10_dense_analyze.py','figure2_interference.py','phase10_dense_validate.py'):
        assert sha(root/'scripts/benchmark'/n)==start['source_sha256'][n],'Workload/audit changed: '+n
    schedule=plan['schedule'];assert 0<len(completed)<len(schedule)
    for item,done in zip(schedule,completed):
        assert all(item[k]==done[k] for k in ('batch','repeat','kind')) and gid(item)==done['id'],'Completion is not a schedule prefix'
        arc=e/'archives'/(done['id']+'.tar.gz');m=json.loads((e/'archive-manifests'/(done['id']+'.json')).read_text())
        assert sha(arc)==m['archive_sha256']==done['archive_sha256'],'Accepted archive hash changed'
        assert json.loads((e/'summaries'/(done['id']+'.json')).read_text())['verification']['passed']
    # Verify the previously synchronized remote archives in one host operation.
    expected={str(op.HOST/'artifacts'/e.name/'archives'/(x['id']+'.tar.gz')):x['archive_sha256'] for x in completed}
    result=json.loads(op.host(node,f'''import hashlib,json,pathlib
expected={expected!r}
for name,want in expected.items():
 h=hashlib.sha256()
 with pathlib.Path(name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==want,'Remote accepted archive differs: '+name
print(json.dumps(dict(passed=True,archives=len(expected))))'''))
    remaining=schedule[len(completed):];failed=json.loads((e/'failure.json').read_text())
    assert gid(remaining[0])==failed['current_group'],'Unexpected resume boundary'
    assert (e/'scratch'/failed['current_group']).is_dir(),'Original failed evidence missing'
    for n in ('status.json','failure.json','completed-groups.json'):
        shutil.copy2(e/n,a/('prior-'+n))
    save(a/'preflight-verification.json',dict(passed=True,accepted_groups=len(completed),
        accepted_formal_trials=sum(x['trials'] for x in completed if x['kind']=='formal'),
        remaining_groups=len(remaining),remaining_trials=3675-sum(x['trials'] for x in completed if x['kind']=='formal'),
        replacement_group=gid(remaining[0]),remote=result,
        original_failed_evidence=str(e/'scratch'/failed['current_group']),
        unchanged_workload=True,unchanged_audit=True))
    return plan,completed,remaining


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=Path('/workspace/vllm-ppd'))
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--attempt',default='resume1');p.add_argument('--node',default='dcn9',choices=('dcn9','dcn10'))
    args=p.parse_args();root=args.project_root.resolve();e=args.evidence.resolve();a=e/args.attempt
    assert re.fullmatch(r'resume[0-9]+',args.attempt)
    lock=(e/'resume.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.mkdir();(a/'scratch').mkdir();run.ROOT=root
    status=json.loads((e/'status.json').read_text());launched=False;completed=[];syncs=[]
    tag='phase10-dense-workload-'+args.attempt+'-runtime'
    ip,name,_=op.paths(args.node,tag);server='http://'+ip+':8300'
    gpu_offset=0;started=time.monotonic();initial_count=0
    save(a/'runner-start.json',dict(node=args.node,pid=os.getpid(),wall_ns=time.time_ns(),tag=tag,attempt=args.attempt))
    try:
        plan,completed,remaining=preflight(root,e,a,args.node);initial_count=len(completed)
        source=a/'source-code';source.mkdir()
        names=(*run.NAMES,'phase10_dense_resume.py','phase10_dense_export_validate.py')
        for n in names:shutil.copy2(root/'scripts/benchmark'/n,source/n)
        shutil.copy2(e/'source-code/gpu-monitor.py',source/'gpu-monitor.py')
        start=json.loads((a/'runner-start.json').read_text());start['source_sha256']={x.name:sha(x) for x in source.glob('*.py')}
        save(a/'runner-start.json',start)
        for script in ('phase10_dense_validate.py','phase10_dense_export_validate.py'):
            run.command([run.PY,str(source/script)],a/(script+'.log'))
        publications=[root/'docs/TWO_NODE_PHASE10_DENSE.md',*[root/'scripts/benchmark'/n for n in names if n.startswith('phase10_dense_')]]
        # Capture remote hashes now; final publication cannot clobber concurrent edits.
        expected={str(q.relative_to(root)):remote_digest(args.node,op.HOST/q.relative_to(root)) for q in publications}
        mutable={n:remote_digest(args.node,op.HOST/'artifacts'/e.name/n) for n in ('status.json','completed-groups.json','sync-verification.json')}
        save(a/'publication-before.json',dict(project=expected,evidence=mutable))
        save(a/'inventory-before.json',op.inventory())
        status.update(state='preparing-resume',attempt=args.attempt,pid=os.getpid(),current_group=None,wall_ns=time.time_ns())
        status.pop('error',None);save(e/'status.json',status)
        (a/'stage.log').write_text(op.stage(args.node,tag,source))
        save(a/'container-command.json',op.launch(args.node,tag));launched=True
        print('LAUNCHED',name,flush=True)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));ready=False
        for _ in range(240):
            try:
                with opener.open(server+'/health',timeout=2) as response:ready=response.status==200
            except Exception:ready=False
            if ready:break
            time.sleep(2)
        state=op.state(args.node,tag);save(a/'ready-state.json',state)
        logs=state['logs']['out']+'\n'+state['logs']['err'];(a/'ready-server.log').write_text(logs)
        if not ready:raise RuntimeError('Server readiness timeout')
        capacities=re.findall(r'GPU KV cache size: ([\d,]+) tokens',logs)
        if not capacities:raise RuntimeError('Actual GPU KV pool not found')
        capacity=int(capacities[-1].replace(',',''));assert capacity>=114240,'KV capacity below unchanged requirement'
        save(a/'capacity-check.json',dict(actual_tokens=capacity,required_conservative_tokens=114240))
        op.host(args.node,f"import subprocess;subprocess.run(['docker','exec','-d',{name!r},'python','/opt/figure2-tools/gpu-monitor.py'],check=True)")
        for q in [a/'runner-start.json',a/'preflight-verification.json',*source.glob('*.py')]:
            syncs.append(run.copy_remote(args.node,q,q.relative_to(e),e))
        # Resume pilot is excluded from formal totals and completed-groups prefix.
        todo=[dict(kind='resume-pilot',batch=250,repeat=1),*remaining]
        for seq,item in enumerate(todo):
            b=item['batch'];r=item['repeat'];kind=item['kind'];group=gid(item);pilot=kind=='resume-pilot'
            status.update(state='resume-pilot' if pilot else 'formal',current_group=group,wall_ns=time.time_ns())
            save(e/'status.json',status)
            work=a/'scratch'/group;work.mkdir()
            cmd=[run.PY,str(source/'phase10_dense_interference.py'),'--server',server,'--output',str(work/'run'),
                 '--namespace','p10dense-pilot' if pilot else 'p10dense-formal','--repeats','1','--repeat-start',str(r),'--batches',str(b)]
            if pilot:cmd+=['--pilot']
            save(work/'client-command.json',cmd);print('START',seq+1,len(todo),group,flush=True)
            error=None
            try:run.command(cmd,work/'client.log')
            except Exception as exc:error=exc
            token=run.export_trace(args.node,tag);op.fetch(args.node,tag,work,token)
            (work/'runtime-export.tar.gz').unlink()
            samples=run.gpu_slice(args.node,tag,gpu_offset);gpu_offset=samples['offset']
            (work/'gpu-samples.jsonl').write_text(samples['text'])
            if error:raise error
            run.command([run.PY,str(source/'phase10_dense_analyze.py'),'--evidence',str(work),'--run','run'],work/'audit.log')
            summary=json.loads((work/'run/plot-data.json').read_text());assert summary['verification']['passed']
            summary['group']=item;summary['execution_attempt']=args.attempt
            if not pilot and seq==1:
                summary['replaces_failed_attempt']=str(Path('scratch')/group)
            if b in CONTROL_BATCHES:summary['trajectories']=json.loads((work/'run/step-trajectories.json').read_text())
            target=a if pilot else e
            for directory in ('summaries','archives','archive-manifests'):(target/directory).mkdir(exist_ok=True)
            summary_path=target/'summaries'/(group+'.json');arc=target/'archives'/(group+'.tar.gz')
            manifest_path=target/'archive-manifests'/(group+'.json')
            assert not any(q.exists() for q in (summary_path,arc,manifest_path)),'Would overwrite accepted shard'
            save(summary_path,summary);manifest=run.archive_group(work,arc);save(manifest_path,manifest)
            for q in (arc,summary_path,manifest_path):syncs.append(run.copy_remote(args.node,q,q.relative_to(e),e))
            shutil.rmtree(work)
            v=summary['verification']
            if pilot:
                save(a/'pilot-verification.json',dict(passed=True,excluded_from_formal=True,verification=v,export_protocol=2))
                print('RESUME PILOT PASS',flush=True)
            else:
                completed.append(dict(**item,id=group,trials=v['trials'],requests=v['requests'],output_tokens=v['output_tokens'],
                    archive_sha256=manifest['archive_sha256'],execution_attempt=args.attempt))
                save(e/'completed-groups.json',completed)
                status.update(completed_groups=len(completed),formal_trials=sum(x['trials'] for x in completed if x['kind']=='formal'),
                    completed_requests=sum(x['requests'] for x in completed),resume_elapsed_seconds=time.monotonic()-started,
                    wall_ns=time.time_ns())
                count=len(completed)-initial_count
                status['estimated_remaining_seconds']=(time.monotonic()-started)/count*(len(plan['schedule'])-len(completed))
                save(e/'status.json',status)
                # Versioned checkpoints are immutable and safe to synchronize during collection.
                if count==1 or count%25==0:
                    for n in ('completed-groups','status'):
                        q=a/'progress'/f'{n}-{len(completed):04d}.json'
                        save(q,completed if n=='completed-groups' else status)
                        syncs.append(run.copy_remote(args.node,q,q.relative_to(e),e))
                print('PASS',group,v['trials'],v['requests'],'formal',status['formal_trials'],'/3675',flush=True)
        save(e/'collection-complete.json',dict(wall_ns=time.time_ns(),groups=len(completed),formal_trials=status['formal_trials'],attempt=args.attempt))
        status.update(state='finalizing',wall_ns=time.time_ns());save(e/'status.json',status)
        # GPU is released before generating all final plots.
        op.signal(args.node,tag,'monitor-stop','stop')
        token=run.export_trace(args.node,tag,allow_empty=True)
        final=a/'final-runtime';final.mkdir();op.fetch(args.node,tag,final,token)
        samples=run.gpu_slice(args.node,tag,gpu_offset);(final/'gpu-samples-tail.jsonl').write_text(samples['text'])
        save(a/'before-cleanup-state.json',op.state(args.node,tag))
        (a/'cleanup.log').write_text(op.cleanup(args.node,tag));launched=False
        save(a/'inventory-after.json',op.inventory())
        run.command([run.PLOT_PY,str(source/'phase10_dense_plot.py'),'--evidence',str(e)],a/'analysis.log')
        # Plot publication paths are new; capture before installation, never overwrite different figures.
        figures=sorted((root/'results/figures/phase10/dense-workload').glob('*'))
        for q in figures:expected[str(q.relative_to(root))]='absent'
        for q in [*publications,*figures]:
            syncs.append(install(args.node,q,op.HOST/q.relative_to(root),expected[str(q.relative_to(root))]))
        already={x['path'] for x in syncs}
        # All accepted archives were checked at resume or synchronized per shard.
        for q in [*e.glob('*.json'),*a.rglob('*'),*e.glob('*.log')]:
            if not q.is_file() or 'scratch' in q.relative_to(e).parts or q.name=='operator.log':continue
            rel=str(q.relative_to(e))
            if rel in already or rel in mutable:continue
            syncs.append(run.copy_remote(args.node,q,Path(rel),e));already.add(rel)
        syncs.append(install(args.node,e/'completed-groups.json',op.HOST/'artifacts'/e.name/'completed-groups.json',mutable['completed-groups.json']))
        save(e/'sync-verification.json',dict(passed=True,node=args.node,attempt=args.attempt,files=syncs,count=len(syncs),
            accepted_before_resume=len(completed)-len(remaining),accepted_archives_rechecked=True,
            live_operator_log='Excluded from immutable sync while runner owns stdout; local original retained.'))
        install(args.node,e/'sync-verification.json',op.HOST/'artifacts'/e.name/'sync-verification.json',mutable['sync-verification.json'])
        status.update(state='complete',sync='complete',wall_ns=time.time_ns());save(e/'status.json',status)
        install(args.node,e/'status.json',op.HOST/'artifacts'/e.name/'status.json',mutable['status.json'])
        print('COMPLETE; audited, plotted, synchronized; no next stage started',flush=True)
    except BaseException as exc:
        status.update(state='failed',attempt=args.attempt,error=f'{type(exc).__name__}: {exc}',wall_ns=time.time_ns())
        save(e/'status.json',status);save(a/'failure.json',status)
        print('FAILED',status['error'],flush=True)
        raise
    finally:
        if launched:
            try:
                op.signal(args.node,tag,'monitor-stop','stop')
                token=run.export_trace(args.node,tag,allow_empty=True)
                final=a/'failure-runtime';final.mkdir(exist_ok=True);op.fetch(args.node,tag,final,token)
            except Exception as exc:save(a/'cleanup-export-error.json',dict(error=repr(exc)))
            finally:
                try:
                    save(a/'before-cleanup-state.json',op.state(args.node,tag))
                    (a/'cleanup.log').write_text(op.cleanup(args.node,tag))
                except Exception as exc:save(a/'cleanup-error.json',dict(error=repr(exc)))
            save(a/'inventory-after.json',op.inventory())


if __name__=='__main__':main()

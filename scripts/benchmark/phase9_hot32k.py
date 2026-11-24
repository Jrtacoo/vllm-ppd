#!/usr/bin/env python3
"""Serial hot-P/cold-D Phase 9 extension. Only named experiment containers mutate."""
import argparse
import asyncio
import base64
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import urllib.request

import aiohttp

ROOT = Path('/workspace/vllm-ppd')
E = ROOT/'artifacts/two-node-phase9-kv-transport-evidence/extension-hot32k'
REMOTE = '/tmp/ppd-phase9-hot32k-v3'
IMAGE = 'ppd-qwen25:source-cu129'
IMAGE_ID = 'sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad'
OPS = '/workspace/conda-env/bin/python3.11'
HELPER = '/workspace/ppd-ops/host_exec.py'
IPS = {'dcn9':'192.168.1.9', 'dcn10':'192.168.1.11'}
MODEL = 'Qwen2.5-7B-Instruct'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def host(node, code, timeout=180):
    p = subprocess.run([OPS, HELPER, node, 'python3 -c '+shlex.quote(code)],
                       capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(node+': '+p.stdout[-2000:]+p.stderr[-1000:])
    return p.stdout


def inventory():
    code = '''import json,subprocess,socket,time
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=40);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
print(json.dumps(dict(node=socket.gethostname(),wall_ns=time.time_ns(),gpu=run(['nvidia-smi']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),containers=run(['docker','ps','--format','{{.Names}} {{.Status}} {{.Image}}']),original_proxy=run(['docker','inspect','ppd-dcn9-proxy']),image=run(['docker','image','inspect','ppd-qwen25:source-cu129']))))'''
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        return dict(zip(IPS, pool.map(lambda n:json.loads(host(n, code)), IPS)))


def prepare():
    if (E/'plan.json').exists():
        raise FileExistsError('Existing experiment plan; do not overwrite')
    E.mkdir(parents=True, exist_ok=True)
    source = E/'source-code'
    source.mkdir()
    for name in ('phase9_hot32k.py','phase9_hot32k_runtime.py'):
        (source/name).write_bytes((ROOT/'scripts/benchmark'/name).read_bytes())
    telemetry = ROOT/'artifacts/two-node-phase9-kv-transport-evidence/telemetry/p2p_nccl_engine.py'
    (source/'p2p_nccl_engine.py').write_bytes(telemetry.read_bytes())
    write(E/'source-hashes.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source.iterdir()})
    write(E/'initial-inventory.json', inventory())
    write(E/'plan.json', dict(input_tokens=32768, output_tokens=1,
        kv_payload_bytes=32768*57344, model=MODEL, prefix_cache={'prefill':True,'decode':False},
        sequence=['socket','rdma','rdma','socket'], formal_requests_per_deployment=5,
        warmups_per_deployment={'cold':1,'hot':1}, max_model_len=32769,
        max_num_batched_tokens=2048, max_num_seqs=1, gpu_memory_utilization=0.85,
        kv_buffer_size=2**31, mem_pool_size_gb=2,
        evidence_scope='Hot-P/cold-D application transfer; not pure network latency',
        gpu_policy='Require no non-experiment compute process before each deployment unless explicitly authorized',
        isolated_proxy={'http':10101,'zmq':30101}, model_http={'prefill':8100,'decode':8200}))
    write(E/'status.json',dict(state='prepared',message='Awaiting GPU availability / user preference'))


def stage():
    sources = {p.name:base64.b64encode(p.read_bytes()).decode() for p in (E/'source-code').iterdir()}
    for node in IPS:
        code = f'''import pathlib,base64,json
p=pathlib.Path({REMOTE!r});p.mkdir(exist_ok=True)
source=p/'source';source.mkdir(exist_ok=True)
for name,blob in {sources!r}.items():
 data=base64.b64decode(blob);target=source/name
 if target.exists():assert target.read_bytes()==data,'Existing snapshot differs'
 else:target.write_bytes(data)
print('source staged')'''
        (E/f'stage-{node}.txt').write_text(host(node, code))


def get(url):
    with OPENER.open(url,timeout=5) as response:
        body=response.read()
        return json.loads(body) if body else True


def ready(url, timeout=600):
    end=time.monotonic()+timeout
    last=''
    attempts=0
    while time.monotonic()<end:
        try:return get(url)
        except Exception as exc:last=str(exc)
        attempts+=1
        if attempts % 5 == 0 and url.endswith('/health'):
            node,role=('dcn9','prefill') if ':8100/' in url else ('dcn10','decode')
            state=json.loads(host(node,f'import subprocess;print(subprocess.check_output(["docker","inspect",{name(role)!r}],text=True))'))[0]['State']
            if state['Status'] in ('exited','dead'):
                raise RuntimeError('Model container exited during startup: '+role)
        time.sleep(2)
    raise RuntimeError('Readiness timeout '+url+': '+last)


def name(role):return 'ppd-p9hot-'+role


def deploy(transport, block, allow_shared=False):
    directory=E/f'block{block}-{transport}'
    directory.mkdir(exist_ok=False)
    inv=inventory();write(directory/'before-inventory.json', inv)
    for node,data in inv.items():
        if data['processes']['rc']!=0:raise RuntimeError('GPU inventory failed')
        if data['processes']['out'].strip() and not allow_shared:
            raise RuntimeError('GPU busy on '+node+'; no workload stopped')
    stage()
    commands=[]
    for node,role in [('dcn9','proxy'),('dcn9','prefill'),('dcn10','decode')]:
        config=json.loads((ROOT/'configs/two_node'/f'{node}.json').read_text())
        config.update(max_model_len=32769,max_num_batched_tokens=2048,max_num_seqs=1,
            gpu_memory_utilization=0.85,kv_buffer_size=2**31,mem_pool_size_gb=2,
            proxy_zmq_port=30101,transport=transport)
        runtime=f'{REMOTE}/block{block}-{role}'
        cfgpath=runtime+'/config.json'
        cmd=['docker','create','--name',name(role),'--label','ppd.experiment=phase9-hot32k',
            '--init','--stop-timeout','60','--network','host','--shm-size','2g',
            '--ulimit','memlock=-1:-1','--mount',f'type=bind,source={cfgpath},target=/node-config.json,readonly',
            '--mount',f'type=bind,source={REMOTE}/source,target=/opt/hot32k-tools,readonly',
            '--mount',f'type=bind,source={runtime},target=/evidence',
            '--workdir','/opt/ppd','--entrypoint','python']
        env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1',
             'PYTHONPATH':'/opt/hot32k-tools:/opt/ppd:/opt/ppd/vllm-source',
             'VLLM_LOGGING_LEVEL':'DEBUG','VLLM_SERVER_DEV_MODE':'1',
             'VLLM_ALLOW_LONG_MAX_MODEL_LEN':'1',
             'NO_PROXY':'localhost,127.0.0.1,192.168.1.9,192.168.1.11',
             'no_proxy':'localhost,127.0.0.1,192.168.1.9,192.168.1.11'}
        if role!='proxy':
            cmd+=['--gpus','device=0','--mount','type=bind,source=/home/dell/xwd/models/Qwen2.5-7B-Instruct,target=/workspace/models/Qwen2.5-7B-Instruct,readonly']
            if transport=='rdma':
                cmd+=['--device','/dev/infiniband']
                env.update(NCCL_IB_GID_INDEX='3',NCCL_IB_ADDR_FAMILY='AF_INET',
                    NCCL_IB_ROCE_VERSION_NUM='2',NCCL_DEBUG_SUBSYS='INIT,NET,GRAPH,ENV')
        cmd += [x for k,v in env.items() for x in ('--env',k+'='+v)]
        cmd += [IMAGE,'/opt/hot32k-tools/phase9_hot32k_runtime.py','--role',role,'--config','/node-config.json']
        write(directory/f'command-{role}.json',dict(command=cmd,config=config,environment=env))
        code=f'''import subprocess,pathlib,json,socket,time
assert json.loads(subprocess.check_output(['docker','image','inspect',{IMAGE!r}]))[0]['Id']=={IMAGE_ID!r}
assert subprocess.run(['docker','inspect',{name(role)!r}],capture_output=True).returncode!=0,'Existing container'
p=pathlib.Path({runtime!r});p.mkdir(parents=True,exist_ok=False)
(p/'config.json').write_text(json.dumps({config!r}))
for port in {((10101,30101) if role=='proxy' else ((8100,14579) if role=='prefill' else (8200,14580)))!r}:
 for attempt in range(30):
  s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:s.bind(({IPS[node]!r},port));break
  except OSError:
   if attempt==29:raise
   time.sleep(1)
  finally:s.close()
print(subprocess.check_output({cmd!r},text=True))
'''
        (directory/f'create-{role}.txt').write_text(host(node, code))
        if role!='proxy':
            code=f'''import subprocess,hashlib,json,tarfile,io,pathlib
c={name(role)!r};path='/opt/ppd/vllm-source/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py'
original=subprocess.check_output(['docker','cp',c+':'+path,'-'])
with tarfile.open(fileobj=io.BytesIO(original)) as tf:
 original_source=tf.extractfile(next(m for m in tf.getmembers() if m.isfile())).read()
original_sha=hashlib.sha256(original_source).hexdigest()
assert original_sha=='81d1d8c39e509b5cfc733e5c5a8abbeac888f2d2fab5ec3362575cc4d086dace','Unexpected original engine source'
subprocess.run(['docker','cp',{REMOTE!r}+'/source/p2p_nccl_engine.py',c+':'+path],check=True)
print(json.dumps(dict(original_source_sha256=original_sha,instrumented_source_sha256=hashlib.sha256(pathlib.Path({REMOTE!r}+'/source/p2p_nccl_engine.py').read_bytes()).hexdigest())))'''
            (directory/f'telemetry-{role}.txt').write_text(host(node,code))
        commands.append((node,role))
    host('dcn9',f'import subprocess;subprocess.run(["docker","start",{name("proxy")!r}],check=True)')
    ready('http://192.168.1.9:10101/status',60)
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda item:host(item[0],f'import subprocess;subprocess.run(["docker","start",{name(item[1])!r}],check=True)'),commands[1:]))
    ready('http://192.168.1.9:8100/health');ready('http://192.168.1.11:8200/health')
    status=ready('http://192.168.1.9:10101/status',60)
    write(directory/'ready-status.json',status)
    if status.get('config')!='1P_1D':raise RuntimeError('Wrong routing')
    return directory


def capture(directory,label):
    for node,role in [('dcn9','proxy'),('dcn9','prefill'),('dcn10','decode')]:
        code=f'''import subprocess,json,pathlib,time
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=60);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
r=dict(logs=run(['docker','logs','--timestamps',{name(role)!r}]),inspect=run(['docker','inspect',{name(role)!r}]),gpu=run(['nvidia-smi']))
print(json.dumps(r))'''
        data=json.loads(host(node,code))
        write(directory/f'{label}-{role}-state.json',data)
        (directory/f'{label}-{role}.log').write_text(data['logs']['out']+data['logs']['err'])
    for node,role in [('dcn9','prefill'),('dcn10','decode')]:
        with OPENER.open(f'http://{IPS[node]}:{8100 if role=="prefill" else 8200}/metrics',timeout=10) as r:
            (directory/f'{label}-{role}-metrics.txt').write_bytes(r.read())
        code='''import json,pathlib,subprocess
p=pathlib.Path('/sys/class/infiniband/rocep202s0f0/ports/1/counters')
print(json.dumps(dict(roce={n:int((p/n).read_text()) for n in ('port_xmit_data','port_rcv_data','port_xmit_packets','port_rcv_packets')},interface=json.loads(subprocess.check_output(['ip','-s','-j','link','show','dev','ens7f0np0'])))))'''
        write(directory/f'{label}-{role}-counters.json',json.loads(host(node,code)))


def export_trace(directory,block):
    token=str(time.time_ns())
    for node,role in [('dcn9','prefill'),('dcn10','decode')]:
        runtime=f'{REMOTE}/block{block}-{role}'
        code=f'''import pathlib,time
p=pathlib.Path({runtime!r});(p/'export-request').write_text({token!r})
end=time.monotonic()+15
while time.monotonic()<end:
 if (p/'export-ack').exists() and (p/'export-ack').read_text()=={token!r}:break
 time.sleep(.1)
else:raise RuntimeError('trace export timeout')
print((p/'scheduler-trace.json').read_text())'''
        write(directory/f'{role}-scheduler-trace.json',json.loads(host(node,code)))


def cleanup():
    for node,role in [('dcn9','prefill'),('dcn10','decode'),('dcn9','proxy')]:
        code=f'''import subprocess,json
c={name(role)!r};p=subprocess.run(['docker','inspect',c],capture_output=True,text=True)
if p.returncode==0:
 obj=json.loads(p.stdout)[0]
 assert obj['Config']['Labels'].get('ppd.experiment')=='phase9-hot32k'
 subprocess.run(['docker','stop','--time','60',c],check=True)
 subprocess.run(['docker','rm',c],check=True)
print('cleaned only '+c)'''
        print(host(node,code),flush=True)


async def prompt():
    if (E/'manifest.json').exists():return json.loads((E/'manifest.json').read_text())
    text='Session phase9-hot32k-v1. Read these reference notes:'
    tail='\nSummarize the reference notes.\nAssistant:'
    fill=32700
    async with aiohttp.ClientSession(trust_env=False) as session:
        for _ in range(12):
            value=text+' data'*fill+tail
            async with session.post('http://192.168.1.11:8200/tokenize',json={'model':MODEL,'prompt':value,'add_special_tokens':True}) as r:
                r.raise_for_status();data=await r.json()
            count=data['count']
            if count==32768:
                result=dict(prompt=value,input_tokens=count,sha256=hashlib.sha256(value.encode()).hexdigest())
                write(E/'manifest.json',result);return result
            fill+=32768-count
    raise RuntimeError('Exact token construction failed')


async def request(directory,kind,index,manifest):
    rawdir=directory/'raw';rawdir.mkdir(exist_ok=True)
    stem=rawdir/f'{kind}-{index}'
    body=dict(model=MODEL,prompt=manifest['prompt'],max_tokens=1,temperature=0,
        seed=42,ignore_eos=True,stream=True,stream_options={'include_usage':True})
    write(stem.with_suffix('.request.json'),body)
    row=dict(kind=kind,index=index,prompt_sha256=manifest['sha256'],success=False,
        ttft_ms=None,e2e_ms=None,response_id=None,error=None)
    raw=bytearray();events=[];done=False;usage={};finish=None
    start=time.perf_counter()
    try:
        async with aiohttp.ClientSession(trust_env=False,timeout=aiohttp.ClientTimeout(total=300)) as session:
            async with session.post('http://192.168.1.9:10101/v1/completions',json=body) as r:
                row['http_status']=r.status
                if r.status!=200:
                    raw.extend(await r.read());raise RuntimeError(bytes(raw).decode()[:1500])
                async for line in r.content:
                    raw.extend(line)
                    line=line.decode().strip()
                    if not line.startswith('data:'):continue
                    value=line[5:].strip();elapsed=(time.perf_counter()-start)*1000
                    if value=='[DONE]':done=True;events.append(dict(ms=elapsed,data=value));continue
                    item=json.loads(value);events.append(dict(ms=elapsed,data=item))
                    if 'error' in item:raise RuntimeError(str(item['error']))
                    if item.get('id'):row['response_id']=item['id']
                    if item.get('usage'):usage=item['usage']
                    for c in item.get('choices',[]):
                        if c.get('text') and row['ttft_ms'] is None:row['ttft_ms']=elapsed
                        if c.get('finish_reason'):finish=c['finish_reason']
        row.update(e2e_ms=(time.perf_counter()-start)*1000,usage=usage,finish_reason=finish)
        if not(done and usage.get('prompt_tokens')==32768 and usage.get('completion_tokens')==1 and finish=='length'):
            raise RuntimeError('Unexpected response '+str(row))
        row['success']=True
    except Exception as exc:
        row.update(error=repr(exc),e2e_ms=(time.perf_counter()-start)*1000)
    stem.with_suffix('.sse').write_bytes(raw);write(stem.with_suffix('.events.json'),events)
    write(stem.with_suffix('.response.json'),row)
    with (directory/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='usage'}),flush=True)
    if not row['success']:raise RuntimeError('Failed request retained at '+str(stem))
    return row


async def run(allow_shared=False,resume=False):
    if (E/'run-started.json').exists() and not resume:raise FileExistsError('Run already started; preserve evidence')
    if resume:
        assert json.loads((E/'status.json').read_text())['state']=='failed', 'Only a failed run can be resumed'
    write(E/('resume-started.json' if resume else 'run-started.json'),dict(wall_ns=time.time_ns(),allow_shared=allow_shared))
    try:
        for block,transport in enumerate(['socket','rdma','rdma','socket'],1):
            completed=E/f'block{block}-{transport}'
            if resume and (completed/'after-formal-decode.log').exists() and (completed/'decode-scheduler-trace.json').exists():
                formal=list((completed/'raw').glob('formal-*.response.json'))
                assert len(formal)==5 and all(json.loads(p.read_text())['success'] for p in formal)
                print('Preserving completed block '+str(block),flush=True)
                continue
            write(E/'status.json',dict(state='deploying',block=block,transport=transport))
            directory=deploy(transport,block,allow_shared)
            manifest=await prompt()
            capture(directory,'ready')
            await request(directory,'cold-warmup',1,manifest)
            await request(directory,'hot-warmup',1,manifest)
            capture(directory,'before-formal')
            export_trace(directory,block)
            warm_check={}
            for role,tag in [('prefill','TX'),('decode','RX')]:
                values=[json.loads(line.split('PPD_PHASE9_'+tag+' ',1)[1])
                    for line in (directory/f'before-formal-{role}.log').read_text().splitlines()
                    if 'PPD_PHASE9_'+tag+' ' in line]
                assert len(values)==2 and all(v['layers']==28 and v['payload_bytes']==32768*57344 for v in values), 'Warmup full-payload check failed'
                trace=json.loads((directory/f'{role}-scheduler-trace.json').read_text())
                scheduled={}
                for step in trace:
                    for r in step['requests']:
                        scheduled[r['id']]=scheduled.get(r['id'],0)+r['scheduled_tokens']
                assert list(scheduled.values())==([32768,16] if role=='prefill' else [1,1]), 'Warmup compute/cache check failed: '+str(scheduled)
                warm_check[role]=dict(scheduled=scheduled,payloads=values)
            write(directory/'warmup-verification.json',dict(passed=True,roles=warm_check))
            for index in range(1,6):
                write(E/'status.json',dict(state='measuring',block=block,transport=transport,index=index))
                await request(directory,'formal',index,manifest)
            # HTTP response is not an acknowledgement of asynchronous send completion.
            # Check that all per-request TX/RX records exist before stable export/cleanup.
            for attempt in range(30):
                counts=[]
                for node,role,tag in [('dcn9','prefill','TX'),('dcn10','decode','RX')]:
                    out=host(node,f'import subprocess;print(subprocess.check_output(["docker","logs",{name(role)!r}],stderr=subprocess.STDOUT,text=True))')
                    counts.append(len(re.findall('PPD_PHASE9_'+tag+' ',out)))
                if counts==[7,7]:break
                time.sleep(.5)
            else:raise RuntimeError('Asynchronous transport not drained: '+str(counts))
            capture(directory,'after-formal');export_trace(directory,block)
            cleanup()
        write(E/'status.json',dict(state='measurement-complete',formal_requests=20,warmups=8))
    except BaseException as exc:
        write(E/'status.json',dict(state='failed',error=repr(exc)))
        try:
            if 'directory' in locals():
                capture(directory,'failure');export_trace(directory,block)
        except Exception as capture_error:write(E/'capture-error.json',dict(error=repr(capture_error)))
        raise
    finally:
        cleanup()
        write(E/'final-inventory.json',inventory())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','inventory','run','cleanup'])
    p.add_argument('--allow-shared-gpu',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='inventory':print(json.dumps(inventory(),indent=2))
    elif args.action=='cleanup':cleanup()
    else:asyncio.run(run(args.allow_shared_gpu,args.resume))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Isolated Phase 10 container operations; never stops an existing workload."""
import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HOST=Path('/home/dell/xwd/vllm-ppd')
IMAGE='ppd-qwen25:source-cu129'
IMAGE_ID='sha256:ae90422b157e0517caafcb7a4a4756bd2eb845eec7963d7b2f01b166cafce0ad'
PY='/workspace/conda-env/bin/python3.11'
HOST_EXEC='/workspace/ppd-ops/host_exec.py'


def host(node,code):
    p=subprocess.run([PY,HOST_EXEC,node,'python3 -c '+shlex.quote(code)],capture_output=True,text=True)
    if p.returncode:raise RuntimeError(f'{node}: {p.stderr[-1500:]}\n{p.stdout[-1500:]}')
    return p.stdout


def inventory():
    code='''import json,subprocess,socket,time
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
names=['ppd-dcn9-proxy','ppd-dcn9-prefill'] if socket.gethostname()=='dcn9' else ['ppd-dcn10-decode']
print(json.dumps(dict(node=socket.gethostname(),wall_ns=time.time_ns(),containers={n:run(['docker','inspect',n]) for n in names},image=run(['docker','image','inspect','ppd-qwen25:source-cu129']),gpu=run(['nvidia-smi']),processes=run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),docker=run(['docker','ps','--format','{{.Names}} {{.Image}} {{.Status}}']))))'''
    return {node:json.loads(host(node,code)) for node in ('dcn9','dcn10')}


def paths(node,tag):
    check=tag.replace('-','').replace('_','').isalnum()
    if not check:raise ValueError('Invalid experiment tag')
    return '192.168.1.11' if node=='dcn10' else '192.168.1.9',f'ppd-phase10-{tag}-{node}',HOST/'artifacts'/tag


def stage(node,tag,source):
    _,_,remote=paths(node,tag)
    # Exact snapshot only; never copies source into the compiled vLLM path.
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tf:
        for p in source.glob('*.py'):tf.add(p,arcname=p.name)
    blob=base64.b64encode(buf.getvalue()).decode()
    return host(node,f'''import base64,io,tarfile,pathlib
p=pathlib.Path({str(remote)!r});p.mkdir(parents=True,exist_ok=True)
source=p/'source-code';source.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode({blob!r})),mode='r:gz') as tf:
 for m in tf.getmembers():
  assert '/' not in m.name and m.isfile()
  target=source/m.name
  data=tf.extractfile(m).read()
  if target.exists():assert target.read_bytes()==data,'Existing snapshot differs'
  else:target.write_bytes(data)
(p/'runtime').mkdir(exist_ok=True)
print(p)
''')


def launch(node,tag):
    ip,name,remote=paths(node,tag)
    command=['docker','run','--detach','--name',name,'--init','--stop-timeout','60',
        '--network','host','--shm-size','2g','--ulimit','memlock=-1:-1','--gpus','device=0',
        '--mount',f'type=bind,source={remote}/source-code,target=/opt/figure2-tools,readonly',
        '--mount',f'type=bind,source={remote}/runtime,target=/evidence',
        '--mount','type=bind,source=/home/dell/xwd/models/Qwen2.5-7B-Instruct,target=/workspace/models/Qwen2.5-7B-Instruct,readonly',
        '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','VLLM_SERVER_DEV_MODE=1',
        '--env','PYTHONPATH=/opt/figure2-tools:/opt/ppd:/opt/ppd/vllm-source',
        '--env','FIGURE2_TRACE=/evidence/scheduler-trace.jsonl','--env','VLLM_LOGGING_LEVEL=INFO',
        '--env',f'NO_PROXY=localhost,127.0.0.1,{ip}','--env',f'no_proxy=localhost,127.0.0.1,{ip}',
        '--workdir','/opt/ppd','--entrypoint','python',IMAGE,'-m','vllm.entrypoints.openai.api_server',
        '--model','/workspace/models/Qwen2.5-7B-Instruct','--served-model-name','Qwen2.5-7B-Instruct',
        '--host',ip,'--port','8300','--dtype','bfloat16','--seed','42','--max-model-len','18432',
        '--max-num-seqs','251','--max-num-batched-tokens','2048','--gpu-memory-utilization','0.94',
        '--block-size','16','--enable-prefix-caching','--enforce-eager','--generation-config','vllm',
        '--enable-prompt-tokens-details','--scheduler-cls','phase10_scheduler.TraceScheduler']
    code=f'''import subprocess,socket,json,pathlib
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not busy,'GPU is in use; no processes stopped: '+busy
existing=subprocess.run(['docker','container','inspect',{name!r}],capture_output=True)
assert existing.returncode!=0,'Experiment container already exists'
s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(({ip!r},8300));s.close()
image=json.loads(subprocess.check_output(['docker','image','inspect',{IMAGE!r}]))[0]
assert image['Id']=={IMAGE_ID!r},'Image changed'
runtime=pathlib.Path({str(remote/'runtime')!r})
assert not list(runtime.iterdir()),'Existing runtime evidence; do not overwrite'
print(subprocess.check_output({command!r},text=True))'''
    return dict(node=node,command=command,output=host(node,code))


def state(node,tag):
    _,name,_=paths(node,tag)
    return json.loads(host(node,f'''import subprocess,json
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=45);return dict(rc=p.returncode,out=p.stdout,err=p.stderr)
print(json.dumps(dict(inspect=run(['docker','inspect',{name!r}]),logs=run(['docker','logs','--timestamps',{name!r}]),gpu=run(['nvidia-smi']))))'''))


def cleanup(node,tag):
    _,name,_=paths(node,tag)
    return host(node,f'''import subprocess,json
obj=json.loads(subprocess.check_output(['docker','inspect',{name!r}]))[0]
assert obj['Config']['Image']=={IMAGE!r} and obj['Config']['Cmd'][-1]=='phase10_scheduler.TraceScheduler'
subprocess.run(['docker','stop','--time','60',{name!r}],check=True)
subprocess.run(['docker','rm',{name!r}],check=True)
print('Removed only '+{name!r})''')


def signal(node,tag,filename,value):
    if filename not in ('monitor-stop','export-request'):raise ValueError('Invalid signal')
    _,_,remote=paths(node,tag)
    return host(node,f'from pathlib import Path;Path({str(remote/"runtime"/filename)!r}).write_text({value!r})')


def fetch(node,tag,destination):
    _,_,remote=paths(node,tag)
    archive=destination/'runtime-export.tar.gz'
    allowed=('scheduler-trace.jsonl','scheduler-trace.exported','gpu-samples.jsonl','export-request','monitor-stop')
    files=json.loads(host(node,f'import json;from pathlib import Path;p=Path({str(remote/"runtime")!r});print(json.dumps([n for n in {allowed!r} if (p/n).is_file()]))'))
    if not files:raise RuntimeError('No stable runtime evidence to export')
    with archive.open('wb') as f:
        p=subprocess.run([PY,HOST_EXEC,node,shlex.join(['tar','-czf','-','-C',str(remote/'runtime'),*files])],stdout=f,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if not m.isfile():continue
            name=Path(m.name).name
            if name not in ('scheduler-trace.jsonl','scheduler-trace.exported','gpu-samples.jsonl','export-request','monitor-stop'):continue
            (destination/name).write_bytes(tf.extractfile(m).read())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--node',choices=['dcn9','dcn10'],required=True)
    p.add_argument('--tag',default='phase10-multi-interference-evidence')
    p.add_argument('action',choices=['inventory','state','cleanup'])
    args=p.parse_args()
    result=inventory() if args.action=='inventory' else globals()[args.action](args.node,args.tag)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

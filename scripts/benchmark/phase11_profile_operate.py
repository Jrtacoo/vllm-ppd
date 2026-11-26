#!/usr/bin/env python3
"""Scoped Phase11 operations. No host driver/network/image changes."""
import json
from pathlib import Path
import subprocess
import shlex
import phase10_dense_operate as base

NODE='dcn9'
TAG='phase11-resource-timeline-evidence'
REMOTE=base.HOST/'artifacts'/TAG
NAME='ppd-phase11-profile-dcn9'
NSYS='/usr/local/cuda-12.9/nsight-systems-2025.1.3/target-linux-x64/nsys'
NCU='/usr/local/cuda-12.9/nsight-compute-2025.2.1/ncu'


def host(code):return base.host(NODE,code)


def launch():
    args=['docker','run','--detach','--name',NAME,'--init','--stop-timeout','60',
        '--network','host','--shm-size','2g','--ulimit','memlock=-1:-1','--gpus','device=0',
        '--cap-add','SYS_ADMIN',
        '--mount',f'type=bind,source={REMOTE}/source-code,target=/opt/phase11-tools,readonly',
        '--mount',f'type=bind,source={REMOTE}/runtime,target=/evidence',
        '--mount','type=bind,source=/home/dell/xwd/models/Qwen2.5-7B-Instruct,target=/workspace/models/Qwen2.5-7B-Instruct,readonly',
        '--mount','type=bind,source=/usr/local/cuda-12.9/nsight-systems-2025.1.3,target=/usr/local/cuda-12.9/nsight-systems-2025.1.3,readonly',
        '--mount','type=bind,source=/usr/local/cuda-12.9/nsight-compute-2025.2.1,target=/usr/local/cuda-12.9/nsight-compute-2025.2.1,readonly',
        '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','VLLM_SERVER_DEV_MODE=1',
        '--env','PYTHONPATH=/opt/phase11-tools:/opt/ppd:/opt/ppd/vllm-source',
        '--env','NO_PROXY=localhost,127.0.0.1,192.168.1.9',
        '--env','no_proxy=localhost,127.0.0.1,192.168.1.9',
        '--env','FIGURE2_TRACE=/evidence/scheduler-trace.jsonl',
        '--workdir','/opt/ppd','--entrypoint','sleep',base.IMAGE,'infinity']
    return json.loads(host(f'''import subprocess,pathlib,json,socket
p=pathlib.Path({str(REMOTE)!r});(p/'source-code').mkdir(parents=True,exist_ok=True);(p/'runtime').mkdir(exist_ok=True)
busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not busy,'GPU busy; no tasks stopped: '+busy
assert subprocess.run(['docker','inspect',{NAME!r}],capture_output=True).returncode!=0,'Existing container retained'
image=json.loads(subprocess.check_output(['docker','image','inspect',{base.IMAGE!r}]))[0]
assert image['Id']=={base.IMAGE_ID!r}
s=socket.socket();s.bind(('192.168.1.9',8300));s.close()
print(json.dumps(dict(command={args!r},container=subprocess.check_output({args!r},text=True).strip())))'''))


def exec_tool(args,log,background=False):
    command=['docker','exec',NAME,*args]
    return host(f'''import subprocess,pathlib,json
p=pathlib.Path({str(REMOTE/'runtime'/log)!r});p.parent.mkdir(parents=True,exist_ok=True)
with p.open('x') as f:
 p=subprocess.Popen({command!r},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 {'print(json.dumps(dict(pid=p.pid)))' if background else "rc=p.wait();print(json.dumps(dict(rc=rc)));assert rc==0"}
''')


def pull(relative,dest):
    """Copy an immutable remote file, verify digest, never expose host secrets."""
    remote=REMOTE/'runtime'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise RuntimeError('Existing local evidence: '+str(dest))
    code=f'''import hashlib,pathlib,sys
p=pathlib.Path({str(remote)!r})
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):sys.stdout.buffer.write(b)'''
    with dest.open('xb') as f:
        p=subprocess.run([base.PY,base.HOST_EXEC,NODE,'python3 -c '+shlex.quote(code)],stdout=f,stderr=subprocess.PIPE)
    if p.returncode:raise RuntimeError(p.stderr.decode())
    import hashlib
    with dest.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    remote_sha=host(f'''import hashlib,pathlib
h=hashlib.sha256()
with pathlib.Path({str(remote)!r}).open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(h.hexdigest())''').strip()
    assert digest==remote_sha,'Remote transfer hash mismatch'
    return dict(path=str(relative),sha256=digest,bytes=dest.stat().st_size)


def cleanup():
    return host(f'''import subprocess,json
x=json.loads(subprocess.check_output(['docker','inspect',{NAME!r}]))[0]
assert x['Config']['Image']=={base.IMAGE!r} and x['Config']['Entrypoint']==['sleep']
subprocess.run(['docker','stop','--time','60',{NAME!r}],check=True)
subprocess.run(['docker','rm',{NAME!r}],check=True)
print('Removed only '+{NAME!r})''')

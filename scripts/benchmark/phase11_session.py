"""Start/stop only this phase's Nsight model process inside its isolated container."""
import json
from pathlib import Path
import time
import phase11_profile_operate as o

ROOT=Path('/workspace/vllm-ppd/artifacts')/o.TAG

def start(label):
    args=[o.NSYS,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
        '--cuda-event-trace=false','--gpu-metrics-devices=0','--gpu-metrics-set=ad10x',
        '--gpu-metrics-frequency=10000','--capture-range=cudaProfilerApi','--capture-range-end=repeat',
        '--export=sqlite',f'--output=/evidence/{label}/profile',
        'python','-m','vllm.entrypoints.openai.api_server','--model','/workspace/models/Qwen2.5-7B-Instruct',
        '--served-model-name','Qwen2.5-7B-Instruct','--host','192.168.1.9','--port','8300',
        '--dtype','bfloat16','--seed','42','--max-model-len','18432','--max-num-seqs','251',
        '--max-num-batched-tokens','2048','--gpu-memory-utilization','0.94','--block-size','16',
        '--enable-prefix-caching','--enforce-eager','--generation-config','vllm',
        '--enable-prompt-tokens-details','--scheduler-cls','phase10_dense_scheduler.TraceScheduler',
        '--worker-cls','phase11_profile_worker.ProfileWorker']
    d=ROOT/label;d.mkdir(exist_ok=False)
    (d/'server-command.json').write_text(json.dumps(args,indent=2))
    print(o.exec_tool(args,label+'/server.log',True),flush=True)

def stop():
    # Inspect /proc only INSIDE our named container; never signal a host workload.
    code='''import os,signal,json
found=[]
for n in os.listdir('/proc'):
 if not n.isdigit():continue
 try:a=open('/proc/'+n+'/cmdline','rb').read().split(b'\\0')
 except OSError:continue
 if len(a)>3 and a[1:3]==[b'-m',b'vllm.entrypoints.openai.api_server']:
  found.append(int(n))
assert len(found)==1,found
os.kill(found[0],signal.SIGINT);print(json.dumps(found))'''
    return o.host(f'import subprocess;print(subprocess.check_output({["docker","exec",o.NAME,"python","-c",code]!r},text=True))')

if __name__=='__main__':
    import sys
    print(start(sys.argv[2]) if sys.argv[1]=='start' else stop())

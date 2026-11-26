"""Excluded tool-capability probe: two short CUDA captures, no model results."""
import json
import time
import torch

x=torch.randn((2048,2048),device='cuda',dtype=torch.bfloat16)
for _ in range(10):y=x@x
torch.cuda.synchronize()
for i in range(2):
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push('P11_PROBE|'+str(i))
    for _ in range(200):y=x@x
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    time.sleep(1)
print(json.dumps(dict(probe_only=True,captures=2,torch=torch.__version__,cuda=torch.version.cuda)))

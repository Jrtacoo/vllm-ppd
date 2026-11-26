"""NVTX/CUDA capture hooks; no per-forward synchronization or measured file I/O."""
import json
from pathlib import Path
import time
import torch
from vllm.v1.worker.gpu_worker import Worker

class ProfileWorker(Worker):
    def load_model(self, *args, **kwargs):
        result=super().load_model(*args, **kwargs)
        self._p11_trial=None;self._p11_steps=[];self._p11_enabled=False
        original=self.model_runner._model_forward
        def forward(*a,**kw):
            if self._p11_trial is None:return original(*a,**kw)
            torch.cuda.nvtx.range_push(f'P11_FORWARD|{self._p11_trial}|{len(self._p11_steps)}')
            try:return original(*a,**kw)
            finally:torch.cuda.nvtx.range_pop()
        self.model_runner._model_forward=forward
        return result

    def execute_model(self,scheduler_output,*args,**kwargs):
        if self._p11_trial is None:return super().execute_model(scheduler_output,*args,**kwargs)
        index=len(self._p11_steps)+1
        entry=dict(step=index,scheduled=dict(scheduler_output.num_scheduled_tokens),start_ns=time.perf_counter_ns())
        self._p11_steps.append(entry)
        torch.cuda.nvtx.range_push(f'P11_STEP|{self._p11_trial}|{index}')
        try:return super().execute_model(scheduler_output,*args,**kwargs)
        finally:
            torch.cuda.nvtx.range_pop();entry['end_ns']=time.perf_counter_ns()

    def phase11_begin_capture(self,trial,enabled):
        assert self._p11_trial is None
        self._p11_trial=trial;self._p11_steps=[];self._p11_enabled=enabled=='true'
        torch.cuda.synchronize()
        if self._p11_enabled:torch.cuda.cudart().cudaProfilerStart()
        return dict(trial=trial,capture=self._p11_enabled)

    def phase11_end_capture(self,trial):
        assert trial==self._p11_trial
        torch.cuda.synchronize()
        if self._p11_enabled:torch.cuda.cudart().cudaProfilerStop()
        value=dict(trial=trial,enabled=self._p11_enabled,steps=self._p11_steps)
        p=Path('/evidence/worker')/(trial+'.json');p.parent.mkdir(exist_ok=True)
        with p.open('x') as f:json.dump(value,f)
        self._p11_trial=None;self._p11_steps=[]
        return dict(trial=trial,steps=len(value['steps']),file=str(p))

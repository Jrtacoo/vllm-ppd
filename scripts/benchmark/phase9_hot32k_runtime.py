"""Experiment-only launcher/observer; keeps the compiled vLLM tree intact."""
import argparse
import json
import os
from pathlib import Path
import threading
import time


def scheduler_class():
    from vllm.v1.core.sched.scheduler import Scheduler

    class HotTraceScheduler(Scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.hot_records = []
            threading.Thread(target=self.export, daemon=True).start()

        def export(self):
            directory = Path('/evidence')
            last = None
            while True:
                time.sleep(0.2)
                flag = directory / 'export-request'
                if not flag.exists():
                    continue
                token = flag.read_text()
                if token == last:
                    continue
                temp = directory / 'scheduler-trace.tmp'
                temp.write_text(json.dumps(self.hot_records.copy()))
                temp.replace(directory / 'scheduler-trace.json')
                (directory / 'export-ack').write_text(token)
                last = token

        def schedule(self):
            output = super().schedule()
            if output.num_scheduled_tokens:
                begun = time.perf_counter_ns()
                rows = []
                for rid, n in output.num_scheduled_tokens.items():
                    req = self.requests[rid]
                    rows.append(dict(id=rid, scheduled_tokens=n,
                        computed_before=req.num_computed_tokens-n,
                        prompt_tokens=req.num_prompt_tokens,
                        cached_tokens=req.num_cached_tokens,
                        preemptions=req.num_preemptions))
                self.hot_records.append(dict(wall_ns=time.time_ns(), requests=rows,
                    preempted=list(output.preempted_req_ids),
                    observer_us=(time.perf_counter_ns()-begun)/1000))
            return output
    return HotTraceScheduler


# Importing this module as a scheduler should not import/launch the server.
if __name__ != '__main__':
    HotTraceScheduler = scheduler_class()


def main():
    import sys
    sys.path.insert(0, '/opt/ppd')
    from ppd.dcn10 import load_config, build_launch
    from ppd.dcn9 import build_proxy_launch
    p = argparse.ArgumentParser()
    p.add_argument('--role', choices=['prefill', 'decode', 'proxy'], required=True)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    config = load_config(args.config)
    if args.role == 'proxy':
        command, env = build_proxy_launch(config, routing='1P_1D', http_port=10101)
    else:
        command, env = build_launch(config)
        if args.role == 'decode':
            command.remove('--enable-prefix-caching')
            command.append('--no-enable-prefix-caching')
        command += ['--generation-config', 'vllm', '--enable-prompt-tokens-details',
                    '--scheduler-cls', 'phase9_hot32k_runtime.HotTraceScheduler']
    print('HOT32K_LAUNCH ' + json.dumps(dict(command=command,
        environment={k:v for k,v in env.items() if k.startswith('NCCL_')
                     or k == 'SIMULATED_KV_BW_GBps'})), flush=True)
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()

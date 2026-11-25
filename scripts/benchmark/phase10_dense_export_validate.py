#!/usr/bin/env python3
"""Exercise export durability and the legacy empty-signal race without a GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
from unittest.mock import patch


def main():
    # Load the observer independently of the installed vLLM/GPU stack.
    name='vllm.v1.core.sched.scheduler'
    fake=types.ModuleType(name);fake.Scheduler=object
    with patch.dict(sys.modules,{name:fake}):
        spec=importlib.util.spec_from_file_location('export_observer',Path(__file__).with_name('phase10_dense_scheduler.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checks=[]
    with tempfile.TemporaryDirectory(prefix='phase10-export-check-') as td:
        root=Path(td);s=object.__new__(module.TraceScheduler)
        s._figure2_path=root/'scheduler-trace.jsonl'
        s._figure2_lock=threading.Lock();s._figure2_records=[{'step':1},{'step':2}]
        # Deterministic interleaving of the legacy write_text truncate/write:
        # empty token drains good records; real token overwrites with zero rows.
        records=list(s._figure2_records);latest=root/'legacy.jsonl'
        for observed_token in ('','123'):
            latest.write_text(''.join(json.dumps(r)+'\n' for r in records));records=[]
        assert latest.read_bytes()==b''
        checks.append('legacy empty-then-real signal deterministically overwrites a valid trace')
        assert s._figure2_export_token('') is None and len(s._figure2_records)==2
        assert s._figure2_export_token('invalid') is None
        r1=s._figure2_export_token('123');assert r1['records']==2 and not s._figure2_records
        s._figure2_records.append({'step':3})
        assert s._figure2_export_token('123')==r1 and len(s._figure2_records)==1
        r2=s._figure2_export_token('124');r3=s._figure2_export_token('125')
        assert r2['records']==1 and r3['records']==0
        for r in (r1,r2,r3):
            raw=(root/r['file']).read_bytes()
            assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
            assert len(raw.splitlines())==r['records']
        checks.append('empty/invalid ignored; duplicate idempotent; consecutive and empty exports preserve immutable snapshots')
        s._figure2_records.append({'step':4})
        with patch.object(module.os,'fsync',side_effect=OSError('synthetic disk failure')):
            try:s._figure2_export_token('126')
            except OSError:pass
            else:raise AssertionError('I/O failure hidden')
        assert s._figure2_records==[{'step':4}] and not (root/'exports/126.json').exists()
        assert s._figure2_export_token('126')['records']==1
        checks.append('failed disk commit retains records; retry commits the same evidence with a hash receipt')
    print(json.dumps(dict(passed=True,synthetic_only=True,checks=checks),indent=2))


if __name__=='__main__':main()

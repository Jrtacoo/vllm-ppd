#!/usr/bin/env python3
"""Hash-checked Phase11 evidence publication to the existing dcn9 project."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import phase11_profile_operate as o

ROOT=Path('/workspace/vllm-ppd')

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def sync():
    evidence=ROOT/'artifacts'/o.TAG
    selected=[ROOT/'docs/TWO_NODE_PHASE11.md',
              *sorted((ROOT/'scripts/benchmark').glob('phase11_*.py')),
              *sorted((ROOT/'results/figures/phase11').rglob('*')),
              *sorted(evidence.rglob('*'))]
    selected=sorted(set(p for p in selected if p.is_file() and '__pycache__' not in p.parts and p.name!='sync-receipt.json'))
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in selected}
    mp=evidence/'sync-manifest.json'
    # Manifest is generated after final analysis and cleanup, only once.
    assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n');selected.append(mp)
    archive=ROOT/'artifacts/phase11-resource-timeline-transfer.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tf:
        for p in selected:tf.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    code=f'''import hashlib,json,pathlib,tarfile,sys,os
root=pathlib.Path({str(o.base.HOST)!r})
verified=created=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tf:
 for m in tf:
  rel=pathlib.PurePosixPath(m.name)
  assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts
  assert str(rel).startswith(('artifacts/{o.TAG}/','results/figures/phase11/','scripts/benchmark/phase11_')) or str(rel)=='docs/TWO_NODE_PHASE11.md'
  p=root/str(rel);p.parent.mkdir(parents=True,exist_ok=True)
  src=tf.extractfile(m)
  if p.exists():
   h=hashlib.sha256();g=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
   for b in iter(lambda:src.read(1048576),b''):g.update(b)
   assert h.digest()==g.digest(),'Existing remote file differs; retained: '+str(rel)
   verified+=1
  else:
   tmp=p.with_name(p.name+'.p11-transfer-'+str(os.getpid()))
   with tmp.open('xb') as f:
    for b in iter(lambda:src.read(1048576),b''):f.write(b)
   tmp.replace(p);created+=1
manifest=json.loads((root/'artifacts/{o.TAG}/sync-manifest.json').read_text())
for rel,want in manifest.items():
 p=root/rel;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert p.stat().st_size==want['bytes'] and h.hexdigest()==want['sha256'],rel
print(json.dumps(dict(passed=True,created=created,existing_identical=verified,verified_files=len(manifest))))'''
    with archive.open('rb') as f:
        r=subprocess.run([o.base.PY,o.base.HOST_EXEC,'dcn9','python3 -c '+shlex.quote(code)],stdin=f,capture_output=True,text=True)
    receipt=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,archive_sha256=digest(archive),archive_bytes=archive.stat().st_size)
    (evidence/'sync-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if r.returncode:raise RuntimeError('Synchronization stopped without overwriting differences: '+r.stderr[-2000:])
    print(r.stdout,flush=True)

if __name__=='__main__':sync()

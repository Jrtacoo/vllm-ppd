#!/usr/bin/env python3
"""Regenerate both large-batch figures from audited evidence or snapshots."""
import argparse
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--from-snapshot',action='store_true')
p.add_argument('--output-dir',type=Path,default=HERE)
args=p.parse_args()
for run,group in [('formal','shared-history'),('formal-independent','independent-history')]:
 command=[sys.executable,str(HERE.parent/'generate.py'),
 '--evidence-dir',str(ROOT/'artifacts/two-node-phase8-figure2-large-batch-evidence'),
 '--run',run,'--output-dir',str(args.output_dir/group),
 '--snapshot',str(HERE/group/'plot-data.json'),'--linear-x',
 '--figure-label','PHASE 8 / FIGURE 2 LARGE-BATCH ADAPTATION']
 if args.from_snapshot:command.append('--from-snapshot')
 subprocess.run(command,check=True)

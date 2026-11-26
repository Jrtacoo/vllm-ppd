#!/usr/bin/env python3
"""Build portable Phase11 plot snapshots from audited Nsight captures."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import statistics as st
import numpy as np

ROOT=Path('/workspace/vllm-ppd')
EV=ROOT/'artifacts/phase11-resource-timeline-evidence'
OUT=ROOT/'results/figures/phase11'
BATCHES=(10,50,100,150,200,250)
SCENARIOS=('decode_only','append_prefill','full_prefill')

def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def build():
    OUT.mkdir(parents=True,exist_ok=False);snap=OUT/'snapshots';snap.mkdir()
    reports=[];selected=[];summary=[]
    for b in BATCHES:
        group=EV/f'formal-b{b}'
        assert read(group/'resource-verification.json')['passed']
        for f in sorted((group/'extracted').glob('*.json')):
            r=read(f);reports.append(r)
            steps=r['steps']
            event=[s for s in steps if s['prefill_tokens']]
            ordinary=[s for s in steps if not s['background_prefill'] and not s['prefill_tokens'] and s['background']==b
                      and all(32<=x<=80 for x in s['background_outputs'])]
            assert ordinary
            summary.append(dict(trial=r['trial'],batch=b,repeat=r['repeat'],scenario=r['scenario'],
                tpot_ms=r['tpot_ms'],ordinary_gpu_ms=st.mean(s['gpu_execute_ms'] for s in ordinary),
                ordinary_sm=st.mean(s['metrics']['sm'] for s in ordinary),ordinary_read=st.mean(s['metrics']['read'] for s in ordinary),
                event_chunks=len(event),event_gpu_ms=sum(s['gpu_execute_ms'] for s in event),
                event_body_ms=sum(s['gpu_body_ms'] for s in event),
                event_sm=float(np.average([s['metrics']['sm'] for s in event],weights=[s['gpu_execute_ms'] for s in event])) if event else None,
                event_read=float(np.average([s['metrics']['read'] for s in event],weights=[s['gpu_execute_ms'] for s in event])) if event else None))
            if r['repeat']==3:
                shutil.copyfile(f.with_suffix('.npz'),snap/f.with_suffix('.npz').name)
                selected.append(r)
    assert len(reports)==90 and len(selected)==18
    save(OUT/'plot-data.json',dict(version=11,selected_repeat=3,selection='Fixed repeat 3 for every condition; no outcome-based selection.',
        metrics=reports[0]['metric_definitions'],selected=selected,trials=summary,
        method=dict(raw_sampling_us=100,overview_bin_ms=5,zoom_raw_us=100,
          whole='First background GPU kernel to last model execute kernel; cold model loading and prefix warmup excluded.',
          event='Actual interference prefill GPU boundaries; baseline selected near Append event background output ordinal within the same repeat; request skew retained.',
          repeats='Five repetitions are kept separate; scalar summaries show mean and sample SD; no averaging misaligned peaks.',
          forward='GPU bounds linked via NVTX -> CUDA runtime correlationId -> kernel start/end; includes internal gaps.')))
    save(OUT/'verification.json',dict(passed=True,trials=90,requests=sum(read(EV/f'formal-b{b}/on/verification.json')['requests'] for b in BATCHES),
        samples=sum(r['sample_quality']['samples'] for r in reports),
        max_sample_interval_ns=max(r['sample_quality']['max_ns'] for r in reports),
        gaps_over_150us=sum(r['sample_quality']['gaps_over_150us'] for r in reports),
        preemptions=sum(len(read(EV/f'formal-b{b}/on/verification.json')['preemptions']) for b in BATCHES),
        measured_gpu_forwards=sum(len(r['steps']) for r in reports),selected_snapshot_trials=18,
        pilot=read(EV/'pilot-verification.json')))
    shutil.copyfile(ROOT/'scripts/benchmark/phase11_plot.py',OUT/'generate.py')
    (OUT/'requirements.txt').write_text('numpy>=1.24\nmatplotlib>=3.7\n')
    table='\n'.join(f'| {b} | [PNG](resource-timeline-b{b}.png) | [PNG](forward-zoom-b{b}.png) |' for b in BATCHES)
    (OUT/'README.md').write_text("""# Phase11：GPU 资源时间线

单张 RTX 4090、Qwen2.5-7B BF16；模型权重驻留显存。背景请求独立 256 历史 + 32 新增，输出 128；干扰 2048 历史 + 128 新增，输出 1；每次只插入一次干扰。六种 batch、三场景、各五次，共 90 个正式 trial。

[资源与时延汇总](resource-summary.png)：全部 5 次重复的均值 ± 样本标准差。普通 decode GPU step、整段背景 TPOT、干扰期间 SM 活跃度和显存读带宽分别展示。整段 TPOT 包含每个背景请求全部 127 个输出间隔，不能代替事件窗口延迟。

| 背景 batch | 整段资源时间线 | 前向放大图 |
|---|---|---|
"""+table+"""

时间线和放大图固定展示第 3 次重复，不按结果挑选，不将错位的峰值平均。每张图都提供同名 PNG、SVG 和 PDF。

- 整段图：三行分别是 Decode only、Append、Full；左侧 SM Active，右侧显存读取/写入带宽。横轴从各场景首个背景 GPU kernel 起算。100 μs 原始采样以 5 ms 时长加权分箱展示，背景历史预热和磁盘模型加载不计入。阴影标记实际干扰 prefill 的 GPU step。
- 放大图：保留 100 μs 原始采样，每行使用自己的横轴尺度。虚线界定实际 execute kernel 边界，顶部彩条是模型主体边界；Full 的两个 prefill 分块一并显示。普通 decode 取同次重复、接近 Append 事件背景输出位置的一步；Full 事件可能稍后被调度，未将两者视为完全同时发生。
- 纵轴都是 Nsight 定义的百分比。SM Active 不是 occupancy 或 FLOPS 利用率；DRAM read/write 是 GDDR6X 带宽，不是显存占用，也不是 RoCE。曲线无法直接区分权重流量与 KV 流量。

`plot-data.json` 保存 90 个 trial 的标量统计和所选 18 条时间线的完整前向元数据；`snapshots/` 保存这 18 条时间线的原始 100 μs 采样。全部 90 次原始曲线、Nsight 报告、SQLite、逐请求 request/SSE/response、调度 trace 与验收数据均在项目 artifacts/phase11-resource-timeline-evidence/，不放入 Git。

复画：在本目录运行 `python generate.py`（需要 NumPy、Matplotlib）。报告和先导开销说明见 [Phase11 文档](../../../docs/TWO_NODE_PHASE11.md)，采样完整性和计数见 [verification.json](verification.json)。Nsight trace 会带来测量开销，先导每条件仅一次；本图属于开启采样的观测值，未扣除假定固定开销。
""")
    (OUT/'README.md').write_text((OUT/'README.md').read_text()+'\nB=10 的 Append 第 1 次重复包含约 35.16 ms 的 GPU kernel 间隙：该步首尾跨度 60.43 ms，kernel 执行区间并集约 23.40 ms；第 3 次重复相应为 24.78 / 23.86 ms。具体间隙来源未隔离。该点保留在全部 5 次的均值和标准差中，因此小 batch 的大误差条不能直接归因于 Append 计算本身。\n')
    print('Snapshot ready',len(reports),flush=True)

if __name__=='__main__':build()

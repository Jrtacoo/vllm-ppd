#!/usr/bin/env python3
"""Aggregate audited dense Phase10 shards and make standalone figures.

python generate.py --from-snapshot reproduces published figures without raw data.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'decode_only':'#596779','append_prefill':'#168AAD','full_prefill':'#D66B32'}
LABELS={'decode_only':'Decode only','append_prefill':'Append prefill','full_prefill':'Full prefill'}
ANCHORS=(10,50,100,150,200,250)
FIGURES=('background-tpot','single-vs-multiple','first5-background-tpot',
         'decode-step-trajectories','workload-verification','interference-cost')


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def describe(values):
    return dict(mean=st.mean(values),sd=st.stdev(values) if len(values)>1 else None,n=len(values),
                min=min(values),max=max(values)) if values else None


def result_analysis(data):
    """Paired repeat statistics; per-event residual is not a pure GPU timer."""
    grouped=defaultdict(list)
    for t in data['trials']:grouped[t['batch'],t['regime'],t['scenario']].append(t)
    rows=[]
    for (b,regime,scenario),tt in sorted(grouped.items()):
        eventful=[t for t in tt if t['events']]
        rows.append(dict(batch=b,regime=regime,scenario=scenario,
            relative_overhead_pct=describe([100*t['excess_tpot_ms']/t['baseline_tpot_ms'] for t in tt]),
            per_event_span_excess_ms=describe([t['total_span_excess_ms']/t['events'] for t in eventful]),
            contributing_eventful_repeats=len(eventful)))
    anchors=[]
    for b in ANCHORS:
        gg={g['scenario']:g for g in data['groups'] if g['batch']==b and g['regime'] in ('baseline','scaled')}
        base=gg['decode_only']['tpot_ms']['mean'];append=gg['append_prefill']['tpot_ms']['mean'];full=gg['full_prefill']['tpot_ms']['mean']
        anchors.append(dict(batch=b,events_mean=gg['full_prefill']['events']['mean'],
            decode_only_ms=base,append_ms=append,full_ms=full,
            append_overhead_pct=100*(append-base)/base,full_overhead_pct=100*(full-base)/base,
            append_reduction_vs_full_pct=100*(full-append)/full,
            append_first5_delta_ms=gg['append_prefill']['first5']['delta_ms']['mean'],
            full_first5_delta_ms=gg['full_prefill']['first5']['delta_ms']['mean'],
            first5_eventful_repeats=gg['full_prefill']['five_step_repeats']))
    baseline=sorted([g for g in data['groups'] if g['scenario']=='decode_only'],key=lambda g:g['batch'])
    jumps=sorted([dict(from_batch=p['batch'],to_batch=q['batch'],delta_ms=q['tpot_ms']['mean']-p['tpot_ms']['mean'])
                  for p,q in zip(baseline,baseline[1:])],key=lambda x:x['delta_ms'],reverse=True)
    return dict(anchors=anchors,groups=rows,largest_baseline_jumps=jumps[:5],
        method='Percentages in groups: paired ratio per repeat, then mean and sample SD. Anchor percentages: ratio of five-repeat means. Per-event cost: 127*(treatment TPOT-baseline TPOT)/actual event count, per background request; eventful repeats only, negative residuals retained. This is not isolated prefill or GPU execution time.')


def write_tables(data,output):
    analysis=result_analysis(data);save(output/'result-analysis.json',analysis)
    with (output/'anchor-summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(analysis['anchors'][0]));writer.writeheader();writer.writerows(analysis['anchors'])
    with (output/'group-summary.csv').open('w',newline='') as f:
        records=[]
        for g,a in zip(data['groups'],analysis['groups']):
            assert (g['batch'],g['regime'],g['scenario'])==(a['batch'],a['regime'],a['scenario'])
            row={k:g[k] for k in ('batch','regime','scenario','five_step_repeats','five_step_events')}
            metrics={k:g[k] for k in ('tpot_ms','excess_tpot_ms','events','background_schedule_steps')}
            metrics.update({'first5_delta_ms':g['first5']['delta_ms'],
                            'relative_overhead_pct':a['relative_overhead_pct'],
                            'per_event_span_excess_ms':a['per_event_span_excess_ms']})
            for metric,stats in metrics.items():
                for field in ('mean','sd','n'):row[metric+'_'+field]=stats[field] if stats else None
            records.append(row)
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return analysis


def aggregate(e):
    completed=json.loads((e/'completed-groups.json').read_text())
    formal=[x for x in completed if x['kind']=='formal']
    assert len(formal)==1205 and sum(x['trials'] for x in formal)==3675,'Incomplete formal matrix'
    trials=[];events=[];trajectories=[];reports=[]
    for x in formal:
        s=json.loads((e/'summaries'/(x['id']+'.json')).read_text())
        assert s['verification']['passed']
        trials.extend(s['trials']);events.extend(s['events']);reports.append(s['verification'])
        trajectories.extend(s.get('trajectories',[]))
    assert len({t['trial'] for t in trials})==3675
    grouped=defaultdict(list)
    for t in trials:grouped[t['batch'],t['regime'],t['scenario']].append(t)
    groups=[]
    for (b,regime,scenario),rr in sorted(grouped.items()):
        assert len(rr)==5 and sorted(r['repeat'] for r in rr)==list(range(1,6))
        valid=[r for r in rr if r['events'] and r['event_window_unions']['5']['complete']]
        groups.append(dict(batch=b,regime=regime,scenario=scenario,
            **{k:describe([r[k] for r in rr]) for k in ('tpot_ms','baseline_tpot_ms','excess_tpot_ms','events','background_schedule_steps','background_decode_work')},
            first5={k:describe([r['event_window_unions']['5'][k] for r in valid]) for k in ('client_ms','baseline_client_ms','delta_ms')},
            five_step_repeats=len(valid),five_step_events=sum(r['events'] for r in valid)))
    plan=json.loads((e/'plan.json').read_text())
    for b in range(10,251):
        for sc in COLORS:
            key=(b,'baseline' if sc=='decode_only' else 'scaled',sc)
            assert key in grouped,'Missing primary condition'
        for r in range(1,6):
            aa=next(x for x in grouped[b,'scaled','append_prefill'] if x['repeat']==r)
            ff=next(x for x in grouped[b,'scaled','full_prefill'] if x['repeat']==r)
            assert aa['events']==ff['events']==len(plan['trigger_plan'][str(b)]['triggers_by_repeat'][str(r)])
    # Sum actual scheduled tokens, not merely a cache-hit counter.
    event_tokens=defaultdict(int)
    for x in events:
        computed=sum(c['tokens'] for c in x['chunks'])
        assert computed==(128 if x['scenario']=='append_prefill' else 2176)
        event_tokens[x['batch'],x['regime'],x['scenario']]+=computed
    for g in groups:g['actual_prefill_tokens_mean']=event_tokens[g['batch'],g['regime'],g['scenario']]/5
    verification=dict(passed=True,formal_groups=1205,formal_trials=3675,
        requests=sum(r['requests'] for r in reports),output_tokens=sum(r['output_tokens'] for r in reports),
        formal_interference_events=len(events),max_scheduled_tokens=max(r['max_scheduled_tokens'] for r in reports),
        max_scheduled_sequences=max(r['max_scheduled_sequences'] for r in reports),
        bundled_events=sum(r['bundled_events'] for r in reports),
        excluded_local_steps=sum(r['excluded_local_steps'] for r in reports),
        preemptions=sum(len(r['preemptions']) for r in reports),
        observer_mean_us=st.mean(r['observer_us']['mean'] for r in reports),
        observer_max_us=max(r['observer_us']['max'] for r in reports),
        note='Observer mean averages shard means, not a request-weighted CUDA timing.')
    save(e/'verification.json',verification)
    return dict(version=1,verification=verification,plan=plan,groups=groups,trials=trials,events=events,trajectories=trajectories)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#BCC4CF','axes.labelcolor':'#253247','text.color':'#253247',
        'xtick.color':'#596779','ytick.color':'#596779','axes.grid':True,'axes.axisbelow':True,
        'grid.color':'#E6EBF1','grid.linewidth':.7,'legend.frameon':False,
        'savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42,'lines.linewidth':1.7})


def export(fig,output,name,caption):
    fig.text(.015,.012,caption,fontsize=8,color='#596779',va='bottom')
    fig.tight_layout(rect=(0,.12,1,.94))
    for suffix in ('png','svg','pdf'):fig.savefig(output/(name+'.'+suffix),dpi=210,bbox_inches='tight')
    plt.close(fig)


def curve(ax,rr,metric,color,label,band=True,ls='-'):
    rr=sorted(rr,key=lambda r:r['batch']);x=np.array([r['batch'] for r in rr])
    y=np.array([r[metric]['mean'] for r in rr]);sd=np.array([r[metric]['sd'] if r[metric]['sd'] is not None else np.nan for r in rr])
    ax.plot(x,y,color=color,label=label,ls=ls,marker='o' if len(rr)<=6 else None,ms=4)
    if band:ax.fill_between(x,y-sd,y+sd,color=color,alpha=.13,lw=0)


def axis(ax,ylabel):
    ax.set_xlim(8,252);ax.set_xticks([10,50,100,150,200,250]);ax.set_xlabel('Background decode batch size');ax.set_ylabel(ylabel)


def generate(data,output):
    assert data['verification']['passed'];output.mkdir(parents=True,exist_ok=True);style()
    gg=data['groups'];foot='RTX 4090 · Qwen2.5-7B BF16 · budget 2,048 · 5 repeats · shading: sample SD, no smoothing'
    fig,ax=plt.subplots(figsize=(10,5.8))
    for sc in COLORS:
        rr=[g for g in gg if g['scenario']==sc and g['regime'] in ('baseline','scaled')]
        curve(ax,rr,'tpot_ms',COLORS[sc],LABELS[sc])
    axis(ax,'Background TPOT (ms/token)');ax.set_ylim(bottom=0);ax.legend()
    fig.suptitle('Background decoding under workload-normalized prefill interference',fontweight='bold')
    export(fig,output,'background-tpot',foot+'\nB=10–250, every integer; each point averages all 127 gaps, then requests and repeats. Zero-event repeats included.\nBackground: 256 cached + 32 new →128 output. Interference: 2,048 history +128 new →1 output; target count B/50.')

    fig,axs=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    for ax,sc in zip(axs,('append_prefill','full_prefill')):
        curve(ax,[g for g in gg if g['scenario']==sc and g['regime']=='scaled'],'excess_tpot_ms',COLORS[sc],'Work-normalized count')
        rr=[g for g in gg if g['scenario']==sc and g['regime']=='single']
        ax.errorbar([r['batch'] for r in rr],[r['excess_tpot_ms']['mean'] for r in rr],yerr=[r['excess_tpot_ms']['sd'] for r in rr],
            color=COLORS[sc],ls='--',marker='s',ms=4,capsize=3,label='Exactly one interference')
        ax.axhline(0,color='#9DA8B7',lw=.8);axis(ax,'Excess background TPOT (ms/token)');ax.set_title(LABELS[sc]);ax.legend()
    fig.suptitle('Paired excess latency: workload-normalized count versus one event',fontweight='bold')
    export(fig,output,'single-vs-multiple',foot+'\nExcess = treatment − same-repeat decode-only baseline. Single-event controls: six marked batches. Negative differences retained.')

    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,gridspec_kw={'height_ratios':[3,1.25]})
    for col,sc in enumerate(('append_prefill','full_prefill')):
        ax=axs[0,col];rr=[g for g in gg if g['scenario']==sc and g['regime']=='scaled' and g['first5']['client_ms']]
        for key,label,color in [('client_ms',LABELS[sc],COLORS[sc]),('baseline_client_ms','Matched decode-only',COLORS['decode_only'])]:
            transformed=[dict(batch=g['batch'],metric=g['first5'][key]) for g in rr]
            curve(ax,transformed,'metric',color,label)
        axis(ax,'Five-step mean token gap (ms)');ax.set_title(LABELS[sc]);ax.legend()
        ax=axs[1,col]
        ax.plot([g['batch'] for g in rr],[g['five_step_repeats'] for g in rr],color=COLORS[sc],label='Contributing repeats')
        ax.set_ylim(0,5.6);ax.set_yticks(range(6));axis(ax,'Eventful repeats');ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Local background latency: first five steps from actual prefill scheduling',fontweight='bold')
    export(fig,output,'first5-background-tpot',foot+'\nOnly eventful repeats with complete five-step unions contribute; SD unavailable at n=1. Event counts retained in plot-data.json.\nWindows include the first prefill step and next four; baseline matches exact request/output positions; overlapping steps counted once.')

    fig,axs=plt.subplots(2,3,figsize=(13,8),sharex=True,sharey=True)
    tt=[r for r in data['trajectories'] if r['regime'] in ('baseline','scaled')]
    for ax,b in zip(axs.flat,ANCHORS):
        for sc in COLORS:
            by_trial=defaultdict(list)
            for r in tt:
                if r['batch']==b and r['scenario']==sc:by_trial[r['trial']].append(r)
            for i,rr in enumerate(by_trial.values()):
                rr=sorted(rr,key=lambda r:r['step'])
                ax.plot([r['mean_output_ordinal'] for r in rr],[r['client_ms'] for r in rr],color=COLORS[sc],alpha=.45,lw=1,
                        label=LABELS[sc] if i==0 else None)
        ax.set_title(f'Batch {b} · target {b/50:g} event(s)/trial');ax.set_xlim(0,128)
        ax.set_xlabel('Background output ordinal (mean)');ax.set_ylabel('Background token gap (ms)')
    upper=max(r['client_ms'] for r in tt)*1.07
    for ax in axs.flat:ax.set_ylim(0,upper)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=3)
    fig.suptitle('Every measured step at six representative batch sizes',fontweight='bold',y=1.0)
    export(fig,output,'decode-step-trajectories','Each thin line is one repeat; all peaks retained, no smoothing. Triggers differ across phase-stratified repeats.\nOnly complete, unambiguous client intervals are shown; all original SSE and scheduler records remain in verified archives.')

    fig,axs=plt.subplots(1,3,figsize=(14,5.6))
    rr=sorted([g for g in gg if g['scenario']=='full_prefill' and g['regime']=='scaled'],key=lambda r:r['batch'])
    xs=[g['batch'] for g in rr]
    axs[0].plot(xs,[b/50 for b in xs],color='#596779',ls='--',label='Target B/50')
    axs[0].plot(xs,[g['events']['mean'] for g in rr],color=COLORS['full_prefill'],label='Actual five-repeat mean')
    axis(axs[0],'Interferences per trial');axs[0].legend();axs[0].set_title('Event allocation')
    for sc in ('append_prefill','full_prefill'):
        rr=sorted([g for g in gg if g['scenario']==sc and g['regime']=='scaled'],key=lambda r:r['batch'])
        axs[1].plot([g['batch'] for g in rr],[g['actual_prefill_tokens_mean']/(g['batch']*96) for g in rr],color=COLORS[sc],label=LABELS[sc])
    axis(axs[1],'Prefill tokens / active decode tokens');axs[1].legend();axs[1].set_title('Measured computation-token ratio')
    rr=[g for g in gg if g['scenario']=='decode_only']
    curve(axs[2],rr,'background_schedule_steps',COLORS['decode_only'],'Measured schedule steps')
    axs[2].axhline(127,ls='--',color='#9DA8B7',label='127 output intervals')
    axis(axs[2],'Background decode schedule steps');axs[2].legend();axs[2].set_title('Baseline scheduling calibration')
    fig.suptitle('Workload and scheduling checks across the complete sweep',fontweight='bold')
    export(fig,output,'workload-verification','Active output interval: 8–104; one event per 4,800 background token advances. Across five repeats, event counts round to 0.2 increments.\nToken ratio is a workload descriptor, not a FLOP equivalence. Schedule-step counts can exceed 127 when some batches are partial.')

    analysis=result_analysis(data)
    fig,axs=plt.subplots(1,3,figsize=(14.5,5.8),gridspec_kw={'width_ratios':[1.2,1,1]})
    for sc in ('append_prefill','full_prefill'):
        rr=[g for g in analysis['groups'] if g['scenario']==sc and g['regime']=='scaled']
        curve(axs[0],rr,'relative_overhead_pct',COLORS[sc],LABELS[sc])
    axis(axs[0],'Increase over paired baseline (%)');axs[0].set_title('All-output relative overhead');axs[0].legend(loc='upper left',fontsize=9)
    axs[0].axhline(0,color='#9DA8B7',lw=.8)
    for ax,sc in zip(axs[1:],('append_prefill','full_prefill')):
        for regime,label,marker,ls,alpha in [('scaled','Count scaled with batch','o','-',1),('single','Exactly one event','s','--',.65)]:
            rr=[g for g in analysis['groups'] if g['scenario']==sc and g['regime']==regime and g['batch'] in ANCHORS]
            ax.errorbar([g['batch'] for g in rr],[g['per_event_span_excess_ms']['mean'] for g in rr],
                yerr=[g['per_event_span_excess_ms']['sd'] if g['per_event_span_excess_ms']['sd'] is not None else np.nan for g in rr],
                color=COLORS[sc],alpha=alpha,ls=ls,marker=marker,ms=4,capsize=3,label=label)
            if regime=='scaled':
                low=rr[0];ax.annotate('n=1',xy=(low['batch'],low['per_event_span_excess_ms']['mean']),
                                    xytext=(8,9),textcoords='offset points',fontsize=8,color=COLORS[sc])
        axis(ax,'Excess decode span per event (ms/request)');ax.set_title(LABELS[sc]+' · per-event cost');ax.legend(fontsize=8,loc='upper left')
        if sc=='append_prefill':ax.axhline(0,color='#9DA8B7',lw=.8)
    fig.suptitle('Separating event frequency from the cost of each interference',fontweight='bold',fontsize=13)
    export(fig,output,'interference-cost','Left: mean paired percentage over all five repeats; shading = sample SD. Right: mean ± sample SD over eventful repeats.\nPer-event cost = 127 × (treatment TPOT − matched baseline TPOT) / actual events, per background request; not isolated prefill/GPU time.\nZero-event repeats excluded only on the right; n=1 has no SD. Whole-output noise and negative residuals are retained; right panels use different y scales.')


def write_report(data,e,output):
    v=data['verification'];gg=data['groups'];lines=[]
    for b in ANCHORS:
        values=[]
        for sc in COLORS:
            r=next(g for g in gg if g['batch']==b and g['scenario']==sc and g['regime'] in ('baseline','scaled'))
            values.append(f"{r['tpot_ms']['mean']:.2f} ± {r['tpot_ms']['sd']:.2f}")
        n=next(g['events']['mean'] for g in gg if g['batch']==b and g['scenario']=='full_prefill' and g['regime']=='scaled')
        lines.append(f"| {b} | {n:g} | "+' | '.join(values)+' |')
    table='\n'.join(lines)
    text=f'''# Phase10：密集 batch 与按背景工作量安排干扰

本轮为单 RTX 4090、Qwen2.5-7B BF16 的适配实验，保留此前 Phase10 结果。正式覆盖 batch=10～250 的全部 241 个整数，每条件五次重复；三种主场景共 3,615 组，六个代表 batch 的固定单次对照共 60 组。先导不混入正式结果。

正式审计通过：{v['formal_trials']:,} 组、{v['requests']:,} 条请求（含历史预热），输出 {v['output_tokens']:,} tokens；实际干扰事件 {v['formal_interference_events']:,} 个。最大调度 {v['max_scheduled_tokens']} tokens、{v['max_scheduled_sequences']} 条序列，无抢占。缓存、计算量、输入配对与输出长度均逐请求核验。

背景：独立256历史＋32新增，固定输出128。干扰：独立2048历史＋128新增，固定输出1；Append复用历史、计算128，Full冷盐重算2176。每trial清缓存，各条件均预热B条背景和ceil(B/50)条独立干扰历史。预算2048、最大序列251、FCFS连续批处理、chunked prefill、eager、显存比例0.94。

在输出8～104之间，每4,800个背景decode token推进量安排一次干扰，目标平均次数B/50。用五个均匀相位分配到五次重复，相位按固定种子和batch置换，与运行延迟无关；同batch/repeat的Append与Full使用同一输入及触发点。累计次数取最接近B/10的整数，恰好半个时向上；五次平均的误差最多0.1次。batch10的五次重复合计一次，不强制每次至少一次。触发依据所有背景已达到的输出进度，实际首次调度及排队另行记录。

全输出TPOT先按每条背景(末token−首token)/127计算，再对请求与五次重复平均；包含零事件重复。下表均值±样本标准差，单位ms/token。

| Batch | 平均干扰次数 | Decode only | Append | Full |
|---:|---:|---:|---:|---:|
{table}

五步窗口以实际首个prefill调度为起点，包含该步和后续四步；按请求编号和输出序号匹配同次基线，重叠步骤去重。只汇总有事件且窗口完整的重复，每个batch实际样本数与事件数保留；n=1时没有可估计的样本标准差。局部排除不完整或SSE合并隐藏的间隔，共{v['excluded_local_steps']:,}步；SSE合并事件{v['bundled_events']:,}个；全输出端点统计保留。异常峰值和负配对差值不删除，不做平滑或趋势筛选。

六组图：全输出TPOT、单次对照超额TPOT、五步窗口及贡献重复数、六个代表batch逐步轨迹、干扰负载与基线调度核验，以及相对开销与单次干扰成本。图和可独立重画数据在results/figures/phase10/dense-workload/。

增加次数改变负载；本实验是按背景工作量归一的干扰，不是固定每秒到达率。更多batch点不能单独证明固定一次干扰的硬件扩展机制，也不能声称完整复现H100论文Figure2。token比不是FLOP比；GPU采样包含模型与KV预分配，不能解释为每步有效KV占用。客户端SSE间隔包含主机调度和网络等开销，并非CUDA kernel计时。输入及输出长度配对，不强制生成正文逐token一致。

证据在artifacts/{e.name}/：每个batch/repeat的全部原始请求、SSE、事件、响应、指标、调度trace与审计保存为archives/*.tar.gz；archive-manifests/提供逐文件SHA256，压缩包重新读取逐文件验证后才删除暂存副本，并同步dcn9校验。summaries/与completed-groups.json保留逐组分析；失败停机并保留暂存，不静默重试。导出仅在组间无测量请求时进行，导出成功后清除已保存的观察记录，限制内存占用。
'''
    resumes=sorted(e.glob('resume*/preflight-verification.json'))
    if resumes:
        text+='\n本轮曾因调度记录导出为空而中断；对应请求成功但缺少调度证据，原失败样本保存在scratch/，不混入正式结果。获准恢复后保留既有审计通过结果，仅重跑失败配对组并继续剩余扫描。恢复先导不计入正式结果；恢复版只修改证据导出及控制流程，输入、调度参数、统计和审计保持一致。各次运行源码与恢复核验保存在resume*/，图的source-provenance.json记录全部源码版本。\n'
    a=result_analysis(data);by_batch={r['batch']:r for r in a['anchors']}
    lo=by_batch[10];hi=by_batch[250]
    single={g['batch']:g for g in data['groups'] if g['scenario']=='full_prefill' and g['regime']=='single'}
    text+=f'''

## 结果解读

![全输出背景TPOT](background-tpot.png)

Full干扰明显重于Append。batch=250时，相对无干扰的全输出TPOT增加：Append {hi['append_overhead_pct']:.2f}%，Full {hi['full_overhead_pct']:.2f}%；Append比Full的背景TPOT低{hi['append_reduction_vs_full_pct']:.2f}%。这些是背景decode的延迟，未把干扰请求自己的TTFT作为纵轴。

低batch的全输出曲线接近，不代表插入prefill时完全没有停顿。batch=10的五次重复合计只有一次干扰，其余四次为零事件；一次短暂停顿又均摊到127个输出间隔。所以Full全输出TPOT仅增加{lo['full_overhead_pct']:.2f}%，但真正有事件那次的五步窗口仍比匹配基线高{lo['full_first5_delta_ms']:.2f} ms/token。此点窗口样本n=1，不能估计重复间标准差。batch=250的五步窗口超额为Append {hi['append_first5_delta_ms']:.2f}、Full {hi['full_first5_delta_ms']:.2f} ms/token，两者都说明干扰主要表现为局部峰值。

曲线差距扩大包含明确的负载次数因素。Full固定单次对照中，batch=10→250的全输出超额TPOT仅从{single[10]['excess_tpot_ms']['mean']:.2f}增加至{single[250]['excess_tpot_ms']['mean']:.2f} ms/token；按batch增加次数时却从{lo['full_ms']-lo['decode_only_ms']:.2f}增加至{hi['full_ms']-hi['decode_only_ms']:.2f} ms/token。以batch=50→250为例，干扰次数由1变5，全输出Full超额约增至原来的{(hi['full_ms']-hi['decode_only_ms'])/(by_batch[50]['full_ms']-by_batch[50]['decode_only_ms']):.2f}倍；固定单次对照只增至{single[250]['excess_tpot_ms']['mean']/single[50]['excess_tpot_ms']['mean']:.2f}倍。因此，本轮能证明当前负载规则下Append降低干扰，也有单次成本随batch上升的迹象；不能把完整斜率都归因为GPU在大batch下的资源争用，更不能视作原论文固定实验条件的直接复现。

![相对开销与每次干扰成本](interference-cost.png)

所有已记录Append事件均一次计算128个新增token；Full都分为两块计算总计2176个token。首块需与背景decode共享2048预算，次块完成剩余部分，所以Full会造成较大的局部decode间隔。图中基线本身在batch=129和193附近也发生台阶：即使没有prefill仍然存在，且没有KV抢占。仅凭现有时延和低频GPU采样不能确定具体kernel/执行路径原因，报告保留台阶，不归因于显存不足。

## 图和数据索引

| 文件 | 含义 |
|---|---|
| [background-tpot.png](background-tpot.png) | 主图：全部输出间隔的背景TPOT，五次均值±样本标准差，保留零事件重复 |
| [single-vs-multiple.png](single-vs-multiple.png) | 扣除配对基线后，比较固定单次与随batch增加次数的开销 |
| [first5-background-tpot.png](first5-background-tpot.png) | 每次干扰实际被调度起的五步窗口；重叠窗口去重，下排报告参与重复数 |
| [decode-step-trajectories.png](decode-step-trajectories.png) | 六个代表batch各次重复的逐步间隔；峰值不平滑、不删除 |
| [workload-verification.png](workload-verification.png) | 干扰次数、计算token比例和基线调度步数 |
| [interference-cost.png](interference-cost.png) | 相对TPOT开销，以及每个事件对应的背景请求超额输出跨度；后者保留噪声，不是纯GPU耗时 |

每张图均提供同名SVG/PDF。`anchor-summary.csv`提供六个代表点，`group-summary.csv`提供全部735种条件的均值、标准差和样本数；`result-analysis.json`提供配对百分比及按事件归一的统计。图中阴影为样本标准差，不是置信区间。小幅负超额保留，不能据此声称prefill提高性能。

日志中的观察器平均开销约{v['observer_mean_us']:.1f}微秒（按组均值平均），最大单步观察器时间约{v['observer_max_us']/1000:.1f}毫秒。偶发主机调度/观察器长尾未剔除；因此单个局部异常不能直接解释为GPU kernel变慢。五步窗口排除的是不完整的全batch步等无法严格配对的间隔，而非按延迟大小删除样本；全输出端点统计保留全部正式请求。
'''
    (output/'README.md').write_text(text+'\n重画：`python -m pip install -r requirements.txt`，然后 `python generate.py --from-snapshot`。\n')
    docs=e.parents[1]/'docs/TWO_NODE_PHASE10_DENSE.md'
    for name in FIGURES:text=text.replace(']('+name+'.png)','](../results/figures/phase10/dense-workload/'+name+'.png)')
    docs.write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',type=Path);p.add_argument('--from-snapshot',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.from_snapshot:
        output=args.output or Path(__file__).resolve().parent
        data=json.loads((Path(__file__).resolve().parent/'plot-data.json').read_text())
        generate(data,output);write_tables(data,output);return
    if args.evidence is None:p.error('--evidence is required unless --from-snapshot')
    e=args.evidence.resolve();output=args.output or e.parents[1]/'results/figures/phase10/dense-workload'
    output.mkdir(parents=True,exist_ok=True);data=aggregate(e);save(output/'plot-data.json',data)
    generate(data,output);write_tables(data,output);write_report(data,e,output)
    shutil.copy2(__file__,output/'generate.py')
    (output/'requirements.txt').write_text('matplotlib==3.10.8\nnumpy==2.2.6\n')
    attempts={p.parent.name:json.loads(p.read_text()) for p in sorted(e.glob('resume*/runner-start.json'))}
    save(output/'source-provenance.json',dict(frozen_sources=json.loads((e/'runner-start.json').read_text())['source_sha256'],
        resume_attempts=attempts,
        analysis_generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256((output/'plot-data.json').read_bytes()).hexdigest()))
    # Reproduction check uses ONLY published snapshot + generator, no raw evidence.
    import tempfile,subprocess,sys
    with tempfile.TemporaryDirectory(prefix='phase10-dense-replot-') as temporary:
        q=Path(temporary)
        for n in ('generate.py','plot-data.json'):shutil.copy2(output/n,q/n)
        subprocess.run([sys.executable,str(q/'generate.py'),'--from-snapshot'],check=True)
        assert {p.stem for p in q.glob('*.png')}==set(FIGURES)
        for n in ('result-analysis.json','anchor-summary.csv','group-summary.csv'):
            assert (q/n).read_bytes()==(output/n).read_bytes(),'Standalone statistic mismatch: '+n
    save(e/'plot-verification.json',dict(passed=True,standalone_replot=True,standalone_tables_identical=True,figures=len(FIGURES)))
    print(json.dumps(data['verification'],indent=2))


if __name__=='__main__':main()

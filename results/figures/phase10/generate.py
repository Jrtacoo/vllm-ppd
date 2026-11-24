#!/usr/bin/env python3
"""Portable publication figures for Phase 10 (matplotlib, numpy)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={'decode_only':'#596779','append_prefill':'#168AAD','full_prefill':'#D66B32'}
LABELS={'decode_only':'Decode only','append_prefill':'Append prefill','full_prefill':'Full prefill'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#BCC4CF','axes.labelcolor':'#253247','text.color':'#253247',
        'xtick.color':'#596779','ytick.color':'#596779','axes.grid':True,'axes.axisbelow':True,
        'grid.color':'#E6EBF1','grid.linewidth':.7,'legend.frameon':False,'savefig.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42,'lines.linewidth':2})


def export(fig,output,name,caption):
    fig.text(.015,.015,caption,fontsize=8,color='#596779',va='bottom')
    fig.tight_layout(rect=(0,.09,1,.94))
    if name=='background-tpot':
        fig.subplots_adjust(top=.80,bottom=.20,left=.10,right=.98)
        fig._suptitle.set_y(.98)
    for suffix in ('png','svg','pdf'):fig.savefig(output/(name+'.'+suffix),dpi=210,bbox_inches='tight')
    plt.close(fig)


def axes_common(ax,batches,ylabel):
    ax.set_xticks(batches);ax.set_xlabel('Background decode batch size');ax.set_ylabel(ylabel)
    ax.set_xlim(3,257)


def line(ax,rr,metric,scenario,label=None,ls='-',offset=0):
    rr=sorted(rr,key=lambda r:r['batch']);xs=np.array([r['batch'] for r in rr])+offset
    ys=[r[metric]['mean'] for r in rr];sd=[r[metric].get('sd',0) for r in rr]
    ax.errorbar(xs,ys,yerr=sd,color=COLORS[scenario],marker='s' if ls=='--' else 'o',ms=5,capsize=3,
                label=label or LABELS[scenario],linestyle=ls)


def generate(data,trajectory,output):
    check=data['verification']['passed']
    if not check:raise ValueError('Refuse to plot failed experiment')
    output.mkdir(parents=True,exist_ok=True);style()
    groups=data['groups'];trials=data['trials'];batches=sorted({r['batch'] for r in groups})
    n=data['run']['schedule'];repeats=len({r['repeat'] for r in n})
    foot=f'RTX 4090 · Qwen2.5-7B BF16 · budget 2,048 · {repeats} repeats · error bars: sample SD across repeats'
    fig,ax=plt.subplots(figsize=(9,5.3))
    for sc in COLORS:
        rr=[r for r in groups if r['scenario']==sc and r['regime'] in ('baseline','scaled')]
        line(ax,rr,'tpot_ms',sc)
        for r in rr:
            ys=[t['tpot_ms'] for t in trials if (t['batch'],t['regime'],t['scenario'])==(r['batch'],r['regime'],sc)]
            ax.scatter(r['batch']+np.linspace(-1.5,1.5,len(ys)),ys,s=12,color=COLORS[sc],alpha=.45,zorder=2)
    axes_common(ax,batches,'Background TPOT (ms/token)')
    ax.legend(loc='best');ax.set_ylim(bottom=0)
    top=ax.secondary_xaxis('top');top.set_xticks(batches)
    top.set_xticklabels([str(len(data['run']['trigger_table'][str(b)])) for b in batches]);top.set_xlabel('Number of interference requests (Append / Full)',labelpad=8)
    fig.suptitle('Repeated prefill interference during background decoding',fontweight='bold',y=1.02)
    export(fig,output,'background-tpot',foot+'\nBackground: 256 cached + 32 new, 128 output; TPOT averages all 127 gaps. Interference: 2,048 history + 128 new, 1 output.')

    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,sc in zip(axs,('append_prefill','full_prefill')):
        for regime,ls,label in [('single','--','One interference'),('scaled','-','1–5 interferences')]:
            rr=[r for r in groups if r['scenario']==sc and r['regime']==regime]
            line(ax,rr,'excess_tpot_ms',sc,label,ls)
        ax.axhline(0,color='#9DA8B7',lw=.8);axes_common(ax,batches,'Excess background TPOT (ms/token)')
        ax.set_title(LABELS[sc]);ax.legend()
    fig.suptitle('One interference versus increasing interference count',fontweight='bold')
    export(fig,output,'single-vs-multiple',foot+'\nSingle-event controls exist at batch ≥100. Each excess uses its same-repeat decode-only baseline; negative differences are retained.')

    fig,axs=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    for ax,sc in zip(axs,('append_prefill','full_prefill')):
        rr=[r for r in groups if r['scenario']==sc and r['regime']=='scaled' and r['first5']]
        for source,label,color,marker in [('client_ms',LABELS[sc],COLORS[sc],'o'),('baseline_client_ms','Matched decode-only','#596779','s')]:
            ax.errorbar([r['batch'] for r in rr],[r['first5'][source]['mean'] for r in rr],
                yerr=[r['first5'][source]['sd'] for r in rr],label=label,color=color,marker=marker,capsize=3)
        axes_common(ax,batches,'Five-step mean token gap (ms)');ax.set_title(LABELS[sc]);ax.legend()
    fig.suptitle('First five decode steps from each actual interference schedule',fontweight='bold')
    export(fig,output,'first5-background-tpot',foot+'\nWithin each trial: union of five-step windows, anchored at first prefill chunk; exact sequence/output ordinal matched baseline.\nOnly complete, unambiguous SSE windows contribute; available sample counts are recorded in plot-data.json.')

    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,b in zip(axs.flat,batches):
        for sc in COLORS:
            rr=[r for r in trajectory if r['batch']==b and r['scenario']==sc and r['regime'] in ('baseline','scaled')]
            by_trial=defaultdict(list)
            for r in rr:by_trial[r['trial']].append(r)
            for i,tt in enumerate(by_trial.values()):
                tt=sorted(tt,key=lambda r:r['step'])
                ax.plot([r['mean_output_ordinal'] for r in tt],[r['client_ms'] for r in tt],
                        color=COLORS[sc],alpha=.48,lw=1,label=LABELS[sc] if i==0 else None)
        for x in data['run']['trigger_table'][str(b)]:ax.axvline(x,color='#9DA8B7',lw=.8,ls=':',zorder=0)
        ax.set_title(f'Batch {b} · {len(data["run"]["trigger_table"][str(b)])} interference(s)')
        ax.set_xlabel('Background output ordinal (mean)');ax.set_ylabel('Background token gap (ms)')
        ax.set_xlim(0,128)
    upper=max(r['client_ms'] for r in trajectory if r['regime'] in ('baseline','scaled'))*1.07
    for ax in axs.flat:ax.set_ylim(0,upper)
    handles,labels=axs.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3)
    fig.suptitle('Background decode latency across the output sequence',fontweight='bold',y=1.01)
    export(fig,output,'decode-step-trajectories',f'RTX 4090 · Qwen2.5-7B BF16 · budget 2,048 · {repeats} repeats; each thin line is one repeat; all peaks shown.\nDotted lines mark client triggers; actual scheduling can occur later. X = mean output ordinal before each step.\nClient intervals touching bundled SSE or partial batches are excluded; model-step timings remain available in raw evidence.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(__file__).parent/'plot-data.json')
    p.add_argument('--trajectories',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args();data=json.loads(args.input.read_text())
    trajectory=json.loads((args.trajectories or args.input.parent/'step-trajectories.json').read_text())
    generate(data,trajectory,args.output or args.input.parent)


if __name__=='__main__':main()

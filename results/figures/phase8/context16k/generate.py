#!/usr/bin/env python3
"""Plot event-aligned Figure 2 diagnostics from a portable analysis snapshot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COLORS = {'full_prefill':'#D46A37', 'append_prefill':'#008577'}
LABELS = {'full_prefill':'Full prefill', 'append_prefill':'Append prefill'}
SCENARIOS = tuple(COLORS)
GRAY = '#697D93'


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#172D46',
        'axes.labelcolor':'#41556C','xtick.color':'#41556C','ytick.color':'#41556C',
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#C7D3E0',
        'svg.fonttype':'none','svg.hashsalt':'ppd-figure2-window','pdf.fonttype':42,
        'figure.facecolor':'white','savefig.facecolor':'white'})


def layout(rows, cols, title, subtitle, baseline=False):
    fig, axes = plt.subplots(rows, cols, figsize=(15.5, 3.45*rows+3.3), squeeze=False)
    fig.subplots_adjust(left=.075,right=.97,top=1-1.8/fig.get_figheight(),
                        bottom=1.35/fig.get_figheight(),hspace=.45,wspace=.27)
    h=fig.get_figheight()
    fig.text(.055,1-.22/h,'PHASE 8 / SAME-CONTEXT FULL VS APPEND / 16K HISTORY',color='#008577',weight='bold',fontsize=10,va='top')
    fig.text(.055,1-.61/h,title,fontsize=22,weight='bold',va='top')
    fig.text(.055,1-1.04/h,subtitle,fontsize=10,color='#607286',va='top')
    handles=[Line2D([0],[0],color=c,lw=2.2,label=LABELS[s]) for s,c in COLORS.items()]
    if baseline: handles.append(Line2D([0],[0],color=GRAY,lw=1.5,ls='--',label='Matched decode only'))
    fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.975,1-.17/h),
               ncol=len(handles),frameon=False,fontsize=9)
    for ax in axes.flat:
        ax.grid(color='#E8EDF3',lw=.8);ax.set_axisbelow(True);ax.tick_params(length=0,pad=7)
    return fig,axes


def finish(fig, out, name, notes):
    for i,note in enumerate(notes):
        fig.text(.055,(.60-i*.19)/fig.get_figheight(),note,fontsize=8.5,color='#607286')
    for ext in ('png','svg','pdf'):
        fig.savefig(out/(name+'.'+ext),dpi=200,metadata={'Creator':'PPD event-window diagnostic'})
    plt.close(fig)


def band(ax,x,mean,sd,color,ls='-',alpha=.13):
    x=np.array(x);mean=np.array(mean);sd=np.array(sd)
    ax.plot(x,mean,color=color,lw=2,ls=ls)
    ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=alpha,lw=0)


def timeline(groups, out, focus, subtitle):
    lookup={(g['batch_size'],g['scenario']):g for g in groups}
    fig,axes=layout(len(focus),2,'The stall appears at injection',subtitle,True)
    for row,b in enumerate(focus):
        for col,(metric,title) in enumerate((('cycle_ms','Engine iteration'),('client_ms','Client token interval'))):
            ax=axes[row,col]
            for scenario,color in COLORS.items():
                ps=[p for p in lookup[b,scenario]['profile'] if -5<=p['offset']<=20]
                assert all(p['n']==5 for p in ps)
                x=[p['offset'] for p in ps]
                band(ax,x,[p[metric]['mean'] for p in ps],[p[metric]['std'] for p in ps],color)
                ax.plot(x,[p[metric+'_baseline']['mean'] for p in ps],color=GRAY,lw=1.3,ls='--',alpha=.8)
            ax.axvline(0,color='#A3B1C0',ls=':',lw=1)
            ax.set_title(f'Batch {b}  |  {title}',loc='left',weight='bold',pad=12)
            ax.set_xlabel('Iteration relative to mixed prefill step');ax.set_ylabel('Duration (ms)')
            ax.set_xlim(-5,20);ax.set_xticks([-5,0,5,10,15,20]);ax.set_ylim(bottom=0)
    finish(fig,out,'event-timeline',[
        'Offset 0 = actual mixed scheduler step. Matched controls use the same repeat, request index and output-token ordinal. Shading = sample SD, n=5.',
        'Engine iteration includes scheduling/host overhead. Client intervals are singleton token-ID SSE arrival gaps; neither is pure GPU kernel time.',
        'Independent 128-token histories. Full recomputes 17,408 tokens; append reuses 16,384 and computes 1,024. All spikes are retained.'])


def dilution(groups,out,subtitle):
    batches=sorted({g['batch_size'] for g in groups});lookup={(g['batch_size'],g['scenario']):g for g in groups}
    fig,axes=layout(1,3,'A longer averaging window dilutes the stall',subtitle)
    for ax,(window,title) in zip(axes[0],[('1','Mixed-step interval'),('5','First 5 intervals'),('whole_request','Whole request: 127 intervals')]):
        for si,(scenario,color) in enumerate(COLORS.items()):
            ws=[next(w for w in lookup[b,scenario]['windows'] if w['metric']=='client_ms' and w['window']==window) for b in batches]
            means=[w['change_percent']['mean'] for w in ws];sd=[w['change_percent']['std'] for w in ws]
            if len(batches)==1:
                ax.bar(si,means[0],color=color,width=.55,alpha=.88)
                ax.errorbar(si,means[0],yerr=sd[0],color='#41556C',capsize=5,lw=1.3)
                ax.text(si,means[0]+sd[0]+max(means[0]*.04,1),f'{means[0]:.1f}%',ha='center',color=color,weight='bold')
            else:
                ax.errorbar(batches,means,yerr=sd,color=color,lw=2,marker='o' if si==0 else 's',markersize=4,capsize=3)
        ax.axhline(0,color=GRAY,ls='--',lw=1)
        ax.set_title(title,loc='left',weight='bold',pad=12);ax.set_ylabel('Client interval change (%)')
        if len(batches)==1:
            ax.set_xticks([0,1],['Full','Append']);ax.set_xlabel(f'Independent histories | batch {batches[0]}')
            low,high=ax.get_ylim();ax.set_ylim(min(0,low),high*1.13)
        else:
            ax.set_xlabel('Observed decode batch size');ax.set_xticks([0,50,100,150,200]);ax.set_xlim(0,205)
    finish(fig,out,'window-dilution',[
        'All panels use client token intervals and paired decode-only controls. Percent changes are computed per trial, then averaged; error bars = sample SD, n=5.',
        'Panels have different vertical scales. The first two use event-aligned windows; the last reproduces the existing whole-request TPOT metric.',
        'Batch 0 has no decode TPOT and is shown only in the prefill-cost figure. Full and append receive identical 17,408-token prompts.'])


def cumulative(groups,out,focus,subtitle):
    lookup={(g['batch_size'],g['scenario']):g for g in groups}
    fig,axes=layout(len(focus),2,'How much delay remains after injection?',subtitle)
    for row,b in enumerate(focus):
        end=min(max(p['offset'] for p in lookup[b,s]['profile'] if p['n']==5) for s in SCENARIOS)
        for col,(metric,title) in enumerate((('cycle_ms','Engine iteration excess'),('client_ms','Client interval excess'))):
            ax=axes[row,col]
            for scenario,color in COLORS.items():
                ps=[p for p in lookup[b,scenario]['profile'] if 0<=p['offset']<=end]
                assert all(p['n']==5 for p in ps)
                key=metric+'_cumulative_delta'
                band(ax,[p['offset'] for p in ps],[p[key]['mean'] for p in ps],[p[key]['std'] for p in ps],color)
            ax.axhline(0,color=GRAY,lw=1,ls='--');ax.set_xlim(0,end)
            ax.set_title(f'Batch {b}  |  {title}',loc='left',weight='bold',pad=12)
            ax.set_xlabel('Last iteration included, relative to injection');ax.set_ylabel('Cumulative paired excess (ms)')
    finish(fig,out,'cumulative-delay',[
        'Signed differences are summed from step 0: slower steps add delay and faster steps subtract it. Shading = SD of trial cumulative sums, n=5.',
        'Curves end at the last common full-batch, matched step across both scenarios and all five repeats. Partial-batch tails remain in the derived audit CSV.',
        'A plateau indicates little net additional delay; it does not prove every subsequent step is unaffected. Later spikes and host/observer stalls are retained.'])


def prefill_cost(data,groups,out,subtitle):
    zero={g['scenario']:g for g in data['prefill_only_groups']['formal']}
    lookup={(g['batch_size'],g['scenario']):g for g in groups}
    batches=sorted({g['batch_size'] for g in groups})
    fig,axes=layout(1,2,'Full recomputation versus cached append',subtitle,True)
    ax=axes[0,0]
    for scenario,color in COLORS.items():
        ps=[next(p for p in lookup[b,scenario]['profile'] if p['offset']==0) for b in batches]
        points=[zero[scenario]['model_wall_ms']]+[p['model_ms'] for p in ps]
        ax.errorbar([0]+batches,[p['mean'] for p in points],yerr=[p['std'] for p in points],
                    color=color,marker='o' if scenario=='full_prefill' else 's',ms=4,lw=2,capsize=3)
        ax.plot(batches,[p['model_ms_baseline']['mean'] for p in ps],color=GRAY,ls='--',lw=1.3)
    ax.set_title('A   Prefill / mixed-step model wall time',loc='left',weight='bold',pad=12)
    ax.set_xlabel('Observed decode batch size');ax.set_ylabel('Model wall time (ms)')
    ax.set_xticks([0,50,100,150,200]);ax.set_xlim(-3,205);ax.set_ylim(bottom=0)
    ax=axes[0,1]
    for i,(scenario,color) in enumerate(COLORS.items()):
        x=zero[scenario]['ttft_ms'];ax.bar(i,x['mean'],width=.55,color=color,alpha=.9)
        ax.errorbar(i,x['mean'],yerr=x['std'],capsize=5,color=GRAY,lw=1.4)
        ax.text(i,x['mean']+x['std']+50,f"{x['mean']:.1f} ms",ha='center',color=color,weight='bold')
    ax.set_title('B   Batch 0: isolated prefill TTFT',loc='left',weight='bold',pad=12)
    ax.set_xticks([0,1],['Full prefill','Append prefill']);ax.set_ylabel('Client TTFT (ms)')
    ax.set_xlabel('No background decode; no decode TPOT');ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    finish(fig,out,'prefill-cost',[
        'Full and append use exactly the same 16,384 history + 1,024 new tokens. Cold full uses a separate cache salt; append uses the warmed salt.',
        'At batch 0, panel A measures a standalone prefill step. At positive batches it measures prefill mixed with B decode sequences; controls have no prefill.',
        'Means and sample SD across five repeats. Model wall time includes engine/host work; TTFT also includes request handling. Not original-paper numerical replication.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-snapshot',action='store_true')
    p.add_argument('--analysis-dir',type=Path,default=ROOT/'artifacts/phase8-figure2-context16k-window-analysis')
    p.add_argument('--snapshot',type=Path,default=HERE/'plot-data.json')
    p.add_argument('--output-dir',type=Path,default=HERE)
    p.add_argument('--focus-batches',type=int,nargs='+',help='Optional timeline/cumulative batches; dilution always shows all batches.')
    args=p.parse_args();setup()
    source=args.snapshot if args.from_snapshot else args.analysis_dir/'plot-data.json'
    data=json.loads(source.read_text());assert data['validation']['passed']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if not args.from_snapshot:
        # Compact portable snapshot; raw per-iteration/per-trial CSV remains in artifacts.
        (args.output_dir/'plot-data.json').write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    groups=[g for g in data['groups'] if g['run']=='formal']
    assert groups and all(g['prefix_mode']=='independent' for g in groups)
    batches=sorted({g['batch_size'] for g in groups})
    focus=[b for b in (args.focus_batches or [10,100,200]) if b in batches]
    assert focus
    subtitle='Single RTX 4090 | Qwen2.5-7B | Independent 128-token decode histories | 16K history + 1K new input | 5 repeats'
    timeline(groups,args.output_dir,focus,subtitle)
    dilution(groups,args.output_dir,subtitle)
    cumulative(groups,args.output_dir,focus,subtitle)
    prefill_cost(data,groups,args.output_dir,subtitle)
    print('Generated',args.output_dir)



if __name__=='__main__':main()

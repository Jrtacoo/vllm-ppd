#!/usr/bin/env python3
"""Portable plots from the diagnostic snapshot; no inference needed."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
CASES = ['h16k-b17664', 'h16k-b2048', 'h2k-b17664', 'h2k-b2048', 'literal-h2k-b2048']
LABELS = {
    'h16k-b17664': '16K history | budget 17,664',
    'h16k-b2048': '16K history | budget 2,048',
    'h2k-b17664': '2K history | budget 17,664',
    'h2k-b2048': '2K history | budget 2,048',
    'literal-h2k-b2048': 'Caption-literal control | budget 2,048',
}
SCENARIOS = {'decode_only': ('#347BB5','o','Decode only'),
             'append_prefill': ('#008577','s','Append interference'),
             'full_prefill': ('#D46A37','^','Full interference')}
CONFIG_COLORS = ['#B54442','#E49A49','#4756A0','#008577']


def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':'#BAC4CE','axes.labelcolor':'#24364B','text.color':'#24364B',
        'xtick.color':'#526273','ytick.color':'#526273','axes.grid':True,'grid.alpha':.20,
        'axes.axisbelow':True,'figure.facecolor':'white','savefig.facecolor':'white',
        'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'phase8-diagnostic-v1'})


def finish(fig, out, name, title, note):
    fig.suptitle(title,x=.045,y=.98,ha='left',fontsize=18,weight='bold')
    fig.text(.045,.925,'Single RTX 4090 · Qwen2.5-7B · Independent 128-token background histories · 3 repeats, mean ± sample SD',fontsize=10,color='#526273',va='top')
    fig.text(.045,.025,note,fontsize=9,color='#526273',va='bottom',linespacing=1.5)
    for legend in fig.legends:legend.set_bbox_to_anchor((.5,.88))
    fig.subplots_adjust(left=.075,right=.97,top=.76 if fig.get_figheight()<=7 else .80,bottom=.18,hspace=.40,wspace=.28)
    for ext in ('png','svg','pdf'):
        fig.savefig(out/(name+'.'+ext),dpi=200)
    plt.close(fig)


def draw(ax, groups, case, scenario, metric, color, marker, label, linestyle='-'):
    rs=sorted([g for g in groups if g['case']==case and g['scenario']==scenario],key=lambda g:g['batch'])
    ax.errorbar([g['batch'] for g in rs],[g[metric]['mean'] for g in rs],
        yerr=[g[metric]['std'] for g in rs],color=color,marker=marker,lw=2,ms=5,capsize=3,label=label,ls=linestyle)
    ax.set_xticks([10,100,200]);ax.set_xlabel('Observed decode batch size')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=HERE/'plot-data.json')
    p.add_argument('--output-dir',type=Path,default=HERE)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    data=json.loads(args.snapshot.read_text());assert data['validation']['passed']
    groups=data['groups'];setup()
    handles=[Line2D([0],[0],color=c,marker=m,lw=2,label=l) for c,m,l in SCENARIOS.values()]
    fig,axes=plt.subplots(2,3,figsize=(15,9),sharey=True)
    for ax,case in zip(axes.flat,CASES):
        for s,(c,m,l) in SCENARIOS.items():draw(ax,groups,case,s,'tpot_ms',c,m,l)
        ax.set_title(LABELS[case]);ax.set_ylim(bottom=0);ax.set_ylabel('Background TPOT (ms)')
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,.94,'Workload definitions',weight='bold',fontsize=12,va='top',transform=axes.flat[-1].transAxes)
    axes.flat[-1].text(0,.79,'Same-context panels:\nFull recomputes history + 1,024 new tokens.\nAppend reuses history, computes 1,024.\n\nCaption-literal control:\nFull computes a cold 1,024-token prompt.\nAppend caches 2,048 + computes 1,024.\n\nEach background request outputs 128 tokens.',va='top',linespacing=1.5,transform=axes.flat[-1].transAxes)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
    finish(fig,args.output_dir,'background-decode-tpot','History and scheduler budget: background decode TPOT',
        'Each point averages all 127 token gaps per request, then requests, then three repeats. Prefill TTFT is excluded.\nBudgets ran in two sequential engine blocks. Lines connect only three measured batch sizes; they do not establish a transition threshold.')
    fig,axes=plt.subplots(1,3,figsize=(15,7))
    for case,c in zip(CASES[:4],CONFIG_COLORS):
        for ax,metric,title in zip(axes,['prefill_chunks','mixed_model_max_ms','mixed_model_excess_ms'],
            ['Full prefill chunks','Longest mixed model step','Cumulative mixed-step excess']):
            draw(ax,groups,case,'full_prefill',metric,c,'o',LABELS[case]);ax.set_title(title)
    for ax in axes:ax.set_ylim(bottom=0)
    axes[0].set_yticks([0,1,2,4,6,8,10])
    axes[0].set_ylabel('Steps');axes[1].set_ylabel('Wall time (ms)');axes[2].set_ylabel('Wall time (ms)')
    fig.legend(handles=[Line2D([0],[0],color=c,lw=2,marker='o',label=LABELS[k]) for k,c in zip(CASES[:4],CONFIG_COLORS)],
        loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=9)
    finish(fig,args.output_dir,'full-prefill-cost','Chunking changes the peak and duration of interference',
        'Model-step wall time includes mixed prefill + background decode; it is not pure GPU kernel time.\nCumulative excess subtracts the token-position-matched decode-only model step for every prefill chunk, then sums all chunks.')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for i,history in enumerate(['16k','2k']):
        for j,batch in enumerate([10,200]):
            ax=axes[i,j]
            for budget,color in [(17664,'#B54442'),(2048,'#4756A0')]:
                for scenario,ls in [('full_prefill','-'),('append_prefill','--')]:
                    points=sorted([g for g in data['profile_groups'] if g['case']==f'h{history}-b{budget}' and g['batch']==batch and g['scenario']==scenario and -3<=g['offset']<=14 and g['client_delta_ms']['n']==3],key=lambda g:g['offset'])
                    ax.plot([g['offset'] for g in points],[g['client_delta_ms']['mean'] for g in points],color=color,ls=ls,lw=1.8)
                    ax.fill_between([g['offset'] for g in points],[g['client_delta_ms']['mean']-g['client_delta_ms']['std'] for g in points],
                        [g['client_delta_ms']['mean']+g['client_delta_ms']['std'] for g in points],color=color,alpha=.08)
            ax.axhline(0,color='#85929E',lw=.8);ax.axvline(0,color='#85929E',ls=':',lw=.8)
            ax.set_xticks([-2,0,2,4,6,8,10,12,14])
            ax.set_title(f'{history.upper()} history | batch {batch}');ax.set_xlabel('Step offset from first prefill chunk');ax.set_ylabel('Extra client token gap (ms)')
    fig.legend(handles=[Line2D([0],[0],color=c,ls=ls,lw=2,label=f'{label} · {s}') for label,c in [('Budget 17,664','#B54442'),('Budget 2,048','#4756A0')] for s,ls in [('Full','-'),('Append','--')]],loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False,fontsize=9)
    finish(fig,args.output_dir,'event-window','One long stall versus multiple shorter stalls',
        'Each token gap is paired with the same request/output ordinal in decode-only. Negative differences are retained.\nOffset 0 is the first prefill chunk. Shading: sample SD across three repeats; panels use independent vertical scales.')
    fig,axes=plt.subplots(1,2,figsize=(12,7))
    for case,c in zip(CASES[:4],CONFIG_COLORS):
        for ax,scenario in zip(axes,['full_prefill','append_prefill']):
            draw(ax,groups,case,scenario,'excess_tpot_ms',c,'o',LABELS[case]);ax.axhline(0,color='#85929E',lw=.8)
    for ax,title in zip(axes,['Full interference','Append interference']):ax.set_title(title);ax.set_ylabel('Extra background TPOT (ms/token)')
    fig.legend(handles=[Line2D([0],[0],color=c,lw=2,marker='o',label=LABELS[k]) for k,c in zip(CASES[:4],CONFIG_COLORS)],loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=9)
    finish(fig,args.output_dir,'paired-extra-tpot','Paired TPOT impact of history length and token budget',
        'Treatment TPOT minus same-repeat decode-only TPOT. Errors summarize three paired trial differences.\nAll 127 gaps are included; this is not the peak event-window delay.')


if __name__=='__main__':main()

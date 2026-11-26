#!/usr/bin/env python3
"""Portable Phase11 figures: run with Python + NumPy + Matplotlib."""
import json
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
if HERE.name=='benchmark':HERE=HERE.parents[1]/'results/figures/phase11'
D=json.loads((HERE/'plot-data.json').read_text())
BATCHES=(10,50,100,150,200,250)
SCENARIOS=('decode_only','append_prefill','full_prefill')
LABELS={'decode_only':'Decode only','append_prefill':'Append prefill','full_prefill':'Full prefill'}
COLORS={'sm':'#2765B0','read':'#DD8130','write':'#38936B'}
SCOLORS={'decode_only':'#3974AD','append_prefill':'#2D9D83','full_prefill':'#D35B65'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B5BCC4','axes.linewidth':.7,
    'grid.color':'#E5E9EE','grid.linewidth':.6,'legend.frameon':False,
    'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
R={(r['batch'],r['scenario']):r for r in D['selected']}


def hardware(r):
    z=np.load(HERE/'snapshots'/(r['trial']+'.npz'))
    return {k:z[k] for k in z.files}


def weighted_bins(t,v,edges):
    # Each sample describes (previous_timestamp, timestamp]. Integrate before rebinning.
    starts=np.r_[t[0]-.1,t[:-1]]
    dt=t-starts;integral=np.r_[0,np.cumsum(v*dt)]
    points=np.r_[starts[0],t]
    areas=np.interp(edges,points,integral,left=np.nan,right=np.nan)
    return np.diff(areas)/np.diff(edges)


def finish(fig,name,subtitle):
    fig.text(.055,.01,subtitle,ha='left',va='bottom',fontsize=8.7,color='#5E6874')
    for ext in ('png','svg','pdf'):fig.savefig(HERE/(name+'.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)


def whole():
    for b in BATCHES:
        fig,axes=plt.subplots(3,2,figsize=(13.5,8.5),sharey=True)
        fig.suptitle(f'GPU resource timeline  |  Background batch = {b}',x=.055,ha='left',fontsize=18,fontweight='bold',y=1.015)
        for row,scenario in enumerate(SCENARIOS):
            r=R[b,scenario];z=hardware(r);origin=r['capture']['start_ns'];t=(z['timestamp_ns']-origin)/1e6
            end=(r['capture']['end_ns']-origin)/1e6
            edges=np.arange(0,end,5.);edges=np.r_[edges,end]
            centers=(edges[:-1]+edges[1:])/2/1000
            for col,keys in enumerate((('sm',),('read','write'))):
                ax=axes[row,col]
                for k in keys:ax.plot(centers,weighted_bins(t,z[k],edges),lw=.85,color=COLORS[k],label={'sm':'SM active','read':'DRAM read','write':'DRAM write'}[k])
                for s in r['steps']:
                    if s['prefill_tokens']:
                        lo=(s['execute']['start_ns']-origin)/1e9;hi=(s['execute']['end_ns']-origin)/1e9
                        ax.axvspan(lo,hi,color=SCOLORS[scenario],alpha=.18,lw=0)
                        ax.axvline(lo,color=SCOLORS[scenario],lw=.7,ls='--',alpha=.8)
                ax.set_xlim(0,end/1000);ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y')
                ax.set_title(f'{LABELS[scenario]}  ·  '+('SM activity' if col==0 else 'Memory bandwidth'),loc='left')
                ax.set_xlabel('Time since first background GPU kernel (s)')
                ax.set_ylabel('Nsight throughput (%)')
                if row==0 and col==1:ax.legend(loc='upper right',ncol=2,fontsize=9,frameon=True,facecolor='white',edgecolor='none',framealpha=.9)
        fig.tight_layout(rect=(0,.035,1,.99))
        finish(fig,f'resource-timeline-b{b}',
            'RTX 4090 · Qwen2.5-7B BF16 · resident weights · fixed repeat 3 · 100 μs capture, 5 ms time-weighted display bins\n'
            'Shaded spans: actual interference prefill GPU steps. Histories warmed before capture; each scenario starts at its own first kernel.')


def chosen_steps(b):
    result={}
    target=R[b,'append_prefill']
    event=next(s for s in target['steps'] if s['prefill_tokens'])
    ordinal=min(event['background_outputs'])
    for scenario in SCENARIOS:
        r=R[b,scenario]
        if scenario=='decode_only':
            eligible=[s for s in r['steps'] if not s['background_prefill'] and s['background']==b and s['background_outputs']]
            ss=[min(eligible,key=lambda s:abs(st.mean(s['background_outputs'])-ordinal))]
        else:ss=[s for s in r['steps'] if s['prefill_tokens']]
        result[scenario]=ss
    return result


def zoom():
    for b in BATCHES:
        choices=chosen_steps(b)
        fig,axes=plt.subplots(3,1,figsize=(13,8),sharey=True)
        fig.suptitle(f'GPU forward detail  |  Background batch = {b}',x=.06,ha='left',fontsize=18,fontweight='bold',y=1.01)
        for ax,scenario in zip(axes,SCENARIOS):
            r=R[b,scenario];z=hardware(r);ss=choices[scenario]
            origin=ss[0]['execute']['start_ns'];end=ss[-1]['execute']['end_ns'];width=(end-origin)/1e6
            t=(z['timestamp_ns']-origin)/1e6;mask=(t>=-1)&(t<=width+1)
            for k in ('sm','read','write'):ax.plot(t[mask],z[k][mask],lw=.8,color=COLORS[k],label={'sm':'SM active','read':'DRAM read','write':'DRAM write'}[k])
            for i,s in enumerate(ss,1):
                lo=(s['execute']['start_ns']-origin)/1e6;hi=(s['execute']['end_ns']-origin)/1e6
                ax.axvspan(lo,hi,color='#65778C',alpha=.07)
                ax.axvline(lo,color='#59697B',lw=.8,ls='--');ax.axvline(hi,color='#59697B',lw=.8,ls='--')
                blo=(s['body']['start_ns']-origin)/1e6;bhi=(s['body']['end_ns']-origin)/1e6
                ax.plot([blo,bhi],[102,102],color=SCOLORS[scenario],lw=3,solid_capstyle='butt')
                if s['prefill_tokens']:ax.text((lo+hi)/2,95,f"chunk {i}: {s['prefill_tokens']} prefill tokens",ha='center',va='top',fontsize=9,color='#475465',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
            ax.set_xlim(-1,width+1);ax.set_ylim(0,108);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y')
            title=LABELS[scenario]+f"  ·  {sum(s['gpu_execute_ms'] for s in ss):.2f} ms across {len(ss)} GPU "+('step' if len(ss)==1 else 'steps')
            if len(ss)>1:title+=f'  ·  elapsed span {width:.2f} ms'
            if scenario=='decode_only':title+=f'  ·  matched output position ≈ {st.mean(ss[0]["background_outputs"]):.0f}'
            ax.set_title(title,loc='left');ax.set_xlabel('Time since first selected GPU kernel (ms)');ax.set_ylabel('Throughput (%)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.055,.975),ncol=3,fontsize=10)
        fig.tight_layout(rect=(0,.04,1,.95))
        finish(fig,f'forward-zoom-b{b}',
            'Fixed repeat 3 · raw 100 μs samples · each panel has its own time scale · dashed lines bound actual execute kernels\n'
            'Colored top bars: model-body GPU bounds. Full uses two prefill chunks; shown together. This is a mixed background + prefill batch.')


def summary():
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.5))
    fig.suptitle('GPU resource summary across batch sizes',x=.06,ha='left',fontsize=18,fontweight='bold',y=1.01)
    specs=[('ordinary_gpu_ms','Ordinary decode GPU step','GPU execute duration (ms)'),
           ('tpot_ms','Whole-output background TPOT','Mean token gap (ms)'),
           ('event_sm','During interference GPU steps','SM active (%)'),
           ('event_read','During interference GPU steps','DRAM read bandwidth (%)')]
    for ax,(key,title,ylabel) in zip(axes.flat,specs):
        for scenario in SCENARIOS:
            if key.startswith('event_') and scenario=='decode_only':continue
            means=[];sds=[]
            for b in BATCHES:
                vv=[r[key] for r in D['trials'] if r['batch']==b and r['scenario']==scenario]
                assert len(vv)==5
                means.append(st.mean(vv));sds.append(st.stdev(vv))
            ax.errorbar(BATCHES,means,yerr=sds,color=SCOLORS[scenario],label=LABELS[scenario],marker='o',ms=5,lw=1.6,capsize=3)
        ax.set_title(title,loc='left');ax.set_xlabel('Background decode batch size');ax.set_ylabel(ylabel)
        ax.set_xticks(BATCHES);ax.grid(axis='y');ax.set_ylim(bottom=0)
        if key.startswith('event_'):ax.set_ylim(0,100)
    axes[0,0].legend(fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.99))
    finish(fig,'resource-summary',
        'Points: mean of five trial summaries; error bars: sample SD. Ordinary decode: full background batch, output positions 32–80, no prefill.\n'
        'Event resource averages are weighted by GPU step duration across all prefill chunks. Whole-output TPOT averages 127 token gaps per request.\n'
        'B=10 Append: repeat 1 contains a 35 ms inter-kernel gap; retained in the mean and SD. See README for details.')

if __name__=='__main__':
    whole();zoom();summary();print('Generated 13 figures × PNG/SVG/PDF')

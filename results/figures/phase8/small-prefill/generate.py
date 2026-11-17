#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SC={'decode_only':('#3B6EA8','o','Decode only'),'append_prefill':('#16806B','s','Append (256 cache + 32 new)'),'full_prefill':('#C45A35','^','Full (288 recomputed)')}

def setup():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'axes.grid.axis':'y','grid.alpha':.22,'axes.axisbelow':True,'figure.facecolor':'white','savefig.facecolor':'white','axes.edgecolor':'#B8C2CC','text.color':'#24364B','xtick.color':'#526273','ytick.color':'#526273','svg.fonttype':'none','pdf.fonttype':42})
def save(fig,out,name,title,note):
 fig.suptitle(title,x=.06,y=.97,ha='left',fontsize=18,weight='bold');fig.text(.06,.915,'RTX 4090 · Qwen2.5-7B · independent 256-token background histories · 3 repeats',fontsize=10,color='#526273',va='top');fig.text(.06,.025,note,fontsize=9,color='#526273',va='bottom');fig.subplots_adjust(left=.08,right=.97,top=.78,bottom=.18,hspace=.35,wspace=.28)
 for ext in ('png','svg','pdf'):fig.savefig(out/(name+'.'+ext),dpi=220)
 plt.close(fig)
def line(ax,rows,scenario,metric, label=None):
 c,m,l=SC[scenario];rs=sorted([r for r in rows if r['scenario']==scenario],key=lambda x:x['batch']);ax.errorbar([r['batch'] for r in rs],[r[metric]['mean'] for r in rs],yerr=[r[metric]['std'] for r in rs],color=c,marker=m,lw=2,ms=5,capsize=3,label=label or l)
def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,default=Path('artifacts/phase8-small-prefill-analysis/plot-data.json'));p.add_argument('--out',type=Path,default=Path('results/figures/phase8/small-prefill'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);d=json.loads(a.snapshot.read_text());assert d['validation']['passed'];setup()
 fig,ax=plt.subplots(figsize=(10,6));
 for s in SC:line(ax,d['groups'],s,'tpot_ms')
 ax.set_xlabel('Background decode batch size');ax.set_ylabel('Mean background TPOT (ms/token)');ax.set_xticks([10,50,100,150,200,250]);ax.legend(frameon=False,ncol=3,loc='upper left');ax.set_ylim(bottom=0)
 save(fig,a.out,'background-tpot','Background decode TPOT under a 256+32 interference request','Each point averages all 127 token gaps per background request, then requests, then three repeats. Error bars: sample SD across repeats. Interferer TTFT is excluded.')
 fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharex=True)
 for s in ('append_prefill','full_prefill'):line(axes[0],d['groups'],s,'excess_tpot_ms');line(axes[1],d['groups'],s,'slowdown_pct')
 axes[0].axhline(0,color='#8795A1',lw=.8);axes[1].axhline(0,color='#8795A1',lw=.8);axes[0].set_ylabel('Extra TPOT vs decode-only (ms/token)');axes[1].set_ylabel('Slowdown vs decode-only (%)');
 for ax in axes:ax.set_xlabel('Background decode batch size');ax.set_xticks([10,50,100,150,200,250]);ax.legend(frameon=False)
 save(fig,a.out,'paired-impact','Paired interference impact on background decode','Treatment minus same-repeat decode-only baseline; all 127 token gaps are retained.')
 # fixed windows
 fig,axes=plt.subplots(1,2,figsize=(13,5.8),sharex=True)
 for ax,metric,title,ylabel in [(axes[0],'client_delta_ms','Windowed client TPOT increase','Extra client token gap (ms)'),(axes[1],'model_ms','Windowed mixed-step wall time','Model step wall time (ms)')]:
  for s in ('append_prefill','full_prefill'):
   for width,ls in [(1,'-'),(4,'--'),(8,'-.'),(16,':')]:
    rs=[r for r in d['window_groups'] if r['scenario']==s and r['width']==width and r.get('n')==3];rs=sorted(rs,key=lambda x:x['batch']);c=SC[s][0];ax.errorbar([r['batch'] for r in rs],[r[metric]['mean'] for r in rs],yerr=[r[metric]['std'] for r in rs],color=c,ls=ls,marker=SC[s][1],ms=4,lw=1.7,capsize=2)
  ax.set_xlabel('Background decode batch size');ax.set_xticks([10,50,100,150,200,250]);ax.set_ylabel(ylabel);ax.set_title(title);ax.set_ylim(bottom=0)
 handles=[Line2D([0],[0],color=SC[s][0],marker=SC[s][1],lw=2,label=SC[s][2]) for s in ('append_prefill','full_prefill')]+[Line2D([0],[0],color='#526273',ls=ls,lw=2,label=f'Window {w} step'+('s' if w>1 else '')) for w,ls in [(1,'-'),(4,'--'),(8,'-.'),(16,':')]]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=3,frameon=False,fontsize=9)
 save(fig,a.out,'event-windows','Event-window impact after the interference enters the scheduler','Offset 0 is the first complete mixed step. Windows average offsets [0,w); only windows with all three repeats and complete background batches are shown. Shaded uncertainty is omitted for readability; source data retains SD.')
 (a.out/'plot-data.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
if __name__=='__main__':main()

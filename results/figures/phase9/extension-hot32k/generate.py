#!/usr/bin/env python3
"""Audit every hot32K request and publish a self-contained figure snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path('/workspace/vllm-ppd')
E=ROOT/'artifacts/two-node-phase9-kv-transport-evidence/extension-hot32k'
DEST=ROOT/'results/figures/phase9/extension-hot32k'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def telemetry(text,tag):
    rows={}
    for line in text.splitlines():
        if 'PPD_PHASE9_'+tag+' ' in line:
            value=json.loads(line.split('PPD_PHASE9_'+tag+' ',1)[1])
            assert value['request_id'] not in rows, 'duplicate telemetry'
            rows[value['request_id']]=value
    return rows


def metrics(path,name):
    lines=path.read_text().splitlines()
    return sum(float(line.split()[-1]) for line in lines if line.startswith('vllm:'+name+'{'))


def normalize_id(value):
    # The API keeps cmpl-; engine workers append the sampling sequence suffix -0.
    return value[:-2] if value.endswith('-0') else value


def audit():
    rows=[];sources={};errors=[];block_stats=[]
    checks=[]
    def check(label,condition):
        checks.append(dict(check=label,passed=bool(condition)))
        if not condition:errors.append(label)
    manifest=json.loads((E/'manifest.json').read_text())
    for block,transport in enumerate(['socket','rdma','rdma','socket'],1):
        d=E/f'block{block}-{transport}'
        inv=json.loads((d/'before-inventory.json').read_text())
        check(f'block{block}: both GPUs idle before deployment',all(v['processes']['rc']==0 and not v['processes']['out'].strip() for v in inv.values()))
        check(f'block{block}: warmup gate passed',json.loads((d/'warmup-verification.json').read_text())['passed'])
        check(f'block{block}: static PD routing',json.loads((d/'ready-status.json').read_text())['config']=='1P_1D')
        txlog=d/'after-formal-prefill.log';rxlog=d/'after-formal-decode.log'
        tx=telemetry(txlog.read_text(),'TX');rx=telemetry(rxlog.read_text(),'RX')
        ptrace=json.loads((d/'prefill-scheduler-trace.json').read_text())
        dtrace=json.loads((d/'decode-scheduler-trace.json').read_text())
        trace_maps=[]
        for trace in (ptrace,dtrace):
            mapping={}
            for step in trace:
                for r in step['requests']:mapping.setdefault(r['id'],[]).append(r)
            trace_maps.append(mapping)
        check(f'block{block}: 7 complete TX/RX',len(tx)==len(rx)==7)
        check(f'block{block}: no scheduler preemption',all(not s['preempted'] for s in ptrace+dtrace))
        check(f'block{block}: RDMA backend',transport!='rdma' or ('NET/IB' in txlog.read_text() and 'GPU Direct RDMA Disabled' in txlog.read_text() and '/RoCE' in txlog.read_text()))
        check(f'block{block}: Socket backend',transport!='socket' or 'NET/Socket' in txlog.read_text())
        config_line=next(line for line in rxlog.read_text().splitlines() if 'HOT32K_LAUNCH ' in line)
        config=json.loads(config_line.split('HOT32K_LAUNCH ',1)[1])
        check(f'block{block}: D prefix cache explicitly disabled','--no-enable-prefix-caching' in config['command'])
        for role in ('prefill','decode'):
            launch=json.loads(next(line.split('HOT32K_LAUNCH ',1)[1] for line in (d/f'after-formal-{role}.log').read_text().splitlines() if 'HOT32K_LAUNCH ' in line))
            env=launch['environment']
            check(f'block{block}: {role} NCCL network configuration',env['NCCL_NET']==('Socket' if transport=='socket' else 'IB') and env['NCCL_IB_DISABLE']==('1' if transport=='socket' else '0') and env['NCCL_SOCKET_IFNAME']=='=ens7f0np0' and env['SIMULATED_KV_BW_GBps']=='0')
            check(f'block{block}: {role} GDR explicitly disabled',env['NCCL_NET_GDR_LEVEL']=='LOC')
            if transport=='rdma':
                check(f'block{block}: {role} RoCE HCA/GID',env['NCCL_IB_HCA']=='=rocep202s0f0:1' and env['NCCL_IB_GID_INDEX']=='3')
            provenance=json.loads((d/f'telemetry-{role}.txt').read_text())
            expected=json.loads((E/'source-hashes.json').read_text())['p2p_nccl_engine.py']
            check(f'block{block}: {role} telemetry source unchanged',provenance['instrumented_source_sha256']==expected)
        for logfile in (txlog,rxlog):
            check(f'{logfile.name}: no runtime failure',not any(p in logfile.read_text() for p in ('NCCL WARN','CUDA out of memory','KV cache does not match','kv_cache is None','Traceback')))
        for path in sorted((d/'raw').glob('*.response.json')):
            response=json.loads(path.read_text());rid=response['response_id']
            engine_id=normalize_id(rid)
            matches=[key for key in tx if normalize_id(key)==engine_id]
            if len(matches)!=1:
                check(str(path)+': unambiguous ID match',False);continue
            key=matches[0]
            if key not in rx:
                check(key+': RX match',False);continue
            pkeys=[k for k in trace_maps[0] if normalize_id(k)==normalize_id(key)]
            dkeys=[k for k in trace_maps[1] if normalize_id(k)==normalize_id(key)]
            if len(pkeys)!=1 or len(dkeys)!=1:
                check(key+': scheduler match',False);continue
            psteps=trace_maps[0][pkeys[0]];dsteps=trace_maps[1][dkeys[0]]
            row=dict(block=block,transport=transport,**response,engine_id=key,
                tx_bytes=tx[key]['payload_bytes'],rx_bytes=rx[key]['payload_bytes'],
                tx_layers=tx[key]['layers'],rx_layers=rx[key]['layers'],
                tx_nccl_sync_sum_ms=tx[key]['nccl_sync_sum_ms'],
                rx_nccl_sync_sum_ms=rx[key]['nccl_sync_sum_ms'],
                tx_send_span_ms=tx[key]['send_span_ms'],rx_receive_span_ms=rx[key]['receive_span_ms'],
                p_scheduled_tokens=sum(r['scheduled_tokens'] for r in psteps),
                p_cached_tokens=psteps[0]['cached_tokens'],
                d_scheduled_tokens=sum(r['scheduled_tokens'] for r in dsteps),
                p_steps=len(psteps),d_steps=len(dsteps),
                source=str(path.relative_to(ROOT)))
            row['e2e_minus_rx_span_ms']=row['e2e_ms']-row['rx_receive_span_ms']
            row['rx_span_fraction_of_e2e']=row['rx_receive_span_ms']/row['e2e_ms']
            rows.append(row)
            body=json.loads(path.with_name(path.name.replace('.response.json','.request.json')).read_text())
            check(key+': response success',row['success'])
            check(key+': identical prompt and one output token',hashlib.sha256(body['prompt'].encode()).hexdigest()==manifest['sha256'] and body['max_tokens']==1 and row['usage']['completion_tokens']==1)
            check(key+': complete 32K payload',row['tx_bytes']==row['rx_bytes']==32768*57344 and row['tx_layers']==row['rx_layers']==28)
            check(key+': D computes only one token',row['d_scheduled_tokens']==1)
            check(key+': expected P cache behavior',row['p_scheduled_tokens']==(32768 if row['kind']=='cold-warmup' else 16))
        pc=metrics(d/'after-formal-prefill-metrics.txt','prefix_cache_hits_total')-metrics(d/'before-formal-prefill-metrics.txt','prefix_cache_hits_total')
        dc=metrics(d/'after-formal-decode-metrics.txt','prefix_cache_hits_total')-metrics(d/'before-formal-decode-metrics.txt','prefix_cache_hits_total')
        ec=metrics(d/'after-formal-decode-metrics.txt','external_prefix_cache_hits_total')-metrics(d/'before-formal-decode-metrics.txt','external_prefix_cache_hits_total')
        check(f'block{block}: formal P hit tokens',pc==5*(32768-16))
        check(f'block{block}: formal D local hit zero',dc==0)
        check(f'block{block}: formal D external hit tokens',ec==5*(32768-1))
        block_stats.append(dict(block=block,transport=transport,p_local_hit_tokens=pc,d_local_hit_tokens=dc,d_external_hit_tokens=ec,max_observer_us=max(s['observer_us'] for s in ptrace+dtrace)))
    formal=[r for r in rows if r['kind']=='formal']
    check('20 formal requests',len(formal)==20)
    check('8 warmup requests',len(rows)-len(formal)==8)
    check('all request IDs distinct',len({r['response_id'] for r in rows})==len(rows))
    # Prepared inventory may predate authorization; compare the actual run baseline.
    initial=json.loads((E/'block1-socket/before-inventory.json').read_text())
    final=json.loads((E/'final-inventory.json').read_text())
    a=json.loads(initial['dcn9']['original_proxy']['out'])[0]
    b=json.loads(final['dcn9']['original_proxy']['out'])[0]
    check('existing proxy ID, StartedAt and config preserved',a['Id']==b['Id'] and a['State']['StartedAt']==b['State']['StartedAt'] and a['Config']==b['Config'])
    for node in ('dcn9','dcn10'):
        check(node+': experiment containers removed','ppd-p9hot-' not in final[node]['containers']['out'])
        check(node+': GPUs released after cleanup',final[node]['processes']['rc']==0 and not final[node]['processes']['out'].strip())
    summary=[]
    fields=['ttft_ms','e2e_ms','tx_nccl_sync_sum_ms','rx_nccl_sync_sum_ms','tx_send_span_ms','rx_receive_span_ms','e2e_minus_rx_span_ms','rx_span_fraction_of_e2e']
    for transport in ('socket','rdma'):
        group=[r for r in formal if r['transport']==transport]
        summary.append(dict(transport=transport,requests=len(group),metrics={f:dict(median=statistics.median(r[f] for r in group),mean=statistics.mean(r[f] for r in group),min=min(r[f] for r in group),max=max(r[f] for r in group)) for f in fields}))
    for path in E.rglob('*'):
        mutable={'status.json','verification.json','measurements.json','sync-manifest.json','sync-verification.json','analysis.log','all-requests.csv'}
        if path.parent==E and path.name in mutable:
            continue
        if path.is_file() and (path.suffix in ('.log','.json','.jsonl','.txt','.sse','.py')):
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    verification=dict(passed=not errors,errors=errors,checks=checks,blocks=block_stats)
    write(E/'verification.json',verification)
    write(E/'measurements.json',dict(rows=rows,summary=summary))
    with (E/'all-requests.csv').open('w',newline='') as f:
        keys=[k for k in rows[0] if k!='usage'];writer=csv.DictWriter(f,keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    if errors:raise RuntimeError('Audit failed: '+str(errors))
    DEST.mkdir(parents=True,exist_ok=True)
    write(DEST/'plot-data.json',dict(method='P-hot/D-cold 32768-token PD; output=1; NCCL IB uses host-staged RoCE; application transfer spans are not pure wire latency',formal=formal,warmup=[r for r in rows if r['kind']!='formal'],summary=summary,verification=verification,source_sha256=sources))
    print(json.dumps(dict(verification=verification['passed'],summary=summary),indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    data=json.loads((DEST/'plot-data.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'#f6f8fb','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
    panels=[('e2e_ms','Client end-to-end latency'),('rx_receive_span_ms','KV receive span'),('tx_nccl_sync_sum_ms','NCCL send calls, summed'),('rx_nccl_sync_sum_ms','NCCL receive calls, summed')]
    colors=['#4878b8','#159d8c']
    fig,axes=plt.subplots(1,4,figsize=(15,4.8))
    rng=np.random.default_rng(9)
    for ax,(field,title) in zip(axes,panels):
        for x,(transport,color) in enumerate(zip(('socket','rdma'),colors)):
            values=[r[field] for r in data['formal'] if r['transport']==transport]
            ax.scatter(x+rng.uniform(-.09,.09,len(values)),values,color=color,s=33,alpha=.65,zorder=3,edgecolor='white',linewidth=.5)
            med=statistics.median(values)
            ax.errorbar(x,med,yerr=[[med-min(values)],[max(values)-med]],color=color,capsize=6,lw=1.5,zorder=4)
            ax.plot([x-.17,x+.17],[med,med],color=color,lw=3,zorder=5)
            ax.annotate(f'{med:.1f}',(x,max(values)),xytext=(0,10),textcoords='offset points',ha='center',weight='bold',color=color)
        ax.set_xticks([0,1],['Socket / TCP','RoCE / RDMA']);ax.set_xlim(-.5,1.5)
        ax.set_ylim(0,max(r[field] for r in data['formal'])*1.25)
        ax.set_title(title,fontsize=11,pad=12);ax.set_ylabel('Time (ms)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('32K KV transfer with a warm Prefill cache',fontsize=18,weight='bold',y=.98)
    fig.text(.5,.88,'1.75 GiB per request · 28 layers · one output token · 10 samples per transport',ha='center',color='#526174')
    fig.text(.5,.035,'Dots: all formal requests   |   Thick line: median   |   Whiskers: min–max\nRoCE uses host-memory staging. Transfer spans and NCCL call times include software overhead and waiting.',ha='center',fontsize=10,color='#526174')
    fig.subplots_adjust(left=.055,right=.985,top=.77,bottom=.23,wspace=.43)
    for ext in ('png','svg','pdf'):fig.savefig(DEST/f'hot32k-transport.{ext}',dpi=200)
    plt.close(fig)


def main():
    global DEST
    p=argparse.ArgumentParser();p.add_argument('--from-snapshot',action='store_true');args=p.parse_args()
    if args.from_snapshot and (Path(__file__).parent/'plot-data.json').exists():
        DEST=Path(__file__).parent
    if not args.from_snapshot:audit()
    plot()


if __name__=='__main__':main()

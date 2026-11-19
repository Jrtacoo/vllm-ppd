#!/usr/bin/env python3
"""Audit Phase 9 evidence and render portable plots. No service operations."""
import argparse
import csv
import hashlib
import json
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SIZES = (256,512,1024,2048,4096,8192)
TRANSPORTS = ("socket","rdma")
COLORS = ("#426CB4","#009B83")
LABELS = ("Socket / TCP","RoCE (host staging)")


def load_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")


def telemetry(text, tag):
    result = {}
    for line in text.splitlines():
        if tag+" " in line:
            item = json.loads(line.split(tag+" ",1)[1])
            if item["request_id"] in result:
                raise ValueError("duplicate telemetry request ID")
            result[item["request_id"]] = item
    return result


def read_evidence(e):
    rows, blocks, checks = [], [], []
    files = set()
    for repeat in range(1,6):
        for transport in TRANSPORTS:
            label=f"formal-r{repeat}-{transport}"
            run=e/label
            before,after = {},{}
            for role in ("prefill","decode"):
                bp=e/f"{label}-before-{role}.log"
                ap=e/f"{label}-after-{role}.log"
                files.update([bp,ap])
                before[role]=bp.read_text()
                after[role]=ap.read_text()
                assert after[role].startswith(before[role]), f"nonmonotonic log {label} {role}"
            dl=after["decode"][len(before["decode"]):]
            pl=after["prefill"][len(before["prefill"]):]
            rx=telemetry(dl,"PPD_PHASE9_RX")
            tx=telemetry(pl,"PPD_PHASE9_TX")
            assert len(rx)==len(tx)==6, (label,len(rx),len(tx))
            connector=re.findall(r"KV transfer timing:.*recv_time=([\d.]+)ms, layers=(\d+), blocks=(\d+)",dl)
            assert len(connector)==6
            assert "Using network "+("IB" if transport=="rdma" else "Socket") in after["decode"]
            assert "Using network "+("IB" if transport=="rdma" else "Socket") in after["prefill"]
            if transport=="rdma":
                for role in ("prefill","decode"):
                    assert "GPU Direct RDMA Disabled" in after[role]
                    assert "rocep202s0f0:1/RoCE" in after[role]
                    assert "NCCL_IB_GID_INDEX set by environment to 3" in after[role]
            problem=re.findall(r"[^\n]*(?:ERROR|Traceback|Out Of Memory|Out Of Threshold|kv_cache is None|does not match|NCCL WARN)[^\n]*",dl+"\n"+pl)
            assert not problem, (label,problem)
            status=[]
            for edge in ("before","after"):
                p=e/f"{label}-{edge}-proxy-status.json"
                files.add(p)
                status.append(load_json(p))
            assert all(s["config"]=="1P_1D" and not s["ppd_mode_enabled"] for s in status)
            delta={k:status[1]["stats"][k]-status[0]["stats"][k] for k in ("total_requests","pd_requests","ppd_requests","ppd_direct_requests","ppd_dynamic_requests")}
            assert delta==dict(total_requests=6,pd_requests=6,ppd_requests=0,ppd_direct_requests=0,ppd_dynamic_requests=0),delta
            request_rows=list(csv.DictReader((run/"requests.csv").open()))
            assert len(request_rows)==6
            for i,r in enumerate(request_rows):
                n=int(r["input_tokens"])
                assert n==SIZES[i]
                stem=run/"raw"/r["id"]
                paths={s:stem.with_suffix(s) for s in (".request.json",".response.json",".events.json",".sse")}
                files.update(paths.values())
                response=load_json(paths[".response.json"])
                req=load_json(paths[".request.json"])
                events=load_json(paths[".events.json"])
                assert response["success"] and not response["error"] and response["completion_tokens"]==32
                assert response["prompt_tokens_reported"]==n and response["finish_reason"]=="length"
                assert events[-1]["data"]=="[DONE]"
                assert req["max_tokens"]==32 and req["ignore_eos"] and req["temperature"]==0
                rid=response["response_id"]
                engine_id = rid if rid in rx else rid+"-0"
                a,b=rx[engine_id],tx[engine_id]
                assert a["layers"]==b["layers"]==28
                assert a["payload_bytes"]==b["payload_bytes"]==n*57344
                assert tuple(map(int,connector[i][1:]))==(28,n//16)
                rows.append(dict(transport=transport,repeat=repeat,id=r["id"],input_tokens=n,
                    response_id=rid,success=True,prompt_tokens=n,completion_tokens=32,
                    ttft_ms=response["ttft_ms"],e2e_ms=response["e2e_ms"],
                    payload_bytes=a["payload_bytes"],layers=a["layers"],blocks=n//16,
                    rx_nccl_sync_sum_ms=a["nccl_sync_sum_ms"],rx_receive_span_ms=a["receive_span_ms"],
                    tx_nccl_sync_sum_ms=b["nccl_sync_sum_ms"],tx_send_span_ms=b["send_span_ms"],
                    connector_load_ms=float(connector[i][0]),
                    payload_sha256=hashlib.sha256(paths[".request.json"].read_bytes()).hexdigest(),
                    raw_stem=str(stem.relative_to(e))))
            counter_deltas={}
            for node in ("dcn9","dcn10"):
                cs=[]
                for edge in ("before","after"):
                    p=e/f"{label}-{edge}-{node}-counters.json"
                    files.add(p); cs.append(load_json(p))
                counter_deltas[node]={k:cs[1][k]-cs[0][k] for k in cs[0]}
            # HCA data counters are 4-octet units; Ethernet stats use octets.
            if transport=="rdma":
                rx_bytes=counter_deltas["dcn10"]["port_rcv_data"]*4
                tx_bytes=counter_deltas["dcn9"]["port_xmit_data"]*4
            else:
                rx_bytes=counter_deltas["dcn10"]["rx_bytes"]
                tx_bytes=counter_deltas["dcn9"]["tx_bytes"]
            expected=sum(SIZES)*57344
            assert rx_bytes>=expected and tx_bytes>=expected
            blocks.append(dict(repeat=repeat,transport=transport,payload_bytes=expected,
                node_dcn10_received_bytes=rx_bytes,node_dcn9_sent_bytes=tx_bytes,
                counters=counter_deltas,proxy_delta=delta))
            metrics = {}
            for role in ("prefill","decode"):
                vals=[]
                for edge in ("before","after"):
                    path=e/f"{label}-{edge}-{role}-metrics.txt"
                    files.add(path)
                    metric_text=path.read_text()
                    def total(name):
                        matches=re.findall(r"^vllm:"+name+r"\{[^\n]*\} ([0-9.eE+\-]+)$",metric_text,re.M)
                        return sum(float(v) for v in matches)
                    vals.append({k:total(k) for k in (
                        "prefix_cache_hits_total","prefix_cache_queries_total",
                        "external_prefix_cache_hits_total","prompt_tokens_total")})
                metrics[role]={k:vals[1][k]-vals[0][k] for k in vals[0]}
                assert metrics[role]["prefix_cache_hits_total"]==0, (label,role,metrics)
                assert metrics[role]["prompt_tokens_total"]==sum(SIZES), (label,role,metrics)
            blocks[-1]["metrics_delta"]=metrics
            checks.append(dict(label=label,passed=True,formal_requests=6,telemetry_pairs=6,
                full_layer_receives=6,network="IB/RoCE" if transport=="rdma" else "Socket/TCP"))
    assert len(rows)==60
    for n in SIZES:
        for repeat in range(1,6):
            pair=[r for r in rows if r["input_tokens"]==n and r["repeat"]==repeat]
            assert len(pair)==2 and pair[0]["payload_sha256"]==pair[1]["payload_sha256"]
    provenance=load_json(e.parent/"telemetry/provenance.json")
    data=dict(source_directory=str(e.relative_to(ROOT)),requests=rows,blocks=blocks,
        verification=dict(passed=True,requests=60,matched_payload_pairs=30,checks=checks),
        timing_definitions=provenance,
        source_sha256={str(p.relative_to(e)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    with (e/"audited-requests.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save_json(e/"verification.json",data["verification"])
    save_json(e/"network-counter-deltas.json",blocks)
    return data


def validate(d):
    rows=d["requests"]
    assert len(rows)==60 and d["verification"]["passed"]
    for n in SIZES:
        for t in TRANSPORTS:
            rr=[r for r in rows if r["input_tokens"]==n and r["transport"]==t]
            assert len(rr)==5 and all(r["success"] and r["payload_bytes"]==n*57344 and r["layers"]==28 for r in rr)
            assert all(r[k]>0 and np.isfinite(r[k]) for r in rr for k in
                       ("ttft_ms","e2e_ms","rx_nccl_sync_sum_ms","rx_receive_span_ms"))


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#20324B",
        "axes.labelcolor":"#3F526A","xtick.color":"#53667C","ytick.color":"#53667C",
        "axes.edgecolor":"#CBD5E1","axes.spines.top":False,"axes.spines.right":False,
        "axes.titleweight":"bold","axes.titlesize":11,"figure.facecolor":"white",
        "axes.facecolor":"white","svg.fonttype":"none","pdf.fonttype":42,"svg.hashsalt":"phase9"})


def metric(ax,d,key,title,ylabel):
    x=np.arange(len(SIZES))
    for ti,(t,color,label) in enumerate(zip(TRANSPORTS,COLORS,LABELS)):
        mid=[]; lower=[]; upper=[]
        for i,n in enumerate(SIZES):
            vals=[r[key] for r in d["requests"] if r["input_tokens"]==n and r["transport"]==t]
            med=st.median(vals);mid.append(med);lower.append(med-min(vals));upper.append(max(vals)-med)
            ax.scatter(i+np.linspace(-.055,.055,5)+(ti-.5)*.12,vals,s=16,
                       alpha=.38,color=color,zorder=3,edgecolors="none")
        ax.errorbar(x+(ti-.5)*.12,mid,yerr=[lower,upper],color=color,fmt="o-",
                    capsize=3,lw=2,ms=5,label=label,elinewidth=1,alpha=.95,zorder=4)
    ax.set_title(title,loc="left",pad=14)
    ax.set_xticks(x,[f"{n:,}" for n in SIZES])
    ax.set_xlabel("Prefill input tokens",labelpad=9)
    ax.set_ylabel(ylabel,labelpad=8)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",color="#E4EAF2",lw=.7)
    ax.set_axisbelow(True)


def save(fig,name):
    for ext in ("png","svg","pdf"):
        fig.savefig(HERE/f"{name}.{ext}",dpi=210)
    plt.close(fig)


def render(d):
    style()
    fig,axes=plt.subplots(2,2,figsize=(13,9.6))
    axes=axes.ravel()
    fig.subplots_adjust(left=.08,right=.97,top=.80,bottom=.19,wspace=.27,hspace=.48)
    fig.text(.045,.95,"PHASE 9  /  TWO-NODE PD",color="#008577",fontsize=10,weight="bold")
    fig.text(.045,.905,"How KV transfer changes with input size",fontsize=21,weight="bold")
    fig.text(.045,.86,"2 x RTX 4090  |  Qwen2.5-7B-Instruct BF16  |  100 Gb/s  |  Serial requests  |  32 output tokens",fontsize=10,color="#53667C")
    ax=axes[0]
    mib=np.array([n*57344/2**20 for n in SIZES])
    ax.bar(np.arange(6),mib,color="#D9E5F2",width=.58)
    ax.plot(np.arange(6),mib,"o",color="#426CB4",label="Observed RX = TX payload")
    for i,v in enumerate(mib):ax.text(i,v+8,f"{v:g}",ha="center",fontsize=10)
    ax.set_title("A   Actual tensor payload / request",loc="left",pad=14)
    ax.set_xticks(np.arange(6),[f"{n:,}" for n in SIZES]);ax.set_xlabel("Prefill input tokens",labelpad=9)
    ax.set_ylabel("KV tensor payload (MiB)",labelpad=8);ax.set_ylim(0,520)
    ax.grid(axis="y",color="#E4EAF2",lw=.7);ax.set_axisbelow(True)
    metric(axes[1],d,"rx_nccl_sync_sum_ms","B   Sum of synchronized NCCL receives","Time (ms)")
    metric(axes[2],d,"tx_nccl_sync_sum_ms","C   Sum of synchronized NCCL sends","Time (ms)")
    metric(axes[3],d,"rx_receive_span_ms","D   First PUT to last completed receive","Time (ms)")
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.095),ncol=2,frameon=False)
    fig.text(.045,.045,"Median + min/max; dots = all 5 repeats. Synchronized NCCL calls include peer waiting; span also includes allocation and layer gaps.\nTensor payload is counted from shape and dtype; interface counters include protocol overhead. These are not pure wire latencies.",fontsize=9,color="#53667C",linespacing=1.5)
    save(fig,"kv-transport")
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.8))
    fig.subplots_adjust(left=.08,right=.97,top=.73,bottom=.25,wspace=.27)
    fig.text(.05,.95,"PHASE 9  /  TWO-NODE PD",color="#008577",fontsize=10,weight="bold")
    fig.text(.05,.885,"Request latency: Socket / TCP vs RoCE",fontsize=21,weight="bold")
    fig.text(.05,.825,"Matched inputs, 5 paired repeats per size  |  Separate warmup after each deployment  |  60 formal requests",fontsize=10,color="#53667C")
    metric(axes[0],d,"ttft_ms","A   Time to first text","TTFT (ms)")
    metric(axes[1],d,"e2e_ms","B   End-to-end latency","Latency (ms)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.105),ncol=2,frameon=False)
    fig.text(.05,.045,"Median + min/max; dots = all requests. TTFT includes Proxy, prefill, KV transfer and Decode startup.\n2 x RTX 4090 | RoCE uses host staging (no GPUDirect RDMA) | 32 output tokens | No concurrent requests.",fontsize=9,color="#53667C",linespacing=1.5)
    save(fig,"request-latency")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-snapshot",action="store_true")
    p.add_argument("--evidence",type=Path,default=ROOT/"artifacts/two-node-phase9-kv-transport-evidence/calibrated")
    args=p.parse_args()
    d=load_json(HERE/"plot-data.json") if args.from_snapshot else read_evidence(args.evidence)
    validate(d)
    if not args.from_snapshot:save_json(HERE/"plot-data.json",d)
    render(d)
    for n in SIZES:
        groups={t:[r for r in d["requests"] if r["input_tokens"]==n and r["transport"]==t] for t in TRANSPORTS}
        print(n,{t:{k:round(st.median(r[k] for r in rr),3) for k in
            ("rx_nccl_sync_sum_ms","rx_receive_span_ms","ttft_ms","e2e_ms")} for t,rr in groups.items()})
    print("Validated 60 requests / 30 identical-payload pairs; wrote 2 figures in PNG/SVG/PDF.")


if __name__=="__main__":main()

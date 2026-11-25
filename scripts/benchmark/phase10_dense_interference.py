#!/usr/bin/env python3
"""Dense Phase 10: paired workload-normalized interference, B=10..250."""
import argparse
import asyncio
import csv
import json
import random
import statistics as st
import time
from pathlib import Path

import aiohttp
from figure2_interference import MODEL, FIELDS, stream, write_json

BATCHES = tuple(range(10, 251))
CONTROL_BATCHES = (10, 50, 100, 150, 200, 250)
PHASE_REPEATS = 5
WORK_PER_EVENT = 4800
ACTIVE_START, ACTIVE_END = 8, 104
EXTRA_FIELDS = ('regime', 'event_index', 'trigger_tokens')


def phase_numerator(batch, repeat, seed=42):
    """Five evenly spaced phases, permuted once per batch independently of timings."""
    phases = [1, 3, 5, 7, 9]
    random.Random(f'phase10-dense/phase/{seed}/{batch}').shuffle(phases)
    return phases[(repeat - 1) % PHASE_REPEATS]


def triggers(batch, regime, repeat=1, seed=42):
    if regime == 'baseline':
        return []
    if regime == 'single':
        return [8]
    phase = phase_numerator(batch, repeat, seed)
    goals = []
    for event in range(6):
        numerator = WORK_PER_EVENT * (10 * event + phase)
        advance = (numerator + 10 * batch - 1) // (10 * batch)
        goal = ACTIVE_START + advance
        if goal <= ACTIVE_END:
            goals.append(goal)
    return goals


def construct(pool, batch, repeat, seed):
    def tokens(kind, index, size):
        rng = random.Random(f'phase10/{seed}/{repeat}/{batch}/{kind}/{index}')
        return [rng.choice(pool) for _ in range(size)]
    backgrounds = [tokens('background-history', i, 256) + tokens('background-new', i, 32)
                   for i in range(batch)]
    interferers = [tokens('interference-history', i, 2048) + tokens('interference-new', i, 128)
                   for i in range(max(1, (batch + 49) // 50))]
    return backgrounds, interferers


def schedule_for(args):
    schedule = []
    for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
        cells = []
        for batch in args.batches:
            cells.append(dict(batch_size=batch, repeat=repeat, regime='baseline', scenario='decode_only'))
            for scenario in ('append_prefill', 'full_prefill'):
                cells.append(dict(batch_size=batch, repeat=repeat, regime='scaled', scenario=scenario))
                if batch in CONTROL_BATCHES and not args.pilot:
                    cells.append(dict(batch_size=batch, repeat=repeat, regime='single', scenario=scenario))
        random.Random(f'phase10/order/{args.seed}/{repeat}/{args.namespace}').shuffle(cells)
        schedule.extend(cells)
    return schedule


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'raw').mkdir()
    schedule = schedule_for(args)
    write_json(args.output/'run.json', dict(version=2, model=MODEL, seed=args.seed,
        namespace=args.namespace, server=args.server, schedule=schedule,
        background=dict(history=256, new=32, output=128, independent=True),
        interferer=dict(history=2048, new=128, output=1, independent=True),
        token_budget=2048, max_num_seqs=251, prefix_reset='before each trial',
        trigger_table={str(b):triggers(b, 'scaled', args.repeat_start, args.seed) for b in args.batches},
        workload_rule=dict(active_start=8, active_end=104, work_per_event=4800,
            phase_repeats=5, phase='permuted midpoint grid, five repeats; integer ceil output goals',
            target_events='B/50; zero-event repeats retained in whole-output average'),
        statistic='mean all 127 token arrival gaps per background, then requests, then repeats',
        event_windows=[1,5,16], failure_policy='retain; no inference retries; stop on failed checks',
        persistence='buffer raw evidence during inference; flush after every in-flight request has ended'))
    connector = aiohttp.TCPConnector(limit=300, limit_per_host=300, force_close=True)
    timeout = aiohttp.ClientTimeout(total=180)
    with (args.output/'requests.jsonl').open('x') as records, (args.output/'requests.csv').open('x',newline='') as cf, (args.output/'trials.jsonl').open('x') as tf:
        writer=csv.DictWriter(cf,fieldnames=list(FIELDS)+list(EXTRA_FIELDS));writer.writeheader()
        async with aiohttp.ClientSession(connector=connector,timeout=timeout,trust_env=False) as session:
            async with session.get(args.server+'/health') as r:r.raise_for_status()
            async with session.post(args.server+'/tokenize',json={'model':MODEL,'prompt':
                    'Reference notes: data model serving cache memory inference request token sequence analysis performance. '*16,
                    'add_special_tokens':False}) as r:
                r.raise_for_status();pool=(await r.json())['tokens']
            write_json(args.output/'token-pool.json',pool)
            for sequence,meta in enumerate(schedule):
                batch=meta['batch_size']; repeat=meta['repeat'];scenario=meta['scenario'];regime=meta['regime']
                tid=f'{args.namespace}-b{batch}-r{repeat}-{regime}-{scenario}'
                backgrounds,interferers=construct(pool,batch,repeat,args.seed)
                # All five conditions warm exactly the same independent histories.
                prefixes=[p[:256] for p in backgrounds]+[p[:2048] for p in interferers]
                salts=[f'{tid}/history/{i}' for i in range(len(prefixes))]
                async with session.post(args.server+'/reset_prefix_cache') as r:
                    r.raise_for_status(); reset=dict(status=r.status,body=await r.text(),perf_ns=time.perf_counter_ns())
                write_json(args.output/f'{tid}-reset.json',reset)
                artifacts=[]; rows=[]

                def flush():
                    for stem,payload,raw,events,response in artifacts:
                        write_json(Path(str(stem)+'.request.json'),payload)
                        Path(str(stem)+'.sse').write_bytes(raw)
                        write_json(Path(str(stem)+'.events.json'),events)
                        write_json(Path(str(stem)+'.response.json'),response)
                    artifacts.clear()
                    for row in rows:
                        records.write(json.dumps(row,ensure_ascii=False)+'\n')
                        writer.writerow({k:row.get(k) for k in list(FIELDS)+list(EXTRA_FIELDS)})
                    records.flush();cf.flush();rows.clear()

                async def request(prompt,role,index,salt,output,callback=None,trigger=None):
                    name=f'{tid}_{role}_{index}'
                    payload=dict(model=MODEL,prompt=prompt,max_tokens=output,temperature=0,seed=args.seed,
                        ignore_eos=True,stream=True,stream_options={'include_usage':True},
                        return_token_ids=True,request_id=name,cache_salt=salt)
                    row=await stream(session,args.server,payload,args.output/'raw'/name,callback,artifacts)
                    row.update(meta,trial=tid,role=role,event_index=index if role=='interferer' else None,
                        trigger_tokens=trigger,raw_stem='raw/'+name)
                    rows.append(row);return row

                semaphore=asyncio.Semaphore(4)
                async def warm(i):
                    async with semaphore:return await request(prefixes[i],'warmup',i,salts[i],1)
                warmups=await asyncio.gather(*(warm(i) for i in range(len(prefixes))))
                flush()
                if not all(r['success'] for r in warmups):raise RuntimeError(tid+': warmup failed; retained')
                async with session.get(args.server+'/metrics') as r:
                    r.raise_for_status();(args.output/f'{tid}-before-metrics.txt').write_text(await r.text())
                progress=[0]*batch; changed=asyncio.Event()
                def callback(i,n):progress[i]=n;changed.set()
                start=time.perf_counter_ns();start_wall=time.time_ns()
                bg_tasks=[asyncio.create_task(request(p,'decode',i,salts[i],128,
                    lambda n,i=i:callback(i,n))) for i,p in enumerate(backgrounds)]
                int_tasks=[];injections=[];error=None
                goals=[] if scenario=='decode_only' else triggers(batch,regime,repeat,args.seed)
                try:
                    for i,goal in enumerate(goals):
                        deadline=time.monotonic()+90
                        while min(progress)<goal:
                            changed.clear()
                            if any(t.done() for t in bg_tasks):
                                raise RuntimeError('background ended before scheduled interference')
                            await asyncio.wait_for(changed.wait(),timeout=max(.001,deadline-time.monotonic()))
                        injections.append(dict(event_index=i,trigger_tokens=goal,
                            target_decode_work=(goal-8)*batch, actual_min_decode_work=(min(progress)-8)*batch,
                            dispatch_ns=time.perf_counter_ns(),progress=list(progress)))
                        salt=salts[batch+i] if scenario=='append_prefill' else f'{tid}/cold/{i}'
                        int_tasks.append(asyncio.create_task(request(interferers[i],'interferer',i,salt,1,trigger=goal)))
                        await asyncio.sleep(0)
                except Exception as exc:error=f'{type(exc).__name__}: {exc}'
                results=await asyncio.gather(*bg_tasks,*int_tasks,return_exceptions=True)
                flush()
                async with session.get(args.server+'/metrics') as r:
                    r.raise_for_status();(args.output/f'{tid}-after-metrics.txt').write_text(await r.text())
                decoded=[r for r in results if isinstance(r,dict) and r['role']=='decode']
                intf=[r for r in results if isinstance(r,dict) and r['role']=='interferer']
                checks=dict(all_success=all(isinstance(r,dict) and r['success'] for r in results),
                    background_count=len(decoded)==batch,background_cache=all(r['cached_tokens']==256 for r in decoded),
                    interference_count=len(intf)==len(goals),
                    interference_cache=all(r['cached_tokens']==(2048 if scenario=='append_prefill' else 0) for r in intf),
                    triggers=all(min(i['progress'])>=i['trigger_tokens'] for i in injections),
                    overlap=all(d['start_ns']<i['start_ns']<i['end_ns']<d['end_ns'] for i in intf for d in decoded),
                    no_client_error=error is None)
                summary=dict(meta,trial=tid,sequence=sequence,start_ns=start,end_ns=time.perf_counter_ns(),
                    start_wall_ns=start_wall,end_wall_ns=time.time_ns(),
                    injections=injections,checks=checks,error=error,
                    decode_avg_tpot_ms=st.mean(r['token_tpot_ms'] for r in decoded) if checks['all_success'] else None)
                tf.write(json.dumps(summary)+'\n');tf.flush()
                print(json.dumps(summary),flush=True)
                if not all(checks.values()):raise RuntimeError(tid+': trial checks failed; evidence retained')
    print('COLLECTION COMPLETE',len(schedule),'trials',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',default='http://192.168.1.11:8300')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--namespace',required=True)
    p.add_argument('--batches',type=int,nargs='+',default=list(BATCHES))
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--repeat-start',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--pilot',action='store_true')
    args=p.parse_args()
    if args.repeat_start<1 or args.repeat_start+args.repeats>6 or args.repeats<1 or any(b not in BATCHES for b in args.batches):p.error('Invalid repeat count or batch')
    asyncio.run(run(args))
if __name__=='__main__':main()

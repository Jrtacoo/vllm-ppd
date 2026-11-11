#!/usr/bin/env python3
"""Two-GPU NCCL send/receive probe, independent of model serving.

Run one rank per host in containers exposing /dev/infiniband. Environment selects
NCCL transport. Timing includes a GPU acknowledgement and Python/CUDA overhead;
it is a payload round trip, not raw one-way RDMA latency or a serving benchmark.
"""
import argparse
from datetime import timedelta
import json
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", default="192.168.1.9")
    parser.add_argument("--master-port", type=int, default=18620)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 1048576, 16777216, 67108864])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1 or any(n < 1 or n > 67108864 for n in args.sizes):
        parser.error("positive iterations/repeats and sizes in 1..64 MiB required")

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                            rank=args.rank, world_size=2, timeout=timedelta(seconds=60),
                            device_id=torch.device("cuda:0"))
    marker = torch.tensor([args.rank + 1], dtype=torch.int32, device="cuda")
    dist.all_reduce(marker)
    torch.cuda.synchronize()
    if marker.item() != 3:
        raise RuntimeError("initial NCCL all-reduce validation failed")
    records = []
    for size in args.sizes:
        send = torch.full((size,), args.rank + 7, dtype=torch.uint8, device="cuda")
        recv = torch.zeros_like(send)
        ack = torch.ones((1,), dtype=torch.int32, device="cuda")
        for source in (0, 1):
            target = 1 - source

            def transfer():
                if args.rank == source:
                    dist.send(send, target)
                    dist.recv(ack, target)
                else:
                    dist.recv(recv, source)
                    dist.send(ack, source)
                torch.cuda.synchronize()

            dist.barrier()
            for _ in range(3):
                transfer()
            samples = []
            for _ in range(args.repeats):
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    transfer()
                samples.append((time.perf_counter() - start) * 1000 / args.iterations)
            correct = bool(torch.all(recv == source + 7).item()) if args.rank == target else None
            if correct is False:
                raise RuntimeError(f"payload mismatch: {size} bytes, source rank {source}")
            median = statistics.median(samples)
            row = dict(rank=args.rank, source_rank=source, target_rank=target, bytes=size,
                       iterations=args.iterations, repeats=args.repeats,
                       round_trip_ms=samples, median_round_trip_ms=median,
                       payload_goodput_gbps=size * 8 / (median / 1000) / 1e9,
                       payload_correct=correct)
            records.append(row)
            print("RDMA_PROBE_JSON=" + json.dumps(row), flush=True)
        del send, recv, ack
    dist.barrier()
    dist.destroy_process_group()
    print("RDMA_PROBE_DONE=" + json.dumps({"rank": args.rank, "records": len(records),
          "torch": torch.__version__, "nccl": torch.cuda.nccl.version(),
          "max_allocated_bytes": torch.cuda.max_memory_allocated()}), flush=True)


if __name__ == "__main__":
    main()

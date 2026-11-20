"""Decode 尺寸 all-reduce 微基準：RCCL/NCCL vs Iris（graph 內計時，每呼叫 µs）。
用法（容器內，W 張 GPU）：python3 bench_allreduce.py [--world 4] [--hidden 5120] [--rows 6,24,48,96,192,384]
加新的實作：在 IMPLS 裡加一個 (name, setup_fn) —— setup_fn(rank, dev, max_numel) 回傳 fn(x)->out。
"""

import argparse
import socket
import sys
import time
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def timed(fn, inner=43, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / (iters * inner) * 1e6


def setup_rccl(rank, dev, mx):
    return lambda x: (dist.all_reduce(x), x)[1]


def setup_iris(rank, dev, mx):
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_all_reduce,
    )

    st = create_iris_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=mx,
        producer_direct_max_numel=mx,
        dtype=torch.bfloat16,
        device=dev,
        heap_size=1 << 30,
    )
    return lambda x: iris_all_reduce(st, x, safe=False)


IMPLS = [("rccl", setup_rccl), ("iris", setup_iris)]


def worker(rank, a, port):
    try:
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(dev)
        dist.init_process_group(
            "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=a.world
        )
        rows = [int(r) for r in a.rows.split(",")]
        mx = max(rows) * a.hidden
        fns = {}
        for name, setup in IMPLS:
            try:
                fns[name] = setup(rank, dev, mx)
            except Exception as e:
                if rank == 0:
                    print(f"{name}: setup failed: {str(e)[:100]}")
        torch.cuda.synchronize()
        dist.barrier()
        for r in rows:
            n = r * a.hidden
            x = torch.randn(n, dtype=torch.bfloat16, device=dev) * (rank + 1)
            ref = x.clone()
            dist.all_reduce(ref)
            line = []
            for name, fn in fns.items():
                try:
                    out = fn(x.clone())
                    err = (out.float() - ref.float()).abs().max().item()
                    y = x.clone()
                    us = timed(lambda: fn(y))
                    line.append(f"{name} {us:7.1f}us err {err:.2e}")
                except Exception as e:
                    line.append(f"{name} FAIL {str(e)[:60]}")
            if rank == 0:
                print(f"rows {r:4d} {n*2/1e6:5.2f} MB  " + "  ".join(line), flush=True)
        dist.destroy_process_group()
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--rows", default="6,24,48,96,192,384")
    a = ap.parse_args()
    mp.spawn(worker, args=(a, free_port()), nprocs=a.world)

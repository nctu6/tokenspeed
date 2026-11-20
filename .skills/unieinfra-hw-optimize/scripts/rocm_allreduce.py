"""4-GPU RCCL BF16 all_reduce latency/bandwidth; run with torchrun --nproc-per-node 4 under each runtime."""

import json
import os
import time

import torch
import torch.distributed as dist

r = int(os.environ["RANK"])
torch.cuda.set_device(r)
dist.init_process_group("nccl")
res = {}
for size in [16 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20]:
    x = torch.ones(size // 2, device="cuda", dtype=torch.bfloat16)
    for _ in range(20):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    s = time.perf_counter()
    for _ in range(100):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    us = (time.perf_counter() - s) / 100 * 1e6
    res[f"{size >> 10}KiB"] = {
        "us": round(us, 1),
        "busbw_GBps": round(2 * 3 / 4 * size / us / 1e3, 1),
    }
if r == 0:
    print(json.dumps(res, indent=1))
dist.destroy_process_group()

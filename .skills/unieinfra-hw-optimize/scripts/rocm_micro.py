"""Single-GPU ROCm runtime microbenchmarks (run under each runtime and diff the JSON): eager launch, HIP graph replay, BF16 GEMM (hipBLASLt), streaming bandwidth."""

import ctypes
import json
import sys
import time

import torch

torch.zeros(1, device="cuda")
# The HIP runtime torch actually loaded (loading another copy by name would mix two runtimes).
hip = next(l.split()[-1] for l in open("/proc/self/maps") if "libamdhip64" in l)
v = ctypes.c_int()
ctypes.CDLL(hip).hipRuntimeGetVersion(ctypes.byref(v))
out = {"hip_lib": hip, "hip_runtime": v.value, "torch": torch.__version__}


def t(fn, iters, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return round((time.perf_counter() - s) / iters * 1e6, 2)  # us


x = torch.zeros(1024, device="cuda")
out["eager_launch_us"] = t(lambda: x.add_(1), 20000)
out["eager50_sync_us"] = t(
    lambda: ([x.add_(1) for _ in range(50)], torch.cuda.synchronize()), 1000
)
g = torch.cuda.CUDAGraph()
x.add_(1)
torch.cuda.synchronize()
with torch.cuda.graph(g):
    for _ in range(50):
        x.add_(1)


def replay_sync():
    g.replay()
    torch.cuda.synchronize()


# Synchronized per replay: ROCm 7.2.4's runtime segfaults after ~2000 back-to-back replays of this graph.
out["graph50_replay_sync_us"] = t(replay_sync, 1000)
gemm = {}
for m, k, n in [
    (16, 7168, 2048),
    (64, 7168, 2048),
    (256, 7168, 2048),
    (4096, 7168, 2048),
    (64, 4096, 7168),
    (8192, 4096, 7168),
]:
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    us = t(lambda: a @ b, 200)
    gemm[f"{m}x{k}x{n}"] = {"us": us, "tflops": round(2 * m * k * n / us / 1e6, 1)}
out["gemm_bf16"] = gemm
y = torch.randn(64 << 20, device="cuda")
out["rw_256MiB_GBps"] = round(2 * y.numel() * 4 / t(lambda: y.mul_(1.0), 50) / 1e3, 1)
print(json.dumps(out))

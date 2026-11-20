import torch
import triton
import triton.language as tl


@triton.jit
def rd(x, o, N, BLOCK: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], tl.int32)
    base = pid * BLOCK * ITERS
    for i in range(ITERS):
        offs = base + i * BLOCK + tl.arange(0, BLOCK)
        acc += tl.load(x + offs, mask=offs < N, other=0)
    tl.store(o + pid, tl.sum(acc))


for gb in (0.5, 1, 2, 4):
    N = int(gb * 1024**3) // 4
    x = torch.randint(0, 3, (N,), dtype=torch.int32, device="cuda")
    for BLOCK, IT, nw in (
        (4096, 8, 8),
        (8192, 4, 8),
        (2048, 16, 4),
        (4096, 32, 8),
        (16384, 2, 16),
    ):
        grid = triton.cdiv(N, BLOCK * IT)
        o = torch.empty(grid, dtype=torch.int32, device="cuda")
        f = lambda: rd[(grid,)](x, o, N, BLOCK=BLOCK, ITERS=IT, num_warps=nw)
        t = triton.testing.do_bench(f, warmup=10, rep=100)
        print(f"{gb}GB B{BLOCK} it{IT} w{nw}: {N*4/t/1e9:.2f} TB/s", flush=True)

# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Benchmark of the SM120 tensor-core V4 selected attention.

Run on an SM120 GPU: ``python bench_dsv4_mma_sm120.py``. Rows are one layer's
window plus top-k selection (640 rows) or the window alone (128) over a BF16
workspace, with a per-head sink; ``portable`` is ``triton_dsv4_prefill``.
"""

import torch
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

DIM = 512
FAST, PORTABLE = "triton_dsv4_mma_prefill", "triton_dsv4_prefill"


def _graph_us(fn, repeat: int = 5) -> float:
    fn()
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(4):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / repeat * 1e3)
    return best


def _run(tokens: int, heads: int, width: int, rows: int) -> None:
    q = (torch.randn(tokens, heads, DIM, device="cuda") * 0.8).bfloat16()
    kv = (torch.randn(rows, 1, DIM, device="cuda") * 0.8).bfloat16()
    indices = torch.randint(0, rows, (tokens, width), device="cuda", dtype=torch.int32)
    lens = torch.full((tokens,), width, dtype=torch.int32, device="cuda")
    sink = torch.randn(heads, device="cuda")

    def call(name):
        return lambda: dsv4_prefill(
            q, kv, indices, lens, sink, DIM**-0.5, override=name
        )

    fast = _graph_us(call(FAST))
    tflops = 4.0 * tokens * heads * width * DIM / fast / 1e6
    portable = ""
    if tokens * heads <= 64 * 2048:
        slow = _graph_us(call(PORTABLE), repeat=1)
        portable = f"portable {slow:9.0f} us ({slow / fast:4.0f}x)"
    print(
        f"tokens={tokens:5d} heads={heads:3d} width={width:4d} | {fast:8.1f} us "
        f"{tflops:6.1f} TFLOP/s | {portable}",
        flush=True,
    )


def main() -> None:
    for heads in (16, 64, 128):
        for tokens, width, rows in (
            (1, 640, 4096),
            (16, 640, 20000),
            (128, 640, 100000),
            (512, 640, 4096),
            (2048, 640, 8192),
            (2048, 128, 4096),
        ):
            _run(tokens, heads, width, rows)


if __name__ == "__main__":
    main()

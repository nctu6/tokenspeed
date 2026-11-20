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
"""Benchmark of the SM120 tensor-core MLA decode against the portable kernel.

Run on an SM120 GPU: ``python bench_mla_decode_sm120.py``. The roofline is the
latent-cache bytes of the batch at the device copy bandwidth; ``TFLOP/s``
counts the QK and PV products (2 * rows * length * (576 + 512)).
"""

import torch
from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache

RANK, ROPE = 512, 64
DIM = RANK + ROPE
COPY_GBPS = 1460.0  # measured device-to-device copy of a 1 GB tensor
FAST, PORTABLE = (
    "triton_mla_mma_decode_with_kvcache",
    "triton_mla_decode_with_kvcache",
)


def _graph_us(fn, repeat: int = 10) -> float:
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


def _run(dtype, heads, batch, length, q_len=1, *, with_portable=True) -> None:
    pages = -(-length // 64)
    table = torch.randperm(batch * pages + 3, device="cuda")[: batch * pages]
    table = table.reshape(batch, pages).to(torch.int32)
    kv = (torch.randn(batch * pages + 3, 64, 1, DIM, device="cuda") * 0.8).to(dtype)
    q = (torch.randn(batch, q_len, heads, DIM, device="cuda") * 0.8).to(dtype)
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    def call(name):
        return lambda: mla_decode_with_kvcache(
            q, kv, table, lengths, length, 128, RANK, ROPE, 0.1352, override=name
        )

    fast = _graph_us(call(FAST))
    roof = batch * length * DIM * q.element_size() / (COPY_GBPS * 1e3)
    tflops = 2.0 * batch * heads * q_len * length * (DIM + RANK) / fast / 1e6
    portable = ""
    if with_portable and q_len == 1:
        slow = _graph_us(call(PORTABLE), repeat=1)
        portable = f"portable {slow:9.0f} us ({slow / fast:5.0f}x)"
    print(
        f"{str(dtype)[6:]:>13} H={heads:3d} B={batch:3d} L={length:6d} Q={q_len} | "
        f"{fast:8.1f} us  roofline {roof:7.1f} us ({fast / roof:5.2f}x) "
        f"{tflops:6.1f} TFLOP/s | {portable}",
        flush=True,
    )


def main() -> None:
    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        for heads in (16, 128):
            for batch, length in ((1, 4096), (1, 32768), (8, 8192), (32, 8192)):
                _run(dtype, heads, batch, length)
        for heads, batch, length, q_len in ((16, 8, 8192, 4), (128, 8, 8192, 4)):
            _run(dtype, heads, batch, length, q_len)


if __name__ == "__main__":
    main()

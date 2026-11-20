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
"""Benchmark of the SM120 Hadamard leaf against the portable Triton leaf.

Run on an SM120 GPU: ``python bench_hadamard_sm120.py``. Rows are
``tokens x 32 heads x 128`` BF16 (the GLM indexer query); the roofline is a
device-to-device copy of the same tensor (one read plus one write).
"""

import torch
from tokenspeed_kernel.ops.transform import triton as transform


def _graph_us(fn, repeat: int = 30) -> float:
    fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeat):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / repeat * 1e3)
    return best


def main() -> None:
    scale = 128**-0.5
    print("tokens  copy_us  portable_us  mma_us  mma/copy  speedup")
    for tokens in (1, 8, 64, 512, 4096, 16384, 65536):
        x = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        copy = _graph_us(lambda: y.copy_(x))
        old = _graph_us(lambda: transform.triton_hadamard_transform_128(x, scale=scale))
        new = _graph_us(
            lambda: transform.triton_hadamard_transform_128_mma(x, scale=scale)
        )
        print(
            f"{tokens:6d}  {copy:7.1f}  {old:11.1f}  {new:6.1f}  "
            f"{new / copy:8.2f}  {old / new:6.1f}x"
        )


if __name__ == "__main__":
    main()

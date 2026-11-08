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
"""Benchmark of the SM120 tensor-core DSA sparse attention cores.

Run on an SM120 GPU: ``python bench_dsa_mma_sm120.py``. The 64 buckets are
16/32/64/128 heads x decode batch 1/8/32/128 (16K history) and prefill chunks
128/512/2048/8192 (32K history), top-k 2048, for E4M3 and BF16 caches of 512 +
64 channels; ``triton`` is ``triton_dsa_mma_*`` and ``cute_dsl`` is
``cute_dsl_dsa_mma_*``. Times are CUDA-graph microseconds (best of 5).
"""

import math

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.registry import KernelRegistry

PAGE, RANK, ROPE, TOPK = 64, 512, 64, 2048
DIM = RANK + ROPE


def _graph_us(fn, repeat: int = 10) -> float:
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
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / repeat * 1e3)
    return best


def main() -> None:
    solutions = ["triton"]
    if KernelRegistry.get().get_by_name("cute_dsl_dsa_mma_decode") is not None:
        solutions.append("cute_dsl")
    cache = torch.randn(32768, DIM, device="cuda") * 0.7
    ratios = []
    print("bucket", *solutions, sep="\t")
    for dtype in (torch.float8_e4m3fn, torch.bfloat16):
        kv = cache.to(dtype).view(-1, PAGE, 1, DIM)
        for heads in (16, 32, 64, 128):
            for mode, tokens in [("d", n) for n in (1, 8, 32, 128)] + [
                ("p", n) for n in (128, 512, 2048, 8192)
            ]:
                history = 16384 if mode == "d" else 32768
                q = (torch.randn(tokens, heads, DIM, device="cuda") * 1.5).to(dtype)
                slots = torch.randint(
                    0, history, (tokens, TOPK), device="cuda", dtype=torch.int32
                )
                lens = torch.full((tokens,), TOPK, dtype=torch.int32, device="cuda")
                call = dsa_decode if mode == "d" else dsa_prefill
                extra = {"q_len_per_req": 1} if mode == "d" else {}
                times = []
                for solution in solutions:
                    times.append(
                        _graph_us(
                            lambda: call(
                                q=q,
                                kv_cache=kv,
                                sparse_kv_cache=None,
                                topk_slots=slots,
                                topk_lens=lens,
                                max_seqlen_k=history,
                                qk_nope_head_dim=128,
                                kv_lora_rank=RANK,
                                qk_rope_head_dim=ROPE,
                                softmax_scale=0.1352,
                                page_size=PAGE,
                                return_lse=True,
                                solution=solution,
                                slot_order="selection",
                                **extra,
                            )
                        )
                    )
                if len(times) == 2:
                    ratios.append(times[0] / times[1])
                name = f"{str(dtype).split('.')[-1]}_h{heads}_{mode}{tokens}"
                print(name, *(f"{t:.1f}" for t in times), sep="\t", flush=True)
    if ratios:
        geomean = math.exp(sum(map(math.log, ratios)) / len(ratios))
        print(
            f"triton / cute_dsl: geomean {geomean:.2f}x, "
            f"range {min(ratios):.2f}-{max(ratios):.2f}x"
        )


if __name__ == "__main__":
    main()

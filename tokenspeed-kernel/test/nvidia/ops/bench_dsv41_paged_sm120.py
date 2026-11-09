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
"""Benchmark of the SM120 selected attention that reads packed cache rows itself.

Run on an SM120 GPU: ``python bench_dsv41_paged_sm120.py``. A decode step of
``tokens`` queries with a 128-row window (``swa`` rows) and 512 compressed rows
(``global`` rows) per query, per-head sink, CUDA graph with cold caches.

* ``v41``: ``triton_dsv41_selected_attention`` (gather every row into a BF16
  workspace, then the dense kernel) against ``triton_dsv41_mma_selected_attention``.
* ``v4``: ``dsv4_decode`` on V4 FP8 page-planar caches, workspace route
  (``triton_dsv4_decode``, which reaches the CuTe DSL dense kernel above 16
  heads) against ``triton_dsv4_mma_decode``.
"""

import sys

import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_decode
from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41_triton

ROW_BYTES = {"swa": 528, "global": 288}


def _graph_us(fn, repeat: int = 4) -> float:
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
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / repeat)
    return best


def _v41_cache(rows, fmt, generator):
    pages = (rows + 63) // 64
    cache = torch.zeros((pages, 64, ROW_BYTES[fmt]), dtype=torch.uint8, device="cuda")
    values = torch.randn((pages * 64, 512), device="cuda", generator=generator)
    dsv41.cache_scatter(
        values.to(torch.bfloat16), cache, torch.arange(pages * 64, device="cuda"), fmt
    )
    return cache


def _v4_cache(pages, generator, page_size=64):
    nope = torch.randn(pages, page_size, 448, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    rope = torch.randn(
        pages, page_size, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    cache = torch.full((pages, page_size * 584), 127, dtype=torch.uint8, device="cuda")
    payload = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    cache[:, : page_size * 576] = payload.reshape(pages, -1)
    return cache


def _selection(tokens, width, rows, generator):
    slots = torch.randint(0, rows, (tokens, width), device="cuda", generator=generator)
    lens = torch.randint(
        width // 2, width + 1, (tokens,), device="cuda", generator=generator
    )
    return slots.to(torch.int32), lens.to(torch.int32)


def main(kind: str = "v41") -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    print(f"{kind}: us per call (CUDA graph); workspace route / direct reader")
    ratios = []
    for heads in (16, 64, 128):
        for tokens in (1, 4, 16, 32, 64, 128, 256):
            swa_rows, global_rows = tokens * 192, tokens * 2048
            q = torch.randn(
                tokens,
                heads,
                512,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            sink = torch.randn(heads, device="cuda", generator=generator)
            swa_slots, swa_lens = _selection(tokens, 128, swa_rows, generator)
            glob_slots, glob_lens = _selection(tokens, 512, global_rows, generator)
            out = torch.empty_like(q)
            if kind == "v41":
                swa = _v41_cache(swa_rows, "swa", generator)
                glob = _v41_cache(global_rows, "global", generator)
                runs = {
                    "workspace": dsv41_triton.selected_attention,
                    "direct": dsv41_triton.mma_selected_attention,
                }
                calls = {
                    name: (
                        lambda fn=fn: fn(
                            q,
                            swa,
                            swa_slots,
                            swa_lens,
                            glob,
                            glob_slots,
                            glob_lens,
                            sink,
                            512**-0.5,
                            out,
                            256,
                            None,
                            None,
                            None,
                        )
                    )
                    for name, fn in runs.items()
                }
            else:
                swa = _v4_cache(swa_rows // 64, generator)
                glob = _v4_cache(global_rows // 64, generator)
                calls = {
                    name: (
                        lambda override=override: dsv4_decode(
                            q,
                            swa,
                            swa_slots,
                            swa_lens,
                            64,
                            sink,
                            512**-0.5,
                            extra_kv_cache=glob,
                            extra_slots=glob_slots,
                            extra_lens=glob_lens,
                            extra_page_size=64,
                            out=out,
                            override=override,
                        )
                    )
                    for name, override in (
                        ("workspace", "triton_dsv4_decode"),
                        ("direct", "triton_dsv4_mma_decode"),
                    )
                }
            times = {name: _graph_us(call) for name, call in calls.items()}
            ratios.append(times["workspace"] / times["direct"])
            print(
                f"heads {heads:3} tokens {tokens:3}: workspace {times['workspace']:7.1f} "
                f"direct {times['direct']:7.1f} ratio {ratios[-1]:.2f}",
                flush=True,
            )
    geomean = torch.tensor(ratios).log().mean().exp().item()
    print(f"geomean {geomean:.2f}, worst {min(ratios):.2f}, best {max(ratios):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])

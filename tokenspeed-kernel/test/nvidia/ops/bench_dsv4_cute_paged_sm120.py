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
"""Benchmark of the CuTe DSL selected attention that dequantizes packed rows.

Run on an SM120 GPU: ``python bench_dsv4_cute_paged_sm120.py v4|v41 [heads]``
(heads is a comma-separated list, default ``16,32,64,128``). A decode step of
``tokens`` queries with a 128-row window (``swa`` rows) and 0 / 512 / 1024
compressed rows (``global`` rows) per query, per-head sink, CUDA graph with
cold caches, best of 5. Three routes per point:

* ``workspace``: gather and dequantize every row into a BF16 workspace, then
  ``cute_dsl_dsv4_mma_prefill`` (``triton_dsv4_decode`` for V4; the V4.1 glue of
  ``triton_dsv41_selected_attention`` with the CuTe core).
* ``triton``: the Triton direct reader (``triton_dsv4_mma_decode`` /
  ``triton_dsv41_mma_selected_attention``).
* ``cute``: ``paged_attention`` of ``dsv4/_cute_dsl/sparse_mma.py`` on the plan
  ``plan_paged`` chooses.
"""

import sys

import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_decode, dsv4_prefill
from tokenspeed_kernel.ops.attention.dsv4._cute_dsl import sparse_mma as cute_mma
from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41_triton
from tokenspeed_kernel.platform import current_platform

ROW_BYTES = {"swa": 528, "global": 288}
CORE = "cute_dsl_dsv4_mma_prefill"
SCALE = 512**-0.5


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


def _v41_workspace_route(q, segments, sink, out):
    """The portable V4.1 gather, then the CuTe DSL dense core."""
    tokens = q.shape[0]
    rows, live = [], []
    for cache, slots, lens, fmt in segments:
        valid = (
            (torch.arange(slots.shape[1], device="cuda") < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
        masked = slots.masked_fill(~valid, -1)
        rows.append(dsv41_triton.cache_gather(cache, masked, fmt, None))
        live.append(valid)
    kv, valid = torch.cat(rows, 1), torch.cat(live, 1)
    width = kv.shape[1]
    indices = (
        torch.arange(tokens * width, dtype=torch.int32, device="cuda")
        .view(tokens, width)
        .masked_fill(~valid, -1)
    )
    lens = torch.full((tokens,), width, dtype=torch.int32, device="cuda")
    return dsv4_prefill(
        q=q, kv=kv, indices=indices, lens=lens, attn_sink=sink,
        softmax_scale=SCALE, out=out, override=CORE,
    )  # fmt: skip


def bench(kind, heads, tokens, extra_width, generator):
    """``(workspace, triton, cute)`` microseconds and the cute plan."""
    swa_rows, global_rows = tokens * 192, tokens * 2048
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, swa_rows, generator)
    extra = extra_width > 0
    if extra:
        glob_slots, glob_lens = _selection(tokens, extra_width, global_rows, generator)
    out = torch.empty_like(q)
    if kind == "v41":
        swa = _v41_cache(swa_rows, "swa", generator)
        glob = _v41_cache(global_rows, "global", generator) if extra else None
        formats = ("swa", "global")
        segments = [(swa, swa_slots, swa_lens, "swa")]
        if extra:
            segments.append((glob, glob_slots, glob_lens, "global"))
        args = (
            q, swa, swa_slots, swa_lens, glob, glob_slots if extra else None,
            glob_lens if extra else None, sink, SCALE, out, 256, None, None, None,
        )  # fmt: skip
        calls = {
            "workspace": lambda: _v41_workspace_route(q, segments, sink, out),
            "triton": lambda: dsv41_triton.mma_selected_attention(*args),
        }
    else:
        swa = _v4_cache(swa_rows // 64 + 1, generator)
        glob = _v4_cache(global_rows // 64 + 1, generator) if extra else None
        formats = ("swa_v4", "global_v4")
        kwargs = (
            dict(
                extra_kv_cache=glob,
                extra_slots=glob_slots,
                extra_lens=glob_lens,
                extra_page_size=64,
            )
            if extra
            else {}
        )
        calls = {
            name: (
                lambda override=override: dsv4_decode(
                    q, swa, swa_slots, swa_lens, 64, sink, SCALE, out=out,
                    override=override, **kwargs,
                )  # fmt: skip
            )
            for name, override in (
                ("workspace", "triton_dsv4_decode"),
                ("triton", "triton_dsv4_mma_decode"),
            )
        }
    first = (swa, formats[0], swa_slots, swa_lens, 64)
    second = (glob, formats[1], glob_slots, glob_lens, 64) if extra else None
    block_h, cwarps, splits, full = cute_mma.plan_paged(
        tokens, heads, current_platform().sm_count, first, second
    )
    out_cute = torch.empty_like(q)

    def cute():
        cute_mma.paged_attention(
            q, first, second, sink, out_cute, None, softmax_scale=SCALE,
            block_h=block_h, cwarps=cwarps, num_splits=splits, full_tokens=full,
        )  # fmt: skip

    calls["cute"] = cute
    times = {name: _graph_us(call) for name, call in calls.items()}
    return times, (block_h, cwarps, splits, full)


def main(kind: str = "v41", heads_list: str = "16,32,64,128") -> None:
    generator = torch.Generator(device="cuda").manual_seed(0)
    print(f"{kind}: us per call (CUDA graph); workspace / triton direct / cute packed")
    ratios = {"workspace": [], "triton": []}
    for heads in [int(h) for h in heads_list.split(",")]:
        for extra_width in (0, 512, 1024):
            for tokens in (1, 4, 16, 64, 256, 1024, 2048):
                times, plan = bench(kind, heads, tokens, extra_width, generator)
                for route in ratios:
                    ratios[route].append(times[route] / times["cute"])
                print(
                    f"heads {heads:3} width {128 + extra_width:4} tokens {tokens:4}: "
                    f"workspace {times['workspace']:8.1f} triton {times['triton']:8.1f} "
                    f"cute {times['cute']:8.1f}  vs workspace "
                    f"{ratios['workspace'][-1]:.2f}x vs triton {ratios['triton'][-1]:.2f}x "
                    f"plan {plan}",
                    flush=True,
                )
    for route, values in ratios.items():
        ratio = torch.tensor(values)
        print(
            f"cute vs {route}: geomean {ratio.log().mean().exp().item():.2f}x, "
            f"worst {ratio.min().item():.2f}x, best {ratio.max().item():.2f}x"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])

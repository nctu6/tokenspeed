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
"""CuTe DSL selected attention that reads the packed cache rows itself, on SM120.

``cute_dsl_dsv4_mma_decode`` and ``cute_dsl_dsv41_mma_selected_attention``
dequantize the selected FP8 / FP4 page-planar rows while they gather them into
the BF16 ring of the CuTe core, instead of writing a BF16 workspace for
``cute_dsl_dsv4_mma_prefill`` to read. The rows they build are exact in BF16
and the loop is the same, so they must return the workspace route's result bit
for bit.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4 import dsv4_decode, dsv4_prefill
from tokenspeed_kernel.ops.attention.dsv4 import triton as dsv4_triton
from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41_triton
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry

V4_LEAF = "cute_dsl_dsv4_mma_decode"
V41_LEAF = "cute_dsl_dsv41_mma_selected_attention"
CORE = "cute_dsl_dsv4_mma_prefill"
HAS_CUTE = (
    KernelRegistry.get().get_by_name(V4_LEAF) is not None
    and KernelRegistry.get().get_by_name(V41_LEAF) is not None
)
pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and Platform.get().is_nvidia
        and Platform.get().arch_version.major == 12
        and HAS_CUTE
    ),
    reason="SM120 with the CuTe DSL only",
)

if HAS_CUTE:
    from tokenspeed_kernel.ops.attention.dsv4._cute_dsl import sparse_mma as cute_mma
    from tokenspeed_kernel.ops.attention.dsv41 import cute_dsl_attention

SCALE = 512**-0.5
_ROW_BYTES = {"swa": 528, "global": 288}


def _v41_cache(rows, fmt, generator, *, page_pad=0):
    """A packed ``[pages, 64, row_bytes]`` cache of random rows (cache codec).

    ``page_pad`` bytes of slack after each page make the field a strided view.
    """
    pages = (rows + 63) // 64
    width = _ROW_BYTES[fmt]
    storage = torch.zeros(
        (pages, 64 * width + page_pad), dtype=torch.uint8, device="cuda"
    )
    cache = storage[:, : 64 * width].view(pages, 64, width)
    values = torch.randn((pages * 64, 512), device="cuda", generator=generator).to(
        torch.bfloat16
    )
    values[::7] *= 40
    values[3::11] *= 1e-3
    dsv41.cache_scatter(values, cache, torch.arange(pages * 64, device="cuda"), fmt)
    return cache


def _v4_cache(pages, page_size, generator):
    nope = torch.randn(pages, page_size, 448, device="cuda", generator=generator).to(
        torch.float8_e4m3fn
    )
    rope = torch.randn(
        pages, page_size, 64, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    scales = torch.randint(
        100, 140, (pages, page_size, 8), device="cuda", generator=generator
    ).to(torch.uint8)
    cache = torch.empty((pages, page_size * 584), device="cuda", dtype=torch.uint8)
    payload = torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1)
    cache[:, : page_size * 576] = payload.reshape(pages, -1)
    cache[:, page_size * 576 :] = scales.reshape(pages, -1)
    return cache


def _selection(tokens, width, rows, generator, dtype=torch.int32):
    slots = torch.randint(
        0, rows, (tokens, width), device="cuda", generator=generator
    ).to(dtype)
    lens = torch.randint(
        0, width + 1, (tokens,), device="cuda", generator=generator
    ).to(dtype)
    lens[0] = width
    slots[torch.rand(tokens, width, device="cuda", generator=generator) < 0.1] = -1
    slots[-1, : width // 2] = rows + 5  # past the cache
    return slots, lens


def _v41_workspace_route(
    q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink
):
    """The portable route's gather, then the CuTe DSL dense core."""
    parts = [(swa, swa_slots, swa_lens, "swa")]
    if glob is not None:
        parts.append((glob, glob_slots, glob_lens, "global"))
    tokens = q.shape[0]
    rows, live = [], []
    for cache, slots, lens, fmt in parts:
        valid = (
            (torch.arange(slots.shape[1], device="cuda") < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
        rows.append(
            dsv41_triton.cache_gather(cache, slots.masked_fill(~valid, -1), fmt, None)
        )
        live.append(valid)
    kv = torch.cat(rows, 1)
    valid = torch.cat(live, 1)
    width = kv.shape[1]
    indices = (
        torch.arange(tokens * width, dtype=torch.int32, device="cuda")
        .view(tokens, width)
        .masked_fill(~valid, -1)
    )
    lens = torch.full((tokens,), width, dtype=torch.int32, device="cuda")
    return dsv4_prefill(
        q=q, kv=kv, indices=indices, lens=lens, attn_sink=sink,
        softmax_scale=SCALE, override=CORE,
    )  # fmt: skip


def _v41_leaf(q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink, lse=None):
    return cute_dsl_attention.cute_dsl_dsv41_mma_selected_attention(
        q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink, SCALE,
        None, 256, None, None, None, lse=lse,
    )  # fmt: skip


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("heads", [16, 40, 128])
@pytest.mark.parametrize(
    ("tokens", "swa_width", "extra_width"),
    [(1, 128, 512), (5, 128, 200), (37, 96, 64), (9, 32, 1024)],
)
def test_v41_leaf_equals_the_workspace_route_bit_for_bit(
    heads, tokens, swa_width, extra_width, index_dtype
):
    generator = torch.Generator(device="cuda").manual_seed(heads * 100 + tokens)
    swa_rows, global_rows = 500, 900
    swa = _v41_cache(swa_rows, "swa", generator)
    glob = _v41_cache(global_rows, "global", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(
        tokens, swa_width, swa_rows, generator, index_dtype
    )
    glob_slots, glob_lens = _selection(
        tokens, extra_width, global_rows, generator, index_dtype
    )
    args = (q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink)
    expected = _v41_workspace_route(*args)
    actual = _v41_leaf(*args)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual) > 0


@pytest.mark.parametrize("heads", [16, 64])
def test_v41_window_only_layer_equals_the_workspace_route(heads):
    """A layer without a compressed cache reads its FP8 window the same way."""
    generator = torch.Generator(device="cuda").manual_seed(heads)
    tokens = 6
    swa = _v41_cache(300, "swa", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 300, generator)
    args = (q, swa, swa_slots, swa_lens, None, None, None, sink)
    assert torch.equal(_v41_leaf(*args), _v41_workspace_route(*args))


def test_v41_leaf_returns_the_partial_lse_without_a_sink():
    generator = torch.Generator(device="cuda").manual_seed(5)
    heads, tokens = 64, 9
    swa = _v41_cache(300, "swa", generator)
    glob = _v41_cache(700, "global", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    swa_slots, swa_lens = _selection(tokens, 128, 300, generator)
    glob_slots, glob_lens = _selection(tokens, 256, 700, generator)
    args = (q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, None)
    lse = torch.empty(tokens, heads, device="cuda")
    out = _v41_leaf(*args, lse=lse)
    # The dense core returns the same partial and LSE over the gathered rows.
    parts = [(swa, swa_slots, swa_lens, "swa"), (glob, glob_slots, glob_lens, "global")]
    rows, live = [], []
    for cache, slots, lens, fmt in parts:
        valid = (
            (torch.arange(slots.shape[1], device="cuda") < lens[:, None])
            & (slots >= 0)
            & (slots < cache.shape[0] * 64)
        )
        rows.append(
            dsv41_triton.cache_gather(cache, slots.masked_fill(~valid, -1), fmt, None)
        )
        live.append(valid)
    kv, valid = torch.cat(rows, 1), torch.cat(live, 1)
    width = kv.shape[1]
    indices = (
        torch.arange(tokens * width, dtype=torch.int32, device="cuda")
        .view(tokens, width)
        .masked_fill(~valid, -1)
    )
    expected_lse = torch.empty(tokens, heads, device="cuda")
    expected = dsv4_prefill(
        q, kv, indices, torch.full((tokens,), width, dtype=torch.int32, device="cuda"),
        None, SCALE, lse=expected_lse, override=CORE,
    )  # fmt: skip
    assert torch.equal(out, expected)
    assert torch.equal(lse, expected_lse)


def test_v41_leaf_reads_strided_cache_views():
    generator = torch.Generator(device="cuda").manual_seed(11)
    heads, tokens = 64, 6
    swa = _v41_cache(256, "swa", generator, page_pad=96)
    glob = _v41_cache(512, "global", generator, page_pad=32)
    assert swa.stride(0) != swa.stride(1) * swa.shape[1]
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 256, generator)
    glob_slots, glob_lens = _selection(tokens, 512, 512, generator)
    args = (q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink)
    assert torch.equal(_v41_leaf(*args), _v41_workspace_route(*args))


def test_v41_leaf_falls_back_on_a_misaligned_cache():
    """A cache the vector loads cannot address runs the Triton direct reader."""
    generator = torch.Generator(device="cuda").manual_seed(13)
    heads, tokens = 32, 4
    swa = _v41_cache(256, "swa", generator)
    glob = _v41_cache(512, "global", generator)
    padded = torch.zeros(glob.numel() + 16, dtype=torch.uint8, device="cuda")
    padded[8 : 8 + glob.numel()] = glob.reshape(-1)
    shifted = padded[8 : 8 + glob.numel()].view(glob.shape)
    assert shifted.data_ptr() % 16 == 8
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 256, generator)
    glob_slots, glob_lens = _selection(tokens, 512, 512, generator)
    args = (q, swa, swa_slots, swa_lens, shifted, glob_slots, glob_lens, sink)
    triton = dsv41_triton.mma_selected_attention(
        *args, SCALE, None, 256, None, None, None
    )
    assert torch.equal(_v41_leaf(*args), triton)


def _v4_workspace_route(q, swa, swa_slots, swa_lens, extra, page_size, sink):
    tokens = q.shape[0]
    width = swa_slots.shape[1] + (0 if extra is None else extra[1].shape[1])
    workspace = torch.empty(tokens, width, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.empty(tokens, width, dtype=torch.int32, device="cuda")
    lens = torch.empty(tokens, dtype=torch.int32, device="cuda")
    dequantize = dsv4_triton._dsv4_dequantize_selected_cache_segment
    dequantize(swa, swa_slots, swa_lens, page_size, workspace, indices, lens, 0, width)
    if extra is not None:
        dequantize(
            extra[0], extra[1], extra[2], page_size, workspace, indices, lens,
            swa_slots.shape[1], width,
        )  # fmt: skip
    return dsv4_prefill(
        q=q, kv=workspace, indices=indices, lens=lens, attn_sink=sink,
        softmax_scale=SCALE, override=CORE,
    )  # fmt: skip


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("heads", [8, 16, 64, 128])
@pytest.mark.parametrize("extra", [False, True])
def test_v4_decode_leaf_equals_the_workspace_route(heads, extra, index_dtype):
    generator = torch.Generator(device="cuda").manual_seed(heads)
    tokens, page_size = 11, 64
    swa = _v4_cache(4, page_size, generator)
    compressed = _v4_cache(3, page_size, generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 4 * page_size, generator, index_dtype)
    extra_slots, extra_lens = _selection(
        tokens, 96, 3 * page_size, generator, index_dtype
    )
    kwargs = dict(
        extra_kv_cache=compressed if extra else None,
        extra_slots=extra_slots if extra else None,
        extra_lens=extra_lens if extra else None,
        extra_page_size=page_size if extra else None,
    )
    actual = dsv4_decode(
        q, swa, swa_slots, swa_lens, page_size, sink, SCALE, override=V4_LEAF, **kwargs
    )
    expected = _v4_workspace_route(
        q, swa, swa_slots, swa_lens,
        (compressed, extra_slots, extra_lens) if extra else None, page_size, sink,
    )  # fmt: skip
    assert torch.equal(actual, expected)


def test_v4_decode_leaf_is_what_the_registry_picks():
    generator = torch.Generator(device="cuda").manual_seed(2)
    swa = _v4_cache(2, 64, generator)
    for heads in (8, 16, 128):
        q = torch.randn(40, heads, 512, device="cuda", dtype=torch.bfloat16)
        slots, lens = _selection(40, 128, 128, generator)
        sink = torch.randn(heads, device="cuda")
        picked = dsv4_decode(q, swa, slots, lens, 64, sink, SCALE)
        forced = dsv4_decode(q, swa, slots, lens, 64, sink, SCALE, override=V4_LEAF)
        assert torch.equal(picked, forced)


def test_v4_decode_leaf_falls_back_on_a_page_size_it_cannot_shift():
    generator = torch.Generator(device="cuda").manual_seed(3)
    tokens, page_size, heads = 5, 48, 32  # not a power of two
    swa = _v4_cache(4, page_size, generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    slots, lens = _selection(tokens, 64, 4 * page_size, generator)
    actual = dsv4_decode(q, swa, slots, lens, page_size, sink, SCALE, override=V4_LEAF)
    expected = dsv4_decode(
        q, swa, slots, lens, page_size, sink, SCALE, override="triton_dsv4_decode"
    )
    assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# The dequantized values themselves
# ---------------------------------------------------------------------------

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _valid_e4m3_bytes():
    """Every E4M3 byte but the two NaN codes."""
    codes = torch.arange(256, dtype=torch.int32)
    return codes[(codes & 0x7F) != 0x7F].to(torch.uint8)


def _ue8m0(code):
    """The float32 scale of a UE8M0 byte (0 is the subnormal ``2**-127``)."""
    bits = code.to(torch.int32) << 23
    scale = bits.view(torch.float32)
    return torch.where(code == 0, torch.tensor(2.0**-127), scale)


def _read_back(q_heads, swa_segment, extra_segment, tokens):
    """Attend one live key per token with a zero query and no sink weight.

    A single live key has probability 1, so the output row is that key's
    BF16 row: a direct readout of the dequantization.
    """
    q = torch.zeros(tokens, q_heads, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.full((q_heads,), -1e30, device="cuda")
    out = torch.empty_like(q)
    width = swa_segment[2].shape[1] + (
        0 if extra_segment is None else extra_segment[2].shape[1]
    )
    plan = cute_mma.plan_launch(tokens, q_heads, Platform.get().sm_count, width)
    assert cute_mma.supports_paged(q, out, swa_segment, extra_segment)
    cute_mma.paged_attention(
        q, swa_segment, extra_segment, sink, out, None, softmax_scale=SCALE,
        block_h=plan[0], cwarps=plan[1], num_splits=plan[2], full_tokens=plan[3],
    )  # fmt: skip
    return out[:, 0]


def _one_live(tokens, width, rows):
    """Column 0 of token ``t`` names row ``t``; the other columns are dead."""
    slots = torch.full((tokens, width), -1, dtype=torch.int32, device="cuda")
    slots[:, 0] = torch.arange(tokens, device="cuda") % rows
    return slots


def test_fp8_g32_rows_dequantize_exactly():
    rows = 128
    data = torch.arange(rows * 512, dtype=torch.int32) % 256
    data[(data & 0x7F) == 0x7F] = 0x7E
    data = data.to(torch.uint8).view(rows, 512)
    # Every UE8M0 scale whose largest value still fits BF16, subnormal one included.
    scales = (torch.arange(rows * 16) % 247).to(torch.uint8).view(rows, 16)
    cache = torch.zeros(rows // 64, 64, 528, dtype=torch.uint8)
    flat = cache.view(rows // 64, 64 * 528)
    flat[:, : 64 * 512] = data.view(rows // 64, 64 * 512)
    flat[:, 64 * 512 :] = scales.view(rows // 64, 64 * 16)
    expected = (
        data.view(torch.float8_e4m3fn).float()
        * _ue8m0(scales).repeat_interleave(32, dim=1)
    ).to(torch.bfloat16)
    lens = torch.ones(rows, dtype=torch.int32, device="cuda")
    out = _read_back(
        16, (cache.cuda(), "swa", _one_live(rows, 32, rows), lens, 64), None, rows
    )
    assert torch.equal(out, expected.cuda())


def test_fp8_v4_rows_dequantize_exactly():
    rows, page = 128, 64
    data = torch.arange(rows * 448, dtype=torch.int32) % 256
    data[(data & 0x7F) == 0x7F] = 0x7E
    data = data.to(torch.uint8).view(rows, 448)
    rope = torch.randn(rows, 64).to(torch.bfloat16)
    scales = torch.zeros(rows, 8, dtype=torch.uint8)
    scales[:, :7] = (1 + torch.arange(rows * 7) % 246).to(torch.uint8).view(rows, 7)
    cache = torch.zeros(rows // page, page * 584, dtype=torch.uint8)
    payload = torch.cat((data, rope.view(torch.uint8).view(rows, 128)), dim=1)
    cache[:, : page * 576] = payload.view(rows // page, page * 576)
    cache[:, page * 576 :] = scales.view(rows // page, page * 8)
    expected = torch.cat(
        (
            (
                data.view(torch.float8_e4m3fn).float()
                * _ue8m0(scales[:, :7]).repeat_interleave(64, dim=1)
            ).to(torch.bfloat16),
            rope,
        ),
        dim=1,
    )
    lens = torch.ones(rows, dtype=torch.int32, device="cuda")
    out = _read_back(
        16, (cache.cuda(), "swa_v4", _one_live(rows, 32, rows), lens, page), None, rows
    )
    assert torch.equal(out, expected.cuda())


def test_fp4_g16_rows_dequantize_exactly():
    rows = 128
    nibbles = (torch.arange(rows * 256, dtype=torch.int32) % 256).to(torch.uint8)
    nibbles = nibbles.view(rows, 256)
    valid = _valid_e4m3_bytes()
    scales = valid[torch.arange(rows * 32) % valid.numel()].view(rows, 32)
    cache = torch.zeros(rows // 64, 64, 288, dtype=torch.uint8)
    flat = cache.view(rows // 64, 64 * 288)
    flat[:, : 64 * 256] = nibbles.view(rows // 64, 64 * 256)
    flat[:, 64 * 256 :] = scales.view(rows // 64, 64 * 32)
    low, high = (nibbles & 15).long(), (nibbles >> 4).long()
    codes = torch.stack((low, high), dim=-1).reshape(rows, 512)
    magnitude = _E2M1[codes & 7]
    values = torch.where(codes >= 8, -magnitude, magnitude)
    expected = (
        values * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    ).to(torch.bfloat16)
    # The window segment holds one dead row; the live key is in the FP4 cache.
    window = torch.zeros(1, 64, 528, dtype=torch.uint8, device="cuda")
    dead = torch.full((rows, 32), -1, dtype=torch.int32, device="cuda")
    lens = torch.ones(rows, dtype=torch.int32, device="cuda")
    out = _read_back(
        16,
        (window, "swa", dead, torch.zeros_like(lens), 64),
        (cache.cuda(), "global", _one_live(rows, 32, rows), lens, 64),
        rows,
    )
    assert torch.equal(out, expected.cuda())


# ---------------------------------------------------------------------------
# Batch invariance and compilation
# ---------------------------------------------------------------------------


def _paged_rows(q, swa, extra, sink, plan, with_lse=False):
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], device="cuda") if with_lse else None
    cute_mma.paged_attention(
        q, swa, extra, None if with_lse else sink, out, lse, softmax_scale=SCALE,
        block_h=plan[0], cwarps=plan[1], num_splits=plan[2], full_tokens=plan[3],
    )  # fmt: skip
    return out, lse


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_rows_do_not_depend_on_the_launch_shape(index_dtype):
    """Rows and LSE are bitwise the same for every head tile, warp layout,
    split count and stream-K mix of whole and split tokens."""
    generator = torch.Generator(device="cuda").manual_seed(21)
    tokens, heads = 37, 40
    swa = _v41_cache(500, "swa", generator)
    glob = _v41_cache(900, "global", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 500, generator, index_dtype)
    glob_slots, glob_lens = _selection(tokens, 512, 900, generator, index_dtype)
    segments = (
        (swa, "swa", swa_slots, swa_lens, 64),
        (glob, "global", glob_slots, glob_lens, 64),
    )
    reference = _paged_rows(q, *segments, sink, (32, 4, 1, tokens), with_lse=True)
    for block_h, cwarps in ((16, 8), (32, 4)):
        for splits in (1, 2, 4):
            for full_tokens in (tokens, 0, 1, 20, tokens - 1):
                plan = (block_h, cwarps, splits, full_tokens)
                out, lse = _paged_rows(q, *segments, sink, plan, with_lse=True)
                assert torch.equal(out, reference[0]), plan
                assert torch.equal(lse, reference[1]), plan


@pytest.mark.parametrize("heads", [2, 16, 64])
def test_leaf_rows_do_not_depend_on_the_batch(heads):
    generator = torch.Generator(device="cuda").manual_seed(heads)
    tokens = 300
    swa = _v41_cache(2000, "swa", generator)
    glob = _v41_cache(4000, "global", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 2000, generator)
    glob_slots, glob_lens = _selection(tokens, 512, 4000, generator)
    args = (swa, swa_slots, swa_lens, glob, glob_slots, glob_lens)
    full = _v41_leaf(q, *args, sink)
    for start, count in ((0, 1), (5, 4), (17, 72), (100, 200), (0, 300)):
        rows = slice(start, start + count)
        part = _v41_leaf(
            q[rows], swa, swa_slots[rows], swa_lens[rows], glob,
            glob_slots[rows], glob_lens[rows], sink,
        )  # fmt: skip
        assert torch.equal(part, full[rows]), (start, count)


def test_binaries_are_keyed_by_format_and_id_dtypes_only():
    """Token count, width, strides, rows, pages and split count are runtime
    values: a sweep over them compiles nothing once each binary exists."""
    generator = torch.Generator(device="cuda").manual_seed(31)
    heads = 64
    swa = _v41_cache(400, "swa", generator)
    glob = _v41_cache(800, "global", generator)
    sink = torch.randn(heads, device="cuda", generator=generator)

    def run(tokens, swa_width, extra_width):
        q = torch.randn(
            tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        swa_slots, swa_lens = _selection(tokens, swa_width, 400, generator)
        glob_slots, glob_lens = _selection(tokens, extra_width, 800, generator)
        _v41_leaf(q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink)

    sweep = [
        (1, 128, 512), (2, 128, 512), (3, 128, 480), (5, 64, 100), (8, 128, 512),
        (17, 128, 640), (33, 32, 640), (64, 128, 512), (130, 128, 512), (515, 128, 512),
    ]  # fmt: skip
    for case in sweep:  # every head tile and split class
        run(*case)
    torch.cuda.synchronize()
    compiled = cute_mma._core.compiled_attention
    combine = cute_mma._core.compiled_combine
    before = compiled.cache_info().misses, combine.cache_info().misses
    for case in sweep + [(11, 96, 384), (3, 160, 1408), (5, 128, 577)]:
        run(*case)
    assert (compiled.cache_info().misses, combine.cache_info().misses) == before

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

"""Selected attention that reads the packed cache rows itself, on SM120.

``triton_dsv41_mma_selected_attention`` and ``triton_dsv4_mma_decode`` dequantize
the selected rows in registers instead of gathering them into a BF16
workspace first. Dequantization is exact in BF16 and the loop is the dense
kernel's, so they must return the portable route's result bit for bit.
"""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4 import triton as dsv4_triton
from tokenspeed_kernel.ops.attention.dsv41 import triton as dsv41_triton
from tokenspeed_kernel.platform import Platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and Platform.get().is_nvidia
        and Platform.get().arch_version.major == 12
    ),
    reason="SM120 only",
)

_ROW_DIMS = {"swa": 512, "global": 512, "swa_v4": 512, "global_v4": 512}
_ROW_BYTES = {"swa": 528, "global": 288, "swa_v4": 584, "global_v4": 584}
_ROLE = {"swa": "swa", "global": "global", "swa_v4": "swa", "global_v4": "global"}


def _cache(rows, fmt, generator, *, page_pad=0):
    """A packed ``[pages, 64, row_bytes]`` cache of random rows (cache codec).

    ``page_pad`` bytes of slack after each page make the field a strided view.
    """
    pages = (rows + 63) // 64
    width = _ROW_BYTES[fmt]
    storage = torch.zeros(
        (pages, 64 * width + page_pad), dtype=torch.uint8, device="cuda"
    )
    # A page is page-planar: its data rows, then its scale rows. The external
    # field keeps the [pages, rows, row_bytes] shape over the same bytes.
    cache = storage[:, : 64 * width].view(pages, 64, width)
    values = torch.randn(
        (pages * 64, _ROW_DIMS[fmt]), device="cuda", generator=generator
    ).to(torch.bfloat16)
    values[::7] *= 40
    values[3::11] *= 1e-3
    dsv41.cache_scatter(values, cache, torch.arange(pages * 64, device="cuda"), fmt)
    return cache


def _selection(tokens, width, rows, generator, *, holes=True):
    slots = torch.randint(
        0, rows, (tokens, width), device="cuda", generator=generator
    ).to(torch.int32)
    lens = torch.randint(
        0, width + 1, (tokens,), device="cuda", generator=generator
    ).to(torch.int32)
    lens[0] = width
    if holes:
        slots[torch.rand(tokens, width, device="cuda", generator=generator) < 0.1] = -1
        slots[-1, : width // 2] = rows + 5  # past the cache
    return slots, lens


def _portable(*args, lse=None):
    return dsv41_triton.selected_attention(*args, lse=lse)


def _direct(*args, lse=None):
    return dsv41_triton.mma_selected_attention(*args, lse=lse)


@pytest.mark.parametrize("formats", [("swa", "global")])
@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize(
    ("tokens", "swa_width", "extra_width"),
    [(1, 128, 512), (5, 128, 200), (37, 96, 64), (3, 64, 0)],
)
def test_direct_reader_equals_the_portable_gather_bit_for_bit(
    formats, heads, tokens, swa_width, extra_width
):
    generator = torch.Generator(device="cuda").manual_seed(heads * 100 + tokens)
    swa_rows, global_rows = 500, 900
    swa = _cache(swa_rows, formats[0], generator)
    glob = _cache(global_rows, formats[1], generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, swa_width, swa_rows, generator)
    if extra_width:
        glob_slots, glob_lens = _selection(tokens, extra_width, global_rows, generator)
    else:
        glob = glob_slots = glob_lens = None
    args = (q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink, 512**-0.5)
    tail = (None, 256, None, None, None)
    expected = _portable(*args, *tail)
    actual = _direct(*args, *tail)
    assert torch.equal(actual, expected)
    # Rows with nothing live (length 0) attend to the sink alone: zeros
    assert torch.count_nonzero(actual) > 0


@pytest.mark.parametrize("formats", [("swa", "global")])
def test_direct_reader_returns_the_partial_lse_without_a_sink(formats):
    generator = torch.Generator(device="cuda").manual_seed(5)
    heads, tokens = 64, 9
    swa = _cache(300, formats[0], generator)
    glob = _cache(700, formats[1], generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    swa_slots, swa_lens = _selection(tokens, 128, 300, generator)
    glob_slots, glob_lens = _selection(tokens, 256, 700, generator)
    results = []
    for impl in (_portable, _direct):
        lse = torch.empty(tokens, heads, device="cuda")
        out = impl(
            q,
            swa,
            swa_slots,
            swa_lens,
            glob,
            glob_slots,
            glob_lens,
            None,
            512**-0.5,
            None,
            256,
            None,
            None,
            None,
            lse=lse,
        )
        results.append((out, lse))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


def test_direct_reader_reads_strided_cache_views():
    generator = torch.Generator(device="cuda").manual_seed(11)
    heads, tokens = 64, 6
    swa = _cache(256, "swa", generator, page_pad=96)
    glob = _cache(512, "global", generator, page_pad=32)
    assert swa.stride(0) != swa.stride(1) * swa.shape[1]
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 256, generator)
    glob_slots, glob_lens = _selection(tokens, 512, 512, generator)
    args = (q, swa, swa_slots, swa_lens, glob, glob_slots, glob_lens, sink, 512**-0.5)
    tail = (None, 256, None, None, None)
    assert torch.equal(_direct(*args, *tail), _portable(*args, *tail))


def test_direct_reader_rows_do_not_depend_on_the_batch():
    generator = torch.Generator(device="cuda").manual_seed(13)
    heads, tokens = 64, 8
    swa = _cache(400, "swa", generator)
    glob = _cache(800, "global", generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 400, generator)
    glob_slots, glob_lens = _selection(tokens, 512, 800, generator)
    full = _direct(
        q,
        swa,
        swa_slots,
        swa_lens,
        glob,
        glob_slots,
        glob_lens,
        sink,
        512**-0.5,
        None,
        256,
        None,
        None,
        None,
    )
    for row in (0, 3, 7):
        one = _direct(
            q[row : row + 1],
            swa,
            swa_slots[row : row + 1],
            swa_lens[row : row + 1],
            glob,
            glob_slots[row : row + 1],
            glob_lens[row : row + 1],
            sink,
            512**-0.5,
            None,
            256,
            None,
            None,
            None,
        )
        assert torch.equal(one[0], full[row])


def test_direct_reader_does_not_recompile_for_new_batches():
    from tokenspeed_kernel.ops.attention.dsv4._triton import paged_mma

    generator = torch.Generator(device="cuda").manual_seed(17)
    heads = 64
    swa = _cache(400, "swa", generator)
    glob = _cache(800, "global", generator)
    sink = torch.randn(heads, device="cuda", generator=generator)

    def run(tokens, swa_width, extra_width):
        q = torch.randn(
            tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        swa_slots, swa_lens = _selection(tokens, swa_width, 400, generator)
        glob_slots, glob_lens = _selection(tokens, extra_width, 800, generator)
        _direct(
            q,
            swa,
            swa_slots,
            swa_lens,
            glob,
            glob_slots,
            glob_lens,
            sink,
            512**-0.5,
            None,
            256,
            None,
            None,
            None,
        )

    run(2, 128, 512)
    torch.cuda.synchronize()
    with assert_no_triton_compile(paged_mma._paged_mma_kernel):
        for case in [(1, 128, 512), (7, 128, 480), (33, 64, 100), (16, 32, 640)]:
            run(*case)


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


@pytest.mark.parametrize("heads", [16, 64, 128])
@pytest.mark.parametrize("extra", [False, True])
def test_dsv4_decode_reader_equals_the_workspace_route(heads, extra):
    from tokenspeed_kernel.ops.attention.dsv4 import dsv4_prefill

    generator = torch.Generator(device="cuda").manual_seed(heads)
    tokens, page_size = 11, 64
    swa = _v4_cache(4, page_size, generator)
    compressed = _v4_cache(3, page_size, generator)
    q = torch.randn(
        tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    sink = torch.randn(heads, device="cuda", generator=generator)
    swa_slots, swa_lens = _selection(tokens, 128, 4 * page_size, generator)
    extra_slots, extra_lens = _selection(tokens, 96, 3 * page_size, generator)
    kwargs = dict(
        extra_kv_cache=compressed if extra else None,
        extra_slots=extra_slots if extra else None,
        extra_lens=extra_lens if extra else None,
        extra_page_size=page_size if extra else None,
    )
    actual = dsv4_triton.triton_dsv4_mma_decode(
        q, swa, swa_slots, swa_lens, page_size, sink, 512**-0.5, **kwargs
    )

    # The workspace route of triton_dsv4_decode, with the dense kernel it ran
    # before the direct reader existed.
    width = 128 + (96 if extra else 0)
    workspace = torch.empty(tokens, width, 512, dtype=torch.bfloat16, device="cuda")
    indices = torch.empty(tokens, width, dtype=torch.int32, device="cuda")
    lens = torch.empty(tokens, dtype=torch.int32, device="cuda")
    dequantize = dsv4_triton._dsv4_dequantize_selected_cache_segment
    dequantize(swa, swa_slots, swa_lens, page_size, workspace, indices, lens, 0, width)
    if extra:
        dequantize(
            compressed,
            extra_slots,
            extra_lens,
            page_size,
            workspace,
            indices,
            lens,
            128,
            width,
        )
    expected = dsv4_prefill(
        q=q,
        kv=workspace,
        indices=indices,
        lens=lens,
        attn_sink=sink,
        softmax_scale=512**-0.5,
        out=None,
        override="triton_dsv4_mma_prefill",
    )
    assert torch.equal(actual, expected)

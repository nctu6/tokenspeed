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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import torch_mm_fp8_blockscale
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 MXFP8 block32 GEMM tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.gemm import mm as kernel_mm  # noqa: E402
from tokenspeed_kernel.profiling import ShapeCapture  # noqa: E402
from tokenspeed_kernel.workspace import capture_session, zeroed_state  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp8.block32 import (  # noqa: E402
    BLOCK32_CONFIGS,
    Block32Config,
    launch_triton_mm_mxfp8_block32_gfx950,
    triton_mm_mxfp8_block32_gfx950,
    triton_mm_mxfp8_block32_num_counters_gfx950,
    triton_mm_mxfp8_block32_packed_gfx950,
)

# One BF16 rounding of the output (2**-8 relative), plus the scaled E4M3
# MFMA's reduced-precision internal sum: measured at 1e-4 of sum(|a * b|) on
# these shapes, bounded here at 2**-12 of it.
_RTOL = 2**-8
_SUM_RTOL = 2**-12

_TILED = Block32Config(False, 32, 64, 256, 1, True, 4, 2, 16, 0, 1)
_TILED_SPLIT = Block32Config(False, 64, 32, 128, 1, True, 4, 2, 16, 0, 4)
_PACKED = Block32Config(True, 16, 32, 128, 2, False, 4, 2, 16, 0, 1)
_PACKED_SPLIT = Block32Config(True, 16, 32, 256, 2, False, 4, 2, 16, 0, 2)


def _operands(m: int, n: int, k: int):
    a = (torch.randn((m, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    b = (torch.randn((n, k), device="cuda") * 0.5).to(torch.float8_e4m3fn)
    a_scales = torch.randint(120, 130, (m, k // 32), device="cuda", dtype=torch.uint8)
    b_scales = torch.randint(118, 128, (n, k // 32), device="cuda", dtype=torch.uint8)
    return a, b, a_scales, b_scales


def _reference(a, b, a_scales, b_scales) -> torch.Tensor:
    return torch_mm_fp8_blockscale(
        a, b, a_scales, b_scales, torch.float32, block_size=[1, 32]
    )


def _dequantized(values, scales) -> torch.Tensor:
    return values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)


def _assert_close(actual, a, b, a_scales, b_scales, rows=None) -> None:
    a, a_scales = (a, a_scales) if rows is None else (a[:rows], a_scales[:rows])
    expected = _reference(a, b, a_scales, b_scales)
    magnitude = _dequantized(a, a_scales).abs() @ _dequantized(b, b_scales).abs().T
    error = (actual.float() - expected).abs()
    bound = _RTOL * expected.abs() + _SUM_RTOL * magnitude
    assert bool((error <= bound).all()), f"max excess {(error - bound).max().item()}"


def _run(a, b, a_scales, b_scales, config, out=None) -> torch.Tensor:
    counters = zeroed_state(
        triton_mm_mxfp8_block32_num_counters_gfx950(
            a.shape[0], b.shape[0], a.shape[1], config
        ),
        dtype=torch.int32,
        device=a.device,
    )
    return launch_triton_mm_mxfp8_block32_gfx950(
        a,
        b,
        a_scales,
        b_scales,
        torch.bfloat16,
        alpha=None,
        block_size=[1, 32],
        counters=counters,
        out=out,
        config=config,
    )


@pytest.mark.parametrize("config", [_TILED, _TILED_SPLIT, _PACKED, _PACKED_SPLIT])
@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1152, 5120),
        (37, 5120, 576),
        (96, 1792, 5120),
        # Ragged N, and a K that is not a multiple of any K tile or split.
        (45, 1000, 800),
    ],
)
def test_matches_dequantized_reference(m: int, n: int, k: int, config) -> None:
    torch.manual_seed(0)
    a, b, a_scales, b_scales = _operands(m, n, k)
    _assert_close(_run(a, b, a_scales, b_scales, config), a, b, a_scales, b_scales)


def test_tuned_configs_match_reference() -> None:
    torch.manual_seed(0)
    for (n, k), tiers in BLOCK32_CONFIGS.items():
        a, b, a_scales, b_scales = _operands(tiers[-1][0], n, k)
        lower = 1
        for max_m, _ in tiers:
            for rows in sorted({lower, max_m}):
                actual = _run(a[:rows], b, a_scales[:rows], b_scales, None)
                _assert_close(actual, a, b, a_scales, b_scales, rows)
            lower = max_m + 1


def test_writes_strided_out_deterministically() -> None:
    torch.manual_seed(0)
    m, n, k = 40, 1152, 5120
    a, b, a_scales, b_scales = _operands(m, n, k)
    backing = torch.empty((m, n + 24), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :n]
    actual = _run(a, b, a_scales, b_scales, _TILED_SPLIT, out=out)
    assert actual is out
    _assert_close(out, a, b, a_scales, b_scales)
    # The split-K reduction sums partials in split order.
    for _ in range(3):
        assert torch.equal(_run(a, b, a_scales, b_scales, _TILED_SPLIT), actual)


def test_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    n, k = 1792, 5120
    a, b, a_scales, b_scales = _operands(256, n, k)

    def run(rows: int, config) -> torch.Tensor:
        return _run(a[:rows], b, a_scales[:rows], b_scales, config)

    for config in (_TILED_SPLIT, _PACKED):
        run(1, config)
    with assert_no_triton_compile(
        triton_mm_mxfp8_block32_gfx950, triton_mm_mxfp8_block32_packed_gfx950
    ):
        for rows in (2, 6, 16, 24, 48, 96, 100, 192, 256):
            for config in (_TILED_SPLIT, _PACKED):
                # Rows are independent: a shorter batch is a prefix.
                _assert_close(run(rows, config), a, b, a_scales, b_scales, rows)


def test_hip_graph_replay_keeps_counters_zeroed() -> None:
    torch.manual_seed(0)
    m, n, k = 24, 1152, 5120
    a, b, a_scales, b_scales = _operands(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _run(a, b, a_scales, b_scales, _PACKED_SPLIT, out=out)
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            _run(a, b, a_scales, b_scales, _PACKED_SPLIT, out=out)
    stream.synchronize()

    for _ in range(3):
        new_a, _, new_scales, _ = _operands(m, n, k)
        a.copy_(new_a)
        a_scales.copy_(new_scales)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(out, a, b, a_scales, b_scales)


def test_public_mm_selects_tuned_rows() -> None:
    (n, k), tiers = next(iter(BLOCK32_CONFIGS.items()))
    max_m = tiers[-1][0]
    torch.manual_seed(0)
    a, b, a_scales, b_scales = _operands(max_m + 1, n, k)
    ShapeCapture.reset()
    capture = ShapeCapture.get()
    capture.enabled = True
    try:
        for rows in (max_m, max_m + 1):
            kernel_mm(
                a[:rows],
                b,
                A_scales=a_scales[:rows],
                B_scales=b_scales,
                out_dtype=torch.bfloat16,
                block_size=[1, 32],
                quant="mxfp8",
            )
    finally:
        capture.enabled = False
    names = [record.kernel_name for record in capture._records[-2:]]
    assert names[0] == "triton_mm_mxfp8_block32_gfx950"
    assert names[1] != "triton_mm_mxfp8_block32_gfx950"

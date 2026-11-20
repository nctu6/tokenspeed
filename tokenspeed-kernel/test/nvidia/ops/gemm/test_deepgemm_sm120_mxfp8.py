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

"""SM120 DeepGEMM MXFP8 (1x32 UE8M0 on both sides): scale packer and the ``mm`` leaf."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.gemm._triton.ue8m0_pack import (
    _pack_ue8m0_mn_major_kernel,
    pack_ue8m0_mn_major,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.thirdparty.deepgemm import (
    deepgemm_supports_sm120,
    pack_ue8m0_scales,
)
from utils import assert_no_triton_compile

LEAF = "deepgemm_sm120_mm_mxfp8"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)
requires_sm120_deepgemm = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and current_platform().arch_version.major == 12
        and deepgemm_supports_sm120()
    ),
    reason="requires SM120 and a DeepGEMM build with SM120 kernels",
)


def _scales(rows: int, groups: int) -> torch.Tensor:
    return torch.randint(110, 135, (rows, groups), dtype=torch.uint8, device="cuda")


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 9, 64, 130, 4099])
@pytest.mark.parametrize("groups", [4, 20, 40, 256])
def test_pack_matches_the_torch_packer(rows: int, groups: int) -> None:
    sf = _scales(rows, groups)
    packed = pack_ue8m0_mn_major(sf)
    expected = pack_ue8m0_scales(sf)
    assert packed.dtype == torch.int32
    assert packed.stride() == expected.stride() == (1, (rows + 3) // 4 * 4)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@requires_cuda
def test_pack_accepts_a_padded_row_stride() -> None:
    wide = _scales(33, 24)
    view = wide[:, :20]  # row stride 24 bytes, still word aligned
    torch.testing.assert_close(
        pack_ue8m0_mn_major(view), pack_ue8m0_scales(view.contiguous()), rtol=0, atol=0
    )


@requires_cuda
def test_pack_does_not_recompile_across_batch_sizes() -> None:
    groups = 40
    for rows in (1, 17):  # both integer-specialization classes
        pack_ue8m0_mn_major(_scales(rows, groups))
    with assert_no_triton_compile(_pack_ue8m0_mn_major_kernel):
        for rows in (1, 2, 7, 16, 33, 96, 257, 1000, 4096, 5000):
            pack_ue8m0_mn_major(_scales(rows, groups))


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        solution="triton",
    )


def _dequantize(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(sf.float() - 127.0).repeat_interleave(32, dim=1)
    return q.float() * scale


@requires_sm120_deepgemm
@pytest.mark.parametrize("m", [1, 2, 7, 64, 130, 2048])
@pytest.mark.parametrize(
    ("n", "k"),
    [(1280, 5120), (32768, 1280), (512, 5120), (5120, 2304), (5120, 8192)],
)
def test_mm_selects_the_leaf_and_matches_the_dequantized_reference(
    m: int, n: int, k: int
) -> None:
    torch.manual_seed(m + n)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qa, sfa = _quantize(a)
    qw, sfw = _quantize(w)
    out = mm(
        qa,
        qw,
        A_scales=sfa,
        B_scales=sfw,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
    )
    reference = _dequantize(qa, sfa) @ _dequantize(qw, sfw).T
    # FP32 accumulation of exactly representable products; the output is one
    # BF16 rounding of the reference, so the error is at most a BF16 ulp.
    torch.testing.assert_close(
        out.float(), reference, atol=4e-3 * reference.abs().max().item(), rtol=1e-2
    )
    forced = mm(
        qa,
        qw,
        A_scales=sfa,
        B_scales=sfw,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        override=LEAF,
    )
    torch.testing.assert_close(out, forced, rtol=0, atol=0)


@requires_sm120_deepgemm
def test_the_leaf_is_registered_for_sm120_only() -> None:
    spec = KernelRegistry.get().get_by_name(LEAF)
    assert spec is not None
    assert spec.capability.min_arch_version.major == 12
    assert spec.capability.max_arch_version.major == 12


@requires_sm120_deepgemm
def test_unaligned_shapes_stay_on_the_other_leaves() -> None:
    # K = 160 is not a multiple of 128: the DeepGEMM leaf declines and Triton serves it.
    m, n, k = 5, 128, 160
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    qa, sfa = _quantize(a)
    qw, sfw = _quantize(w)
    out = mm(
        qa,
        qw,
        A_scales=sfa,
        B_scales=sfw,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
    )
    reference = _dequantize(qa, sfa) @ _dequantize(qw, sfw).T
    torch.testing.assert_close(out.float(), reference, atol=1e-2, rtol=1e-2)


@requires_sm120_deepgemm
@pytest.mark.parametrize(("n", "k"), [(1280, 5120), (5120, 2304), (32768, 1280)])
def test_rows_do_not_depend_on_the_batch(n: int, k: int) -> None:
    torch.manual_seed(n)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(1024, k, device="cuda", dtype=torch.bfloat16)
    qw, sfw = _quantize(w)
    qa, sfa = _quantize(a)

    def run(m: int) -> torch.Tensor:
        return mm(
            qa[:m],
            qw,
            A_scales=sfa[:m],
            B_scales=sfw,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
        )

    expected = run(1024)
    for m in (1, 2, 3, 4, 8, 17, 64, 130, 257):
        torch.testing.assert_close(run(m), expected[:m], rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("m", [1, 3, 4, 5, 64, 130])
@pytest.mark.parametrize("k", [128, 1280, 5120])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mn_major_quantizer_matches_quantize_then_pack(
    m: int, k: int, dtype: torch.dtype
) -> None:
    from tokenspeed_kernel.ops.quantization.triton import (
        quantize_fp8_group32_mn_major_scales,
    )

    x = torch.randn(m, k, device="cuda") * torch.rand(m, k, device="cuda") ** 4 * 50
    x[0, :32] = 0
    x = x.to(dtype)
    q, sf = _quantize(x)
    q_mn, sf_mn = quantize_fp8_group32_mn_major_scales(x)
    assert torch.equal(q_mn.view(torch.uint8), q.view(torch.uint8))
    torch.testing.assert_close(sf_mn, pack_ue8m0_scales(sf), rtol=0, atol=0)
    assert sf_mn.stride() == (1, (m + 3) // 4 * 4)


@requires_cuda
def test_mn_major_quantizer_does_not_recompile_across_batch_sizes() -> None:
    from tokenspeed_kernel.ops.quantization.triton import (
        _fp8_group32_ue8m0_quantize_mn_major,
        quantize_fp8_group32_mn_major_scales,
    )

    k = 1280
    for m in (1, 17):
        quantize_fp8_group32_mn_major_scales(
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        )
    with assert_no_triton_compile(_fp8_group32_ue8m0_quantize_mn_major):
        for m in (1, 2, 7, 16, 33, 96, 257, 1000, 4096):
            quantize_fp8_group32_mn_major_scales(
                torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            )


@requires_sm120_deepgemm
@pytest.mark.parametrize("m", [1, 7, 130])
@pytest.mark.parametrize("prequantized", [False, True])
def test_prepacked_weight_scales_match_the_leaf(m: int, prequantized: bool) -> None:
    from tokenspeed_kernel.ops.gemm import prepare_mxfp8_weight_scales

    n, k = 1280, 5120
    torch.manual_seed(m)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    qw, sfw = _quantize(w)
    qx, sfx = _quantize(x)
    prepared = prepare_mxfp8_weight_scales(qw, sfw)
    assert prepared is not None and prepared.dtype == torch.int32
    expected = mm(
        qx,
        qw,
        A_scales=sfx,
        B_scales=sfw,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        override=LEAF,
    )
    actual = mm(
        qx if prequantized else x,
        qw,
        A_scales=sfx if prequantized else None,
        B_scales=prepared,
        out_dtype=torch.bfloat16,
        quant="mxfp8",
        block_size=[1, 32],
        prepacked_scales=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_sm120_deepgemm
def test_unaligned_weights_keep_the_canonical_scales() -> None:
    from tokenspeed_kernel.ops.gemm import prepare_mxfp8_weight_scales

    qw, sfw = _quantize(torch.randn(100, 512, device="cuda", dtype=torch.bfloat16))
    assert prepare_mxfp8_weight_scales(qw, sfw) is None

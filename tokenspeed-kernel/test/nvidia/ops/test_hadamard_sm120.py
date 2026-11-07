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
"""``triton_hadamard_transform_128_mma``: the SM120 Hadamard transform.

The fast-hadamard-transform wheel ships no sm_120 cubin, so SM120 runs the
tensor-core Triton leaf. It is checked against a float64 transform of the same
stored values.
"""

import pytest
import torch
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.ops.transform import triton as transform
from tokenspeed_kernel.platform import Platform, PlatformInfo
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from utils import assert_no_triton_compile, warm_specialization_classes

on_sm120 = pytest.mark.skipif(
    not (Platform.get().is_nvidia and Platform.get().arch_version.major == 12),
    reason="SM120 required",
)

SCALE = 128**-0.5
# The sign matrix is exact in BF16/FP16 and the products accumulate in FP32,
# so the result differs from the exact transform by the output rounding only:
# the FP32 accumulation error can flip a near-tie to the neighbouring value,
# one ulp (2**-7 for BF16, 2**-10 for FP16, relative), plus a floor of 1e-5
# for results that cancel to near zero in N(0, 1) rows.
TOLERANCE = {
    torch.bfloat16: {"atol": 1.0e-5, "rtol": 2.0**-7},
    torch.float16: {"atol": 1.0e-5, "rtol": 2.0**-10},
}


def _reference(x: torch.Tensor, scale: float) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64, device=x.device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)))
    return (x.double().reshape(-1, 128) @ h * scale).reshape(x.shape).to(x.dtype)


def _selected(platform: PlatformInfo, dtype: torch.dtype) -> str:
    return select_kernel(
        "transform",
        "hadamard_transform",
        format_signature(x=dense_tensor_format(dtype)),
        platform=platform,
        traits={"last_dim": 128, "contiguous": True},
    ).name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sm120_selects_the_mma_leaf(dtype, sm120_platform: PlatformInfo):
    assert _selected(sm120_platform, dtype) == "triton_hadamard_transform_128_mma"


def test_other_archs_and_float32_keep_their_leaves(
    sm120_platform: PlatformInfo,
    b200_platform: PlatformInfo,
    h100_platform: PlatformInfo,
):
    # FP32 has no tensor-core leaf; SM90 and SM100 use the CUDA wheel.
    assert (
        select_kernel(
            "transform",
            "hadamard_transform",
            format_signature(x=dense_tensor_format(torch.float32)),
            platform=sm120_platform,
            traits={"last_dim": 128, "contiguous": True},
        ).name
        == "triton_hadamard_transform_128"
    )
    for platform in (b200_platform, h100_platform):
        assert _selected(platform, torch.bfloat16) != (
            "triton_hadamard_transform_128_mma"
        )


@on_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 128), (3, 5, 128), (37, 32, 128), (2049, 128)])
def test_matches_the_float64_transform(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    out = hadamard_transform(x, scale=SCALE, solution="triton")
    assert out.shape == x.shape and out.dtype == dtype
    torch.testing.assert_close(out, _reference(x, SCALE), **TOLERANCE[dtype])


@on_sm120
def test_selection_picks_the_mma_leaf_on_this_device():
    assert _selected(Platform.get(), torch.bfloat16) == (
        "triton_hadamard_transform_128_mma"
    )


@on_sm120
def test_non_contiguous_and_empty_inputs():
    torch.manual_seed(1)
    base = torch.randn(9, 40, 128, device="cuda", dtype=torch.bfloat16)
    x = base[:, ::2]
    assert not x.is_contiguous()
    out = hadamard_transform(x, scale=0.5, solution="triton")
    torch.testing.assert_close(out, _reference(x, 0.5), **TOLERANCE[torch.bfloat16])
    empty = torch.empty(0, 4, 128, device="cuda", dtype=torch.bfloat16)
    assert hadamard_transform(empty, solution="triton").shape == empty.shape


@on_sm120
def test_row_count_and_scale_do_not_recompile():
    def run(rows):
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        out = transform.triton_hadamard_transform_128_mma(x, scale=1.0 / (rows + 1))
        torch.testing.assert_close(
            out, _reference(x, 1.0 / (rows + 1)), **TOLERANCE[torch.bfloat16]
        )

    # Rows on both sides of the 2048-row launch-config bucket.
    sweep = (1, 2, 17, 33, 48, 97, 130, 1483, 2048, 2049, 4096, 5000)
    # Triton keys integer arguments on "equals 1" and "divisible by 16".
    warm_specialization_classes(
        run,
        lambda rows: (rows == 1, rows % 16 == 0, rows <= 2048),
        sweep,
        pool=(1, 2, 16, 48, 2049, 4097),
    )
    with assert_no_triton_compile(transform._hadamard_128_mma_kernel):
        for rows in sweep:
            run(rows)

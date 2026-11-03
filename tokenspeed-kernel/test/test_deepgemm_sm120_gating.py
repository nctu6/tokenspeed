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

"""DeepGEMM on SM120: scale-format and registry gating.

SM120 DeepGEMM kernels read FP32 block scales (the SM90 recipe), so only the
SM100 family requantizes to UE8M0. DeepGEMM leaves are offered on SM120 only by
builds that report SM120 kernels; older builds would fail there at runtime.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from tokenspeed_kernel.ops.moe._deep_gemm.ue8m0 import deep_gemm_requires_ue8m0
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo
from tokenspeed_kernel.signature import ScaleFormat, format_signatures
from tokenspeed_kernel.thirdparty.deepgemm import deepgemm_supports_sm120


@pytest.mark.parametrize(
    ("major", "minor", "expected"),
    [
        (9, 0, False),  # Hopper: FP32 scales
        (10, 0, True),  # B200: UE8M0-only 1d1d kernels
        (10, 3, True),  # B300
        (12, 0, False),  # RTX PRO 6000: FP32 scales
    ],
)
def test_ue8m0_only_on_sm100_family(
    major: int,
    minor: int,
    expected: bool,
    h100_platform: PlatformInfo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TOKENSPEED_DISABLE_DEEP_GEMM_UE8M0", raising=False)
    Platform.override(
        dataclasses.replace(h100_platform, arch_version=ArchVersion(major, minor))
    )
    try:
        assert deep_gemm_requires_ue8m0() is expected
    finally:
        Platform.reset()


def _mxfp8_signature():
    signatures = format_signatures(
        ("a", "b"),
        "mxfp8",
        {torch.float8_e4m3fn},
        scale=ScaleFormat(
            storage_dtype=torch.float32, granularity="block", block_shape=(128, 128)
        ),
    )
    return next(iter(signatures))


def _select_dense(platform: PlatformInfo, m: int) -> str:
    from tokenspeed_kernel.selection import select_kernel

    # The same label the ``mm`` facade derives from the platform
    block_scale_layout = (
        "canonical_blackwell" if platform.is_blackwell_plus else "canonical"
    )

    return select_kernel(
        "gemm",
        "mm",
        _mxfp8_signature(),
        platform=platform,
        traits={
            "m": m,
            "n": 4096,
            "k": 7168,
            "a_inner_stride_one": True,
            "b_inner_stride_one": True,
            "block_scale_layout": block_scale_layout,
            "out_dtype": torch.bfloat16,
            "out_inner_stride_one": True,
        },
    ).name


@pytest.mark.parametrize("m", [1, 64, 4096])
def test_sm120_auto_selects_deepgemm_with_sm120_build(
    m: int, sm120_platform: PlatformInfo
) -> None:
    if not deepgemm_supports_sm120():
        pytest.skip("installed DeepGEMM has no SM120 kernels")
    import tokenspeed_kernel.ops.gemm  # noqa: F401  (triggers registration)

    assert _select_dense(sm120_platform, m) == "deepgemm_sm120_mm_fp8_blockscale"


def test_flashinfer_blockscale_steps_back_on_sm120_only_with_sm120_build(
    sm120_platform: PlatformInfo, b200_platform: PlatformInfo
) -> None:
    pytest.importorskip("flashinfer")
    import tokenspeed_kernel.ops.gemm  # noqa: F401  (triggers registration)
    from tokenspeed_kernel.registry import KernelRegistry

    reg = KernelRegistry.get()
    spec = reg.get_by_name("flashinfer_mm_fp8_blockscale")
    if spec is None:
        pytest.skip("FlashInfer block-FP8 GEMM unavailable")
    assert spec.capability.satisfied_by(b200_platform)
    assert spec.capability.satisfied_by(sm120_platform) is not deepgemm_supports_sm120()


def test_hopper_auto_selection_unchanged(h100_platform: PlatformInfo) -> None:
    pytest.importorskip("deep_gemm")
    import tokenspeed_kernel.ops.gemm  # noqa: F401  (triggers registration)

    assert _select_dense(h100_platform, 64) == "triton_mm_fp8_blockscale"


def test_dsa_deepgemm_offered_on_sm120_only_with_sm120_build(
    sm120_platform: PlatformInfo,
) -> None:
    pytest.importorskip("deep_gemm")
    import tokenspeed_kernel.ops.attention.dsa  # noqa: F401  (triggers registration)
    from tokenspeed_kernel.registry import KernelRegistry

    reg = KernelRegistry.get()
    for mode in ("dsa_plan", "dsa_decode_topk", "dsa_prefill_topk"):
        specs = reg.get_for_operator("attention", mode, platform=sm120_platform)
        solutions = {s.solution for s in specs}
        assert ("deep_gemm" in solutions) is deepgemm_supports_sm120(), (
            f"{mode}: deep_gemm offered={('deep_gemm' in solutions)}, "
            f"build supports SM120={deepgemm_supports_sm120()}"
        )
        assert "triton" in solutions, f"{mode}: portable triton must remain"

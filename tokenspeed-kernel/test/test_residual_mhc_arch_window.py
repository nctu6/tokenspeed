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
"""The DeepGEMM mHC prenorm kernels are built for the SM90/SM100-SM103 window.

Their ``tf32_hc_prenorm_gemm`` cubin raises "Unsupported architecture" at
runtime on SM120 (workstation Blackwell), so the registry must not pick them
there: the capability advertises ``max_arch_version = 10.3`` and SM120 falls
through to the portable Triton mHC path. These tests guard that contract so a
future edit that widens or drops the cap is caught before a serve-time crash.
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, PlatformInfo


def _cap() -> CapabilityRequirement:
    # The exact window the DeepGEMM mHC kernels declare.
    return CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    )


@pytest.mark.parametrize(
    ("major", "minor", "admitted"),
    [
        (9, 0, True),  # Hopper H100/H200
        (10, 0, True),  # datacenter Blackwell B200
        (10, 3, True),  # datacenter Blackwell B300 (window top)
        (12, 0, False),  # workstation Blackwell RTX PRO 6000 -> excluded
        (8, 0, False),  # Ampere, below the floor
    ],
)
def test_deep_gemm_mhc_window_admits_only_sm90_to_sm103(
    major: int, minor: int, admitted: bool, h100_platform: PlatformInfo
) -> None:
    platform = PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(major, minor),
        device_name=f"sm{major}{minor}",
        device_count=1,
        total_memory=h100_platform.total_memory,
        memory_bandwidth=h100_platform.memory_bandwidth,
        sm_count=h100_platform.sm_count,
        max_threads_per_sm=h100_platform.max_threads_per_sm,
        max_shared_memory_per_sm=h100_platform.max_shared_memory_per_sm,
        sm_features=h100_platform.sm_features,
        runtime_features=h100_platform.runtime_features,
    )
    assert _cap().satisfied_by(platform) is admitted


def test_window_excludes_sm120_but_min_only_would_admit_it(
    sm120_platform: PlatformInfo,
) -> None:
    # Without the cap (the pre-fix state) SM120 >= 9.0 would wrongly qualify;
    # the max_arch_version is what excludes it.
    without_cap = CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    )
    assert without_cap.satisfied_by(sm120_platform) is True
    assert _cap().satisfied_by(sm120_platform) is False


@pytest.mark.usefixtures("fresh_registry")
def test_registry_picks_triton_mhc_on_sm120_not_deep_gemm(
    sm120_platform: PlatformInfo,
) -> None:
    # Integration against the real residual registrations. Skipped where the
    # DeepGEMM vendor dependency is unavailable (no CUDA toolkit on the host),
    # since then the deep_gemm specs never register and the check is vacuous.
    deep_gemm = pytest.importorskip("deep_gemm")
    del deep_gemm
    import tokenspeed_kernel.ops.residual  # noqa: F401  (triggers registration)
    from tokenspeed_kernel.registry import KernelRegistry

    reg = KernelRegistry.get()
    for mode in ("mhc_mixes", "mhc_pre"):
        specs = reg.get_for_operator("residual", mode, platform=sm120_platform)
        solutions = {s.solution for s in specs}
        assert (
            "deep_gemm" not in solutions
        ), f"{mode}: deep_gemm must be filtered out on SM120, got {solutions}"
        assert (
            "triton" in solutions
        ), f"{mode}: portable triton must remain on SM120, got {solutions}"

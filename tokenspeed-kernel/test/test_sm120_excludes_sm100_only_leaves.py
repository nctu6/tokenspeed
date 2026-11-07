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

"""Leaves whose kernels cannot run on SM120 must not be offered there.

FlashMLA (SM90/SM100-family kernels), FlashInfer's ``trtllm-gen`` backend
(SM100/SM103/SM107), ``tokenspeed_mla`` (SM100/SM103/SM107) and the
fast-hadamard-transform wheel (sm_90/sm_100/sm_103 cubins, no PTX) all fail at
run time on workstation Blackwell (SM120). Their leaves outrank the portable
Triton leaves, so a capability without an upper bound makes ``auto`` selection
pick them there and crash at serve time.
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry

# (family, mode) -> solutions whose kernels cannot run on SM120
_SM100_ONLY_SOLUTIONS = {
    ("attention", "dsa_decode"): {"flashinfer_trtllm", "flashmla"},
    ("attention", "dsa_prefill"): {"flashinfer_trtllm", "flashmla"},
    ("attention", "dsv4_decode"): {"flashmla"},
    ("attention", "dsv4_prefill"): {"flashmla"},
    ("attention", "mla_decode_with_kvcache"): {"tokenspeed_mla"},
    ("transform", "hadamard_transform"): {"fast_hadamard_transform"},
}


@pytest.mark.parametrize(
    ("family", "mode"),
    sorted(_SM100_ONLY_SOLUTIONS),
    ids=[f"{family}.{mode}" for family, mode in sorted(_SM100_ONLY_SOLUTIONS)],
)
def test_sm100_only_leaves_are_not_offered_on_sm120(
    family: str, mode: str, b200_platform: PlatformInfo, sm120_platform: PlatformInfo
) -> None:
    import tokenspeed_kernel.ops.attention  # noqa: F401  (registers the leaves)
    import tokenspeed_kernel.ops.transform  # noqa: F401

    registry = KernelRegistry.get()
    sm100_only = _SM100_ONLY_SOLUTIONS[family, mode]
    on_b200 = {
        spec.solution
        for spec in registry.get_for_operator(family, mode, platform=b200_platform)
    }
    if not sm100_only & on_b200:
        pytest.skip(f"no SM100-only {family}.{mode} leaf registered here")

    on_sm120 = {
        spec.solution
        for spec in registry.get_for_operator(family, mode, platform=sm120_platform)
    }
    assert not on_sm120 & sm100_only, f"{mode} offers {on_sm120 & sm100_only} on SM120"
    assert "triton" in on_sm120, f"{mode} has no portable leaf on SM120: {on_sm120}"


# Leaves of a shared operator whose kernels need SM100/SM103 hardware: the packed
# FP4 paged MQA scorer (DeepGEMM has no such kernel on SM120) and the upstream
# DeepSelect package (sm100 cubins only). SM120 runs FP8 index rows through the
# ``deep_gemm_sm120_*`` leaves and the in-tree CuTe DSL selector instead.
_SM100_ONLY_KERNELS = [
    ("attention", "dsv41_index_topk", "deep_gemm_dsv41_index_topk"),
    ("attention", "dsv41_select_topk", "deepselect_dsv41_select_topk"),
    ("attention", "dsv41_select_candidates", "deepselect_dsv41_select_candidates"),
]


@pytest.mark.parametrize(
    ("family", "mode", "name"),
    _SM100_ONLY_KERNELS,
    ids=[k[2] for k in _SM100_ONLY_KERNELS],
)
def test_sm100_only_kernels_are_not_offered_on_sm120(
    family: str,
    mode: str,
    name: str,
    b200_platform: PlatformInfo,
    sm120_platform: PlatformInfo,
) -> None:
    import tokenspeed_kernel.ops.attention  # noqa: F401  (registers the leaves)
    import tokenspeed_kernel.ops.attention.dsv41  # noqa: F401

    spec = KernelRegistry.get().get_by_name(name)
    assert spec is not None and (spec.family, spec.mode) == (family, mode)
    assert spec.capability.satisfied_by(b200_platform), name
    assert not spec.capability.satisfied_by(sm120_platform), name

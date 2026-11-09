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

None of those kernels can be ported as is (wgmma, tcgen05 and TMEM are absent
on SM120), so SM120 has its own tensor-core Triton leaves for what they served;
``test_sm120_runs_its_own_leaves`` pins which leaf replaces which.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from tokenspeed_kernel.thirdparty.deepgemm import deepgemm_supports_sm120

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


# Leaves of a shared operator whose kernels need SM100/SM103 hardware: the
# upstream DeepSelect package (sm100 cubins only, and its shared memory does not
# fit SM120), which the sm100 packed leaf selects with. SM120 scores both FP8 and
# packed FP4 index rows with DeepGEMM's SM120 MQA kernels in the
# ``deep_gemm_sm120_*`` leaves and selects with the in-tree CuTe DSL kernel.
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


_MLA_DECODE = {
    "q_len": 1,
    "num_q_heads": 128,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "page_size": 64,
    "noncausal_block_size": 1,
    "block_on_query_axis": False,
    "logit_cap": False,
    "return_lse": False,
    "sliding_window": False,
}
_DSA = {
    "q_len": 1,
    "qk_nope_head_dim": 256,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 0,
    "page_size": 64,
    "topk": 2051,
    "has_kv_cache": True,
    "has_sparse_kv_cache": False,
    "topk_layout": "global_slots",
    "slot_order": "selection",
    "logit_cap": False,
    "return_lse": False,
}
_DSV4_PREFILL = {
    "num_q_heads": 64,
    "head_dim": 512,
    "selected_width": 640,
    "cache_layout": "dense_workspace",
    "metadata_dtypes": frozenset({torch.int32}),
    "sinks": True,
}
_DSV4_DECODE = {
    "num_tokens": 32,
    "num_q_heads": 16,
    "head_dim": 512,
    "swa_selected_width": 128,
    "extra_selected_width": 512,
    "cache_layout": "fp8_swa_page_planar",
    "has_extra_segment": True,
    "metadata_dtypes": frozenset({torch.int32}),
    "return_lse": False,
    "sinks": True,
    "topk_layout": "global_slots",
}
_HADAMARD = {"last_dim": 128, "contiguous": True}
_DSV41_PACKED_INDEX = {
    "index_heads": 32,
    "index_k_format": "mxfp4",
    "index_shards": 1,
    "native_indexer": True,
}
# The CuTe DSL attention leaves (DSA cores, MLA decode, V4 prefill from 17
# heads) outrank the Triton ones where the CuTe DSL imports.
_DSA_CORE = (
    "cute_dsl"
    if importlib.util.find_spec("cutlass") is not None
    and importlib.util.find_spec("cutlass.cute") is not None
    else "triton"
)
# (family, mode, signature roles, traits, leaf SM120 runs, leaf it replaces)
_SM120_LEAVES = [
    (
        "transform",
        "hadamard_transform",
        ("x",),
        _HADAMARD,
        "triton_hadamard_transform_128_mma",
        "fast_hadamard_transform",
    ),
    (
        "attention",
        "mla_decode_with_kvcache",
        ("q", "kv_cache"),
        _MLA_DECODE,
        f"{_DSA_CORE}_mla_mma_decode_with_kvcache",
        "tokenspeed_mla_decode_with_kvcache",
    ),
    (
        "attention",
        "dsv4_prefill",
        ("q", "kv"),
        _DSV4_PREFILL,
        f"{_DSA_CORE}_dsv4_mma_prefill",
        "flashmla_dsv4_prefill",
    ),
    (
        "attention",
        "dsv4_decode",
        ("q", "swa_kv_cache"),
        _DSV4_DECODE,
        "triton_dsv4_mma_decode",
        "flashmla_dsv4_decode",
    ),
    (
        "attention",
        "dsv41_selected_attention",
        ("x",),
        {"flashmla_eligible": False},
        "triton_dsv41_mma_selected_attention",
        "flashmla_dsv41_selected_attention",
    ),
    (
        "attention",
        "dsa_decode",
        ("q",),
        {**_DSA, "q_len": 4},
        f"{_DSA_CORE}_dsa_mma_decode",
        "flashinfer_trtllm_nope_dsa_decode",
    ),
    (
        "attention",
        "dsa_prefill",
        ("q",),
        _DSA,
        f"{_DSA_CORE}_dsa_mma_prefill",
        "flashinfer_trtllm_nope_dsa_prefill",
    ),
]
# The V4.1 packed-row indexer exists on SM120 only in DeepGEMM builds with SM120
# kernels; otherwise the portable leaf serves it
if deepgemm_supports_sm120():
    _SM120_LEAVES.append(
        (
            "attention",
            "dsv41_index_topk",
            ("x",),
            _DSV41_PACKED_INDEX,
            "deep_gemm_sm120_dsv41_index_topk",
            "deep_gemm_dsv41_index_topk",
        )
    )


@pytest.mark.parametrize(
    ("family", "mode", "roles", "traits", "sm120_leaf", "replaced"),
    _SM120_LEAVES,
    ids=[f"{leaf[0]}.{leaf[1]}" for leaf in _SM120_LEAVES],
)
def test_sm120_runs_its_own_leaves(
    family, mode, roles, traits, sm120_leaf, replaced, sm120_platform: PlatformInfo
) -> None:
    import tokenspeed_kernel.ops.attention  # noqa: F401  (registers the leaves)
    import tokenspeed_kernel.ops.attention.dsa  # noqa: F401
    import tokenspeed_kernel.ops.attention.dsv4  # noqa: F401
    import tokenspeed_kernel.ops.attention.dsv41  # noqa: F401
    import tokenspeed_kernel.ops.attention.mla  # noqa: F401
    import tokenspeed_kernel.ops.transform  # noqa: F401

    dtype = torch.bfloat16
    signature = format_signature(
        **{
            role: dense_tensor_format(torch.uint8 if role == "swa_kv_cache" else dtype)
            for role in roles
        }
    )
    chosen = select_kernel(
        family, mode, signature, platform=sm120_platform, traits=traits
    )
    assert chosen.name == sm120_leaf
    replaced_spec = KernelRegistry.get().get_by_name(replaced)
    if replaced_spec is not None:
        # The capped leaf stays registered for the platforms it was built for.
        assert not replaced_spec.capability.satisfied_by(sm120_platform), replaced

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
"""MegaMoE input staging: the expert range a kernel call serves."""

import pytest
import torch
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import (
    _stage_mxfp4_mega_moe_inputs_kernel,
    stage_mxfp4_mega_moe_inputs,
)
from utils import assert_no_triton_compile

HIDDEN = 256
TOP_K = 6
NUM_EXPERTS = 32


def _stage(hidden, ids, weights, begin, count):
    num_tokens = hidden.shape[0]
    x = torch.empty(num_tokens, HIDDEN, dtype=torch.float8_e4m3fn, device="cuda")
    sf = torch.empty(num_tokens, HIDDEN // 128, dtype=torch.int32, device="cuda")
    idx = torch.empty(num_tokens, TOP_K, dtype=torch.int64, device="cuda")
    out_weights = torch.empty(num_tokens, TOP_K, dtype=torch.float32, device="cuda")
    stage_mxfp4_mega_moe_inputs(
        hidden, weights, ids, x, sf, idx, out_weights, begin, count
    )
    return x, sf, idx, out_weights


@pytest.fixture
def routed():
    generator = torch.Generator(device="cuda").manual_seed(3)
    hidden = torch.randn(
        24, HIDDEN, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    ids = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, generator=generator, device="cuda")[:TOP_K]
            for _ in range(24)
        ]
    ).to(torch.int32)
    weights = torch.rand(24, TOP_K, generator=generator, device="cuda")
    return hidden, ids, weights


def test_all_experts_range_keeps_the_routes(routed) -> None:
    hidden, ids, weights = routed
    _, _, idx, out_weights = _stage(hidden, ids, weights, 0, NUM_EXPERTS)
    assert torch.equal(idx, ids.long())
    assert torch.equal(out_weights, weights)


@pytest.mark.parametrize(("begin", "count"), [(0, 16), (16, 16), (8, 8), (24, 8)])
def test_other_ranks_experts_become_masked_slots(routed, begin, count) -> None:
    hidden, ids, weights = routed
    _, _, idx, out_weights = _stage(hidden, ids, weights, begin, count)
    local = (ids >= begin) & (ids < begin + count)
    assert torch.equal(idx, torch.where(local, ids.long() - begin, -1))
    # The weights of masked slots stay: the kernel ignores a slot with no expert.
    assert torch.equal(out_weights, weights)


def test_range_does_not_change_the_quantized_rows(routed) -> None:
    hidden, ids, weights = routed
    x_all, sf_all, _, _ = _stage(hidden, ids, weights, 0, NUM_EXPERTS)
    x_half, sf_half, _, _ = _stage(hidden, ids, weights, 16, 16)
    assert torch.equal(x_all.view(torch.uint8), x_half.view(torch.uint8))
    assert torch.equal(sf_all, sf_half)


def test_masked_input_slots_stay_masked(routed) -> None:
    hidden, ids, weights = routed
    ids = ids.clone()
    ids[:, 0] = -1
    _, _, idx, _ = _stage(hidden, ids, weights, 0, NUM_EXPERTS)
    assert (idx[:, 0] == -1).all()


def test_expert_range_is_a_runtime_value(routed) -> None:
    hidden, ids, weights = routed
    # Both integer classes Triton keys on (divisible by 16 or not)
    for begin, count in ((0, 32), (8, 8)):
        _stage(hidden, ids, weights, begin, count)
    with assert_no_triton_compile(_stage_mxfp4_mega_moe_inputs_kernel):
        for begin, count in ((16, 16), (0, 16), (24, 8), (4, 4), (32, 64)):
            _stage(hidden, ids, weights, begin, count)

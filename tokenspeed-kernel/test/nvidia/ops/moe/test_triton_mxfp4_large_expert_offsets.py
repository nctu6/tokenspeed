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
"""Triton MXFP4 MoE with expert-major weights larger than 2**31 elements.

The kernels address one expert's weights as ``expert_id * rows * columns``.
At the DeepSeek V4.1 shapes (384 experts, hidden 5120, intermediate 2304) the
layer's ``w13`` holds 4.5e9 bytes and ``w2`` 2.3e9, so those products passed
2**31 for the later experts: stage 2 read before the allocation (illegal
memory access, seen under compute-sanitizer memcheck) and stage 1 read another
expert's weights. The oracle is the same call over a compact copy of the routed
experts, whose offsets never reach 2**31: the results must be bitwise equal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.triton import (
    triton_mxfp4_moe_weights,
    triton_mxfp4_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import current_platform

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 5120, 2304, 384, 6
# Experts around 2**31 / (2I * H / 2) = 182 (stage 1), 2**31 / (H * I / 2) = 364
# (stage 2), and the last one
ROUTED_EXPERTS = (0, 181, 182, 363, 364, 383)
# w13 + w2 + scales of the full layer, and the compact copy
MIN_FREE_BYTES = 9 << 30


def _weights(raw: dict[str, torch.Tensor], num_experts: int) -> torch.nn.Module:
    module = torch.nn.Module()
    for name, tensor in raw.items():
        setattr(module, name, torch.nn.Parameter(tensor, requires_grad=False))
    module.num_local_experts = num_experts
    module.ep_size = 1
    module.ep_rank = 0
    module.activation = "swiglu"
    module.swiglu_arg = SimpleNamespace(limit=10.0, alpha=None)
    module.swiglu_beta = None
    module.top_k = TOP_K
    module.hidden_size = HIDDEN
    module.intermediate_size = INTERMEDIATE
    return module


@pytest.mark.parametrize("num_tokens", [1, 20])
def test_late_experts_do_not_overflow_weight_offsets(num_tokens: int) -> None:
    if not torch.cuda.is_available() or not current_platform().is_nvidia:
        pytest.skip("requires an NVIDIA GPU")
    if torch.cuda.mem_get_info()[0] < MIN_FREE_BYTES:
        pytest.skip("needs the V4.1-sized expert weights (about 8 GB)")
    generator = torch.Generator(device="cuda").manual_seed(0)

    def draw(shape: tuple[int, ...], low: int, high: int) -> torch.Tensor:
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    raw = {
        "w13_weight": draw((NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2), 0, 256),
        "w13_weight_scale": draw(
            (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 32), 118, 121
        ),
        "w2_weight": draw((NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2), 0, 256),
        "w2_weight_scale": draw((NUM_EXPERTS, HIDDEN, INTERMEDIATE // 32), 118, 121),
    }
    experts = torch.tensor(ROUTED_EXPERTS, device="cuda")
    compact = {name: tensor[experts].contiguous() for name, tensor in raw.items()}
    plan = {"activation": "swiglu", "internal_activation_dtype": "input"}
    full, small = _weights(raw, NUM_EXPERTS), _weights(compact, len(ROUTED_EXPERTS))
    triton_mxfp4_moe_weights(plan, full)
    triton_mxfp4_moe_weights(plan, small)

    x = (
        torch.randn(num_tokens, HIDDEN, device="cuda", generator=generator) * 0.5
    ).bfloat16()
    # Every token takes all the routed experts, so each of them is read
    topk_ids = experts.to(torch.int32).repeat(num_tokens, 1)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, TOP_K, device="cuda", generator=generator), dim=-1
    )
    compact_ids = torch.arange(TOP_K, device="cuda", dtype=torch.int32).repeat(
        num_tokens, 1
    )

    actual = triton_mxfp4_precomputed_moe_apply(
        plan, x, full, None, topk_weights=topk_weights, topk_ids=topk_ids
    )
    expected = triton_mxfp4_precomputed_moe_apply(
        plan, x, small, None, topk_weights=topk_weights, topk_ids=compact_ids
    )
    torch.cuda.synchronize()
    assert torch.isfinite(expected).all()
    assert torch.equal(actual, expected)

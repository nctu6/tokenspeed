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
"""FP8-activation block-FP8 MoE leaf (`triton_fp8_block_fp8_act_precomputed_moe_apply`)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from test_triton_fp8_apply import _experts, _skip_unless_supported, _topk_ids
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process_weights
from utils import (
    assert_no_triton_compile,
    int_specialization_class,
    warm_specialization_classes,
)

_NAME = "triton_fp8_block_fp8_act_precomputed_moe_apply"


def _plan(
    internal_activation_dtype: str,
    activation: str,
    swiglu_limit: float | None,
    hidden_size: int,
    intermediate_size: int,
) -> dict:
    return kernel_moe_plan(
        "fp8",
        input_dtype=torch.bfloat16,
        activation=activation,
        routing_mode="precomputed_topk",
        ispp=intermediate_size,
        fp8_scale_block_shape=(128, 128),
        internal_activation_dtype=internal_activation_dtype,
        solution="triton",
        hidden=hidden_size,
        swiglu_form=("standard" if activation == "swiglu" else None),
        activation_clamped=swiglu_limit is not None,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        ep_token_layout=None,
        features=frozenset(),
    )


def _quantize_rows(t: torch.Tensor) -> torch.Tensor:
    """E4M3 round trip with one FP32 scale per row and 128 columns."""
    groups = t.float().reshape(*t.shape[:-1], t.shape[-1] // 128, 128)
    scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(t.shape)


def _reference(
    x, topk_ids, topk_weights, w13, w2, swiglu_limit, quantize: bool
) -> torch.Tensor:
    num_experts = w13.shape[0]
    expected = torch.zeros(x.shape, device="cuda", dtype=torch.float32)
    for expert_id in range(num_experts):
        token_ids, slots = torch.where(topk_ids == expert_id)
        rows = _quantize_rows(x[token_ids]) if quantize else x[token_ids].float()
        gate, up = F.linear(rows, w13[expert_id]).chunk(2, dim=-1)
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(-swiglu_limit, swiglu_limit)
        intermediate = (F.silu(gate) * up).to(torch.bfloat16)
        if quantize:
            intermediate = _quantize_rows(intermediate)
        expert_output = F.linear(intermediate.float(), w2[expert_id]).to(torch.bfloat16)
        expected.index_add_(
            0, token_ids, expert_output.float() * topk_weights[token_ids, slots, None]
        )
    return expected


# 4 experts, top-2: below 64 routed rows per expert (tokens < 128) the leaf runs
# the BF16-multiply kernels; 130 tokens use the 64-row FP8 tile, 520 the 128-row one.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("num_tokens", [5, 33, 130, 520])
@pytest.mark.parametrize(
    ("activation", "swiglu_limit"), [("silu", None), ("swiglu", 1.5)]
)
def test_fp8_act_moe_matches_reference(
    num_tokens: int, activation: str, swiglu_limit: float | None
) -> None:
    _skip_unless_supported()
    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 384, 256
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weights, w13, w2 = _experts(
        num_experts, top_k, hidden_size, intermediate_size, generator
    )
    topk_ids = _topk_ids(num_tokens, num_experts, top_k, generator)
    topk_ids[0, 1] = -1
    topk_ids[1, 0] = num_experts
    topk_weights = torch.rand(
        num_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )
    if swiglu_limit is not None:
        weights.swiglu_arg = SimpleNamespace(alpha=None, limit=swiglu_limit)
    plan = _plan("fp8", activation, swiglu_limit, hidden_size, intermediate_size)
    assert plan["apply_kernel_name"] == _NAME
    kernel_moe_process_weights(plan, weights)
    actual = kernel_moe_apply(
        plan,
        x,
        weights,
        torch.empty((num_tokens, num_experts), device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    ).float()
    exact = _reference(x, topk_ids, topk_weights, w13, w2, swiglu_limit, False)
    if num_tokens * top_k < 64 * num_experts:
        torch.testing.assert_close(actual, exact, rtol=0.02, atol=0.01)
        return
    quantized = _reference(x, topk_ids, topk_weights, w13, w2, swiglu_limit, True)
    # A scaled value can sit on an E4M3 rounding boundary: one 2^-3 step flips
    # a few elements between this kernel and the reference.
    torch.testing.assert_close(actual, quantized, rtol=0.05, atol=0.02)
    # Against the unquantized math the error is the E4M3 activation rounding.
    assert (actual - exact).norm() / exact.norm() < 0.08
    assert (actual - exact).norm() > 0, "the FP8-activation kernels did not run"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_fp8_act_moe_token_count():
    _skip_unless_supported()
    from tokenspeed_kernel.ops.moe._triton import common as _common
    from tokenspeed_kernel.ops.moe._triton import fp8

    num_experts, top_k, hidden_size, intermediate_size = 4, 2, 256, 128
    max_tokens = 1500
    generator = torch.Generator(device="cuda").manual_seed(0)
    weights, _, _ = _experts(
        num_experts, top_k, hidden_size, intermediate_size, generator
    )
    plan = _plan("fp8", "silu", None, hidden_size, intermediate_size)
    kernel_moe_process_weights(plan, weights)
    x = torch.randn(
        max_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    topk_ids = _topk_ids(max_tokens, num_experts, top_k, generator)
    topk_weights = torch.rand(
        max_tokens, top_k, device="cuda", dtype=torch.float32, generator=generator
    )

    def run(tokens):
        return kernel_moe_apply(
            plan,
            x[:tokens],
            weights,
            torch.empty((tokens, num_experts), device="cuda"),
            topk_weights=topk_weights[:tokens],
            topk_ids=topk_ids[:tokens],
        )

    def key(tokens):
        routes = tokens * top_k
        counts = (
            tokens,
            routes,
            _common._num_programs(x.device, routes, intermediate_size, 128),
            _common._num_programs(x.device, routes, hidden_size, 128),
        )
        return (
            *map(int_specialization_class, counts),
            fp8._fp8_act_is_compute_bound(routes, num_experts),
            fp8._fp8_act_row_tile(routes, num_experts),
            routes <= 128,
        )

    sweep = (3, 7, 12, 24, 100, 300, 1000, 1483)
    warm_specialization_classes(run, key, sweep, range(1, max_tokens))
    with assert_no_triton_compile(
        fp8._quantize_rows_kernel,
        fp8._stage1_fp8_act_kernel,
        fp8._stage2_fp8_act_kernel,
        _common._routing_kernel,
        _common._combine_kernel,
    ):
        for tokens in sweep:
            run(tokens)


def test_fp8_act_tiles_follow_rows_per_expert() -> None:
    from tokenspeed_kernel.ops.moe._triton.fp8 import (
        _fp8_act_is_compute_bound,
        _fp8_act_row_tile,
    )

    experts = 256
    assert [
        _fp8_act_is_compute_bound(routes, experts) for routes in (8, 16383, 16384)
    ] == [False, False, True]
    assert [_fp8_act_row_tile(routes, experts) for routes in (16384, 65535, 65536)] == [
        64,
        64,
        128,
    ]

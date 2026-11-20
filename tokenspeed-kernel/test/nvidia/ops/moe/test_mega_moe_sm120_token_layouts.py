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
"""SM120 MegaMoE token layouts on two ranks (``moe_plan(ep_token_layout=...)``).

``scattered``: each rank brings its own tokens and the kernel pushes them to the
experts' ranks (attention DP). ``replicated``: both ranks hold the same tokens
and each runs its own shard of the experts, one rank alone; the caller sums the
two partials (attention TP). The shard is half of the experts (EP: the other
experts' routes are masked slots) or half of every expert's intermediate width
(MoE TP). For the same tokens the layouts must agree with each other and with a
float reference, and the replicated calls must replay in a CUDA graph.

Needs two SM120 GPUs, so normal one-GPU pytest runs skip it:
``torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>``
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process_weights
from tokenspeed_kernel.ops.moe import moe_warmup as kernel_moe_warmup
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.deepgemm import deepgemm_sm120_mega_moe_max_ranks
from utils import make_mxfp4_moe_weights

NUM_EXPERTS = 16
TOP_K = 4
HIDDEN = 1024
INTERMEDIATE = 256
LIMIT = 10.0
MAX_TOKENS = 192
_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


@dataclass(frozen=True)
class _Swiglu:
    alpha: float | None
    limit: float | None


class _Experts(torch.nn.Module):
    """The attributes the MegaMoE leaf reads from a ``MoELayer``."""

    def __init__(self, raw: dict[str, torch.Tensor], rank: int, shard: str):
        """``shard``: "ep" holds half of the experts, "tp" half of every expert."""
        super().__init__()
        if shard == "ep":
            num_local = NUM_EXPERTS // 2
            lo, hi = rank * num_local, (rank + 1) * num_local
            w13, w13_scale = raw["w13_weight"][lo:hi], raw["w13_scale"][lo:hi]
            w2, w2_scale = raw["w2_weight"][lo:hi], raw["w2_scale"][lo:hi]
        else:
            num_local = NUM_EXPERTS
            half = INTERMEDIATE // 2
            rows = torch.arange(2 * INTERMEDIATE, device="cuda")
            rows = torch.cat(
                [
                    rows[rank * half : (rank + 1) * half],
                    rows[INTERMEDIATE + rank * half : INTERMEDIATE + (rank + 1) * half],
                ]
            )
            w13, w13_scale = raw["w13_weight"][:, rows], raw["w13_scale"][:, rows]
            # K of the down projection is packed two per byte, scaled per 32
            w2 = raw["w2_weight"][:, :, rank * half // 2 : (rank + 1) * half // 2]
            w2_scale = raw["w2_scale"][
                :, :, rank * half // 32 : (rank + 1) * half // 32
            ]
        self.w13_weight = torch.nn.Parameter(w13.clone(), False)
        self.w13_weight_scale = torch.nn.Parameter(w13_scale.clone(), False)
        self.w2_weight = torch.nn.Parameter(w2.clone(), False)
        self.w2_weight_scale = torch.nn.Parameter(w2_scale.clone(), False)
        self.hidden_size = HIDDEN
        self.intermediate_size = INTERMEDIATE
        self.top_k = TOP_K
        self.num_experts = NUM_EXPERTS
        self.num_local_experts = num_local
        self.ep_rank = rank if shard == "ep" else 0
        self.ep_size = 2 if shard == "ep" else 1
        self.tp_size = 2 if shard == "tp" else 1
        self.swiglu_arg = _Swiglu(alpha=None, limit=LIMIT)


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``[E, N, K / 2]`` packed E2M1 and ``[E, N, K / 32]`` UE8M0 to float."""
    lut = torch.tensor(_FP4, device=weight.device)
    codes = torch.stack((weight & 0xF, weight >> 4), dim=-1).flatten(-2).long()
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=-1)


def _reference(x, raw, topk_ids, topk_weights) -> torch.Tensor:
    w13 = _dequant(raw["w13_weight"], raw["w13_scale"])
    w2 = _dequant(raw["w2_weight"], raw["w2_scale"])
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(TOP_K):
            expert = int(topk_ids[token, slot])
            gate_up = w13[expert] @ x[token].float()
            gate = gate_up[:INTERMEDIATE].clamp(max=LIMIT)
            up = gate_up[INTERMEDIATE:].clamp(-LIMIT, LIMIT)
            act = gate * torch.sigmoid(gate) * up
            out[token] += topk_weights[token, slot] * (w2[expert] @ act)
    return out


def _plan_and_experts(raw, rank, layout, shard, group):
    experts = _Experts(raw, rank, shard)
    plan = kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ep_size=experts.ep_size,
        ispp=INTERMEDIATE // experts.tp_size,
        hidden=HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        internal_activation_dtype="input",
        process_group=group,
        persistent_max_num_tokens_per_gpu=MAX_TOKENS,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        solution="mega_moe",
        ep_token_layout=layout,
    )
    assert plan["apply_kernel_name"] == "deep_gemm_sm120_mxfp4_mega_moe_apply"
    kernel_moe_process_weights(plan, experts)
    kernel_moe_warmup(plan, experts, max_num_tokens=MAX_TOKENS)
    return plan, experts


def _routes(num_tokens: int, generator: torch.Generator, experts_of: range | None):
    """Distinct experts per token, all from ``experts_of`` when it is given."""
    pool = torch.arange(NUM_EXPERTS) if experts_of is None else torch.tensor(experts_of)
    ids = torch.stack(
        [
            pool[torch.randperm(len(pool), generator=generator)[:TOP_K]]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, TOP_K, generator=generator)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids.to(torch.int32).cuda(), weights.cuda()


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = actual.float() - expected.float()
    return float(diff.norm() / expected.float().norm().clamp(min=1e-6))


@pytest.mark.skipif(_world_size() != 2, reason="launch with torchrun world size 2")
@pytest.mark.skipif(
    current_platform().arch_version.major != 12
    or deepgemm_sm120_mega_moe_max_ranks() < 2,
    reason="needs SM120 and a DeepGEMM with the multi-rank SM120 MegaMoE",
)
def test_replicated_tokens_match_scattered_tokens_and_a_float_reference() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    generator = torch.Generator(device="cuda").manual_seed(20261112)
    raw = make_mxfp4_moe_weights(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, generator, scale_range=(122, 123)
    )

    # The one-rank group of this rank; every rank creates every group, in order.
    solo_groups = [dist.new_group([r]) for r in range(2)]
    scattered_plan, scattered_experts = _plan_and_experts(
        raw, rank, "scattered", "ep", dist.group.WORLD
    )
    replicated_plan, replicated_experts = _plan_and_experts(
        raw, rank, "replicated", "ep", solo_groups[rank]
    )
    tp_plan, tp_experts = _plan_and_experts(
        raw, rank, "replicated", "tp", solo_groups[rank]
    )

    # One token set on both ranks, with routes of every kind: spread over both
    # ranks' experts, all on rank 0, all on rank 1 (the other rank's call has no
    # route and must return zeros).
    routing = torch.Generator().manual_seed(7)
    x = (torch.randn(MAX_TOKENS, HIDDEN, generator=generator, device="cuda") * 0.5).to(
        torch.bfloat16
    )
    half = NUM_EXPERTS // 2
    ids_a, weights_a = _routes(MAX_TOKENS - 64, routing, None)
    ids_b, weights_b = _routes(32, routing, range(0, half))
    ids_c, weights_c = _routes(32, routing, range(half, NUM_EXPERTS))
    topk_ids = torch.cat([ids_a, ids_b, ids_c])
    topk_weights = torch.cat([weights_a, weights_b, weights_c])

    def run(plan, experts, tokens, ids, weights):
        return kernel_moe_apply(
            plan,
            tokens,
            experts,
            None,
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=tokens.shape[0],
            max_num_tokens_per_gpu=tokens.shape[0],
        )

    def replicated(
        tokens, ids, weights, plan=replicated_plan, experts=replicated_experts
    ):
        partial = run(plan, experts, tokens, ids, weights)
        dist.all_reduce(partial)
        return partial

    # Scattered: each rank takes half of the tokens and gets their full output.
    count = MAX_TOKENS // 2
    mine = slice(rank * count, (rank + 1) * count)
    own = run(
        scattered_plan, scattered_experts, x[mine], topk_ids[mine], topk_weights[mine]
    )
    gathered = [torch.empty_like(own) for _ in range(2)]
    dist.all_gather(gathered, own)
    scattered_out = torch.cat(gathered)

    replicated_out = replicated(x, topk_ids, topk_weights)
    tp_out = replicated(x, topk_ids, topk_weights, tp_plan, tp_experts)
    expected = _reference(x, raw, topk_ids, topk_weights)

    # The sum of the two partials differs from the in-kernel FP32 combine by the
    # BF16 rounding of each partial (half an ulp of the partial, so near a
    # cancellation the error is large against the result but not the tensor).
    assert _relative_error(replicated_out, scattered_out) < 2**-8
    peak = float(scattered_out.float().abs().max())
    assert float((replicated_out.float() - scattered_out.float()).abs().max()) < (
        2**-6 * peak
    )
    # The TP shard sums each expert's two halves of the down projection, and
    # quantizes the same activations: it differs from the EP shard by BF16
    # rounding of the partials too.
    assert _relative_error(tp_out, scattered_out) < 2**-7
    assert float((tp_out.float() - scattered_out.float()).abs().max()) < 2**-6 * peak
    # All are FP8-activation kernels against a float reference.
    assert _relative_error(replicated_out, expected) < 0.06
    assert _relative_error(tp_out, expected) < 0.06
    assert _relative_error(scattered_out, expected) < 0.06

    # Rank 1's experts alone serve the third block: rank 0's partial is zero.
    only_remote = slice(MAX_TOKENS - 32, MAX_TOKENS)
    partial = run(
        replicated_plan,
        replicated_experts,
        x[only_remote],
        topk_ids[only_remote],
        topk_weights[only_remote],
    )
    if rank == 0:
        assert not partial.float().abs().any()
    else:
        assert partial.float().abs().max() > 0

    # Deterministic, and a captured graph replays with new inputs.
    assert torch.equal(replicated(x, topk_ids, topk_weights), replicated_out)
    assert torch.equal(
        replicated(x, topk_ids, topk_weights, tp_plan, tp_experts), tp_out
    )
    for plan, experts in (
        (replicated_plan, replicated_experts),
        (tp_plan, tp_experts),
    ):
        graph = torch.cuda.CUDAGraph()
        static_x = x[:48].clone()
        static_ids = topk_ids[:48].clone()
        static_weights = topk_weights[:48].clone()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            static_out = run(plan, experts, static_x, static_ids, static_weights)
        static_x.copy_(x[48:96])
        static_ids.copy_(topk_ids[48:96])
        static_weights.copy_(topk_weights[48:96])
        graph.replay()
        eager = run(plan, experts, x[48:96], topk_ids[48:96], topk_weights[48:96])
        assert torch.equal(static_out, eager)
    dist.barrier()


def test_mega_moe_plan_rejects_unknown_token_layout() -> None:
    with pytest.raises(ValueError, match="ep_token_layout"):
        kernel_moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            routing_mode="precomputed_topk",
            ep_size=2,
            ispp=INTERMEDIATE,
            hidden=HIDDEN,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            swiglu_form="standard",
            activation_clamped=True,
            expert_id_repeats=False,
            internal_activation_dtype="input",
            persistent_max_num_tokens_per_gpu=MAX_TOKENS,
            fast_math=True,
            combine_order="rank",
            features=frozenset(),
            solution="mega_moe",
            ep_token_layout="gathered",
        )

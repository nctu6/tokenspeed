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
"""``--moe-backend auto`` for DeepSeek V4.1: when it resolves to MegaMoE."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import pytest  # noqa: E402
from tokenspeed_kernel.ops.moe import MEGA_MOE_AUTO_FEATURE  # noqa: E402
from tokenspeed_kernel.registry import KernelRegistry  # noqa: E402
from transformers import PretrainedConfig  # noqa: E402

from tokenspeed.runtime.configs.model_config import (  # noqa: E402
    deepseek_v41_default_moe_backend,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

_LEAF = "deep_gemm_sm120_mxfp4_mega_moe_apply"


def _config(**overrides) -> PretrainedConfig:
    values = dict(
        hidden_size=5120,
        moe_intermediate_size=2304,
        n_routed_experts=128,
        num_experts_per_tok=6,
        swiglu_limit=10.0,
        expert_dtype="fp4",
    )
    return PretrainedConfig(**{**values, **overrides})


def _args(**kwargs) -> ServerArgs:
    values = dict(world_size=2, attn_tp_size=2)
    return ServerArgs(
        model="unused",
        disable_kvstore=True,
        enable_prefix_caching=False,
        **{**values, **kwargs},
    )


@pytest.fixture(scope="module")
def expected_default() -> str | None:
    import tokenspeed_kernel.ops.moe  # noqa: F401  (registers the leaves)

    spec = KernelRegistry.get().get_by_name(_LEAF)
    if spec is None:
        pytest.skip("no SM120 MegaMoE leaf registered here")
    return "mega_moe" if MEGA_MOE_AUTO_FEATURE in spec.features else None


def _resolve(args: ServerArgs, config=None, quantization="fp8"):
    return deepseek_v41_default_moe_backend(args, config or _config(), quantization)


def test_attention_tp2_with_moe_tp2_follows_the_leafs_declared_feature(
    expected_default,
) -> None:
    assert _resolve(_args()) == expected_default


def test_attention_tp2_with_ep2_follows_the_leafs_declared_feature(
    expected_default,
) -> None:
    assert _resolve(_args(ep_size=2)) == expected_default


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attn_tp_size": 1, "data_parallel_size": 2, "ep_size": 2},
        {"all2all_backend": "deepep"},
        {"moe_mxfp4_fp8_activation": True},
        {
            "ep_size": 2,
            "ep_num_redundant_experts": 2,
            "ep_dispatch_algorithm": "static",
        },
        {"speculative_algorithm": "EAGLE"},
    ],
    ids=["attention-dp", "all2all", "fp8-activation", "redundant", "speculative"],
)
def test_launches_outside_the_verified_layout_stay_on_auto(
    expected_default, kwargs
) -> None:
    try:
        args = _args(**kwargs)
    except (ValueError, TypeError) as error:
        pytest.skip(f"ServerArgs refuses this launch itself: {error}")
    assert _resolve(args) is None


def test_block_fp8_routed_experts_stay_on_auto(expected_default) -> None:
    config = _config(expert_dtype="fp8")
    assert _resolve(_args(), config) is None


def test_a_bitwise_envelope_stays_on_auto(expected_default) -> None:
    try:
        args = _args(numerics="rl-bitwise", ep_size=2, moe_tp_size=1)
    except (ValueError, TypeError) as error:
        pytest.skip(f"ServerArgs refuses the envelope here: {error}")
    assert _resolve(args) is None


def test_a_config_without_a_routed_expert_count_stays_on_auto(
    expected_default,
) -> None:
    config = _config()
    del config.n_routed_experts
    assert _resolve(_args(), config) is None

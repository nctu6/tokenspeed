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
"""The gates --moe-backend auto applies before choosing the SM120 MegaMoE.

Every gate is a trait or feature of the registered leaf, checked by the
registry's own selection (``mega_moe_auto_available``).
"""

from __future__ import annotations

import pytest
from tokenspeed_kernel.ops.moe import MEGA_MOE_AUTO_FEATURE, mega_moe_auto_available
from tokenspeed_kernel.registry import KernelRegistry

_LEAF = "deep_gemm_sm120_mxfp4_mega_moe_apply"

# V4.1 at attention TP2 + MoE TP2: every rank holds all tokens and a 1152-wide
# slice of each of the 128 experts.
_V41_TP2 = dict(
    hidden=5120,
    ispp=1152,
    num_experts=128,
    top_k=6,
    ep_size=1,
    moe_ranks=2,
    ep_token_layout="replicated",
    activation_clamped=True,
    features=frozenset(),
)


@pytest.fixture(scope="module")
def leaf():
    import tokenspeed_kernel.ops.moe  # noqa: F401  (registers the leaves)

    spec = KernelRegistry.get().get_by_name(_LEAF)
    if spec is None:
        pytest.skip("no SM120 MegaMoE leaf registered here")
    return spec


def test_v41_tp2_geometry_passes_every_trait_gate(leaf) -> None:
    assert mega_moe_auto_available(**_V41_TP2)


def test_one_rank_and_ep2_shards_pass(leaf) -> None:
    assert mega_moe_auto_available(**{**_V41_TP2, "moe_ranks": 1})
    assert mega_moe_auto_available(**{**_V41_TP2, "ispp": 2304, "ep_size": 2})


@pytest.mark.parametrize(
    "change",
    [
        {"moe_ranks": 4},
        {"moe_ranks": 4, "ep_size": 4, "ispp": 2304},
        {"hidden": 5120 + 128},
        {"ispp": 1152 + 64},
        {"ep_token_layout": "gathered"},
    ],
    ids=["tp4", "ep4", "hidden-unaligned", "ispp-unaligned", "unknown-layout"],
)
def test_widths_and_geometry_outside_the_leaf_are_refused(leaf, change) -> None:
    assert not mega_moe_auto_available(**{**_V41_TP2, **change})


def test_a_bitwise_envelope_is_refused_until_the_leaf_declares_batch_invariance(
    leaf,
) -> None:
    # The gate is the leaf's declared feature, not a model-side special case.
    declared = "batch_invariant" in leaf.features
    got = mega_moe_auto_available(
        **{**_V41_TP2, "features": frozenset({"batch_invariant"})}
    )
    assert got == declared


def test_auto_offers_the_leaf_exactly_when_it_declares_the_measured_feature(
    leaf,
) -> None:
    got = mega_moe_auto_available(
        **{**_V41_TP2, "features": frozenset({MEGA_MOE_AUTO_FEATURE})}
    )
    assert got == (MEGA_MOE_AUTO_FEATURE in leaf.features)

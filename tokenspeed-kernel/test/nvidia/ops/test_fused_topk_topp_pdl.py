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

"""The fused top-k/top-p sampler as a programmatic dependent launch: every kernel
waits for a producer that releases its dependents first and publishes the inputs
late, and returns the bytes of its serial launch.

An unwaited init kernel read stale ``top_ks``/``top_ps``, and a radix pass that
started before the init kernel's counters were visible read them as pointers
(``CUDA error: misaligned address``)."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.sampling.cuda import (
    fused_topk_topp_prepare,
    fused_topk_topp_renorm,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_hopper_plus,
    reason="PDL requires an NVIDIA SM90+ device",
)

_TOP_K_DISABLED = 1 << 30


@triton.jit
def _publish_after_release(sources, targets, sizes: tl.constexpr, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.uint64, is_pure=False, pack=1
    )
    now = start
    while now - start < 2000000:  # 2 ms: dependents start long before the publish
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;",
            "=l",
            [],
            dtype=tl.uint64,
            is_pure=False,
            pack=1,
        )
    for tensor in tl.static_range(len(sizes)):
        for base in range(
            tl.program_id(0) * BLOCK, sizes[tensor], tl.num_programs(0) * BLOCK
        ):
            offsets = base + tl.arange(0, BLOCK)
            values = tl.load(
                sources[tensor] + offsets, offsets < sizes[tensor], other=0
            )
            tl.store(targets[tensor] + offsets, values, offsets < sizes[tensor])


def _publish(sources, targets):
    _publish_after_release[(8,)](
        tuple(t.view(torch.uint8) for t in sources),
        tuple(t.view(torch.uint8) for t in targets),
        tuple(t.numel() * t.element_size() for t in sources),
        BLOCK=256,
        launch_pdl=True,
    )


@pytest.mark.parametrize("rows", (1, 5, 33))
@pytest.mark.parametrize("vocab", (8191, 248320))
def test_fused_topk_topp_waits_for_published_inputs(rows: int, vocab: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(rows + vocab)
    probs = torch.softmax(
        torch.randn(rows, vocab, generator=generator, device="cuda") * 3, dim=-1
    )
    # Greedy, top-p through the radix selection, and top-k rows.
    kinds = [(1, 1.0), (_TOP_K_DISABLED, 0.8), (64, 0.9)]
    ks = torch.tensor(
        [kinds[i % 3][0] for i in range(rows)], dtype=torch.int32, device="cuda"
    )
    ps = torch.tensor(
        [kinds[i % 3][1] for i in range(rows)], dtype=torch.float32, device="cuda"
    )
    fused_topk_topp_prepare(probs.device)
    serial = fused_topk_topp_renorm(
        probs, ks, ps, enable_pdl=False, in_place_skip=False
    )
    fused_topk_topp_renorm(probs, ks, ps, enable_pdl=True, in_place_skip=False)

    # What a dependent that reads before the publish sees: every row uncut, and
    # probabilities that are NaN.
    target_probs = torch.full_like(probs, float("nan"))
    target_ks = torch.full_like(ks, _TOP_K_DISABLED)
    target_ps = torch.ones_like(ps)
    _publish((probs, ks, ps), (target_probs, target_ks, target_ps))
    got = fused_topk_topp_renorm(
        target_probs, target_ks, target_ps, enable_pdl=True, in_place_skip=False
    )
    torch.cuda.synchronize()
    assert torch.equal(got, serial)

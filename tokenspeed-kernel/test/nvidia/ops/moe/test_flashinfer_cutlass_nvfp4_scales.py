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
"""FlashInfer CUTLASS NVFP4 MoE with a shared (1,)-shaped GEMM1 activation scale.

Qwen3.5-122B-A10B-NVFP4 stores one input scale for the whole GEMM1 and one per
expert for GEMM2. The loader keeps the shared scale with shape (1,), but
``cutlass_fused_moe`` accepts only a 0-d scalar or one value per local expert,
so every layer with more than one expert failed with "fc1 act global must be
scalar or (num_experts_on_rank,)" while warming up the model.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from tokenspeed_kernel.platform import current_platform


def _requires_flashinfer_blackwell_plus() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia or not current_platform().is_blackwell_plus:
        pytest.skip("NVIDIA Blackwell or newer required")
    if find_spec("flashinfer") is None:
        pytest.skip("requires flashinfer")


class _NvFp4Experts(torch.nn.Module):
    """Expert weights laid out as ``create_nvfp4_weight_pair`` registers them."""

    def __init__(self, experts: int, hidden: int, inter: int, device: str) -> None:
        super().__init__()
        group = 16

        def param(tensor: torch.Tensor) -> torch.nn.Parameter:
            return torch.nn.Parameter(tensor.to(device), requires_grad=False)

        self.top_k = 2
        self.w13_weight = param(
            torch.randint(0, 255, (experts, 2 * inter, hidden // 2), dtype=torch.uint8)
        )
        self.w2_weight = param(
            torch.randint(0, 255, (experts, hidden, inter // 2), dtype=torch.uint8)
        )
        self.w13_weight_scale = param(
            torch.ones(experts, 2 * inter, hidden // group).to(torch.float8_e4m3fn)
        )
        self.w2_weight_scale = param(
            torch.ones(experts, hidden, inter // group).to(torch.float8_e4m3fn)
        )
        self.w13_weight_scale_2 = param(torch.full((experts, 2), 0.01))
        self.w2_weight_scale_2 = param(torch.full((experts,), 0.01))
        self.w13_input_scale = param(torch.full((1,), 0.02))
        self.w2_input_scale = param(torch.full((experts,), 0.02))


def test_shared_gemm1_activation_scale_with_several_experts() -> None:
    _requires_flashinfer_blackwell_plus()
    from tokenspeed_kernel.ops.moe import flashinfer as moe

    torch.manual_seed(0)
    experts, hidden, inter, tokens = 4, 256, 128, 3
    layer = _NvFp4Experts(experts, hidden, inter, "cuda")
    moe.flashinfer_cutlass_nvfp4_moe_weights({}, layer)
    assert layer.w13_input_scale_quant.shape == (1,)

    x = torch.randn(tokens, hidden, device="cuda").bfloat16()
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 3]], dtype=torch.int32, device="cuda")
    topk_weights = torch.full((tokens, 2), 0.5, device="cuda")
    out = moe.flashinfer_cutlass_nvfp4_moe_apply(
        {},
        x,
        layer,
        None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    assert out.shape == x.shape
    assert torch.isfinite(out).all()

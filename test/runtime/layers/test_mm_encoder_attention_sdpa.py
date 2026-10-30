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

"""The ``sdpa`` multimodal encoder attention backend (equal-length batches).

Through the real ``MultimodalEncoderAttention`` layer: the same weights and
rows give the ``triton_attn`` backend's output within bf16 rounding, each
sequence attends only to itself, and a packed varlen batch is refused.
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (  # noqa: E402
    MultimodalEncoderAttention,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

EMBED, HEADS = 128, 2  # head_dim 64, Whisper's
SEQ = 1500  # Whisper's encoder window


def _layer(backend: str, seed: int) -> MultimodalEncoderAttention:
    torch.manual_seed(seed)
    layer = MultimodalEncoderAttention(
        embed_dim=EMBED,
        num_heads=HEADS,
        mapping=Mapping(rank=0, world_size=1),
        qkv_bias=True,
        proj_bias=True,
        mm_attention_backend=backend,
    )
    layer = layer.cuda().to(torch.bfloat16)
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(0.0, 0.05)
    return layer


def _reference(layer, x: torch.Tensor) -> torch.Tensor:
    """fp32 per-sequence attention over the layer's own projections."""
    bsz, seq, _ = x.shape
    qkv = F.linear(
        x.float(), layer.qkv_proj.weight.float(), layer.qkv_proj.bias.float()
    )
    q, k, v = (
        t.view(bsz, seq, HEADS, EMBED // HEADS).transpose(1, 2)
        for t in qkv.split(EMBED, dim=-1)
    )
    out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bsz, seq, -1)
    return F.linear(out, layer.proj.weight.float(), layer.proj.bias.float())


@pytest.mark.parametrize("bsz", [1, 3])
@torch.inference_mode()
def test_sdpa_matches_triton_and_attends_within_each_sequence(bsz):
    """**Validates: Requirements 2.1, 3.1** (encoder numerics unchanged)."""
    sdpa, triton = _layer("sdpa", seed=bsz), _layer("triton_attn", seed=bsz)
    triton.load_state_dict(sdpa.state_dict())
    x = torch.randn(bsz, SEQ, EMBED, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.arange(0, (bsz + 1) * SEQ, SEQ, device="cuda", dtype=torch.int32)
    got = sdpa(x, cu_seqlens=cu_seqlens, max_seqlen=SEQ)
    without_boundaries = sdpa(x, max_seqlen=SEQ)
    want = triton(x, cu_seqlens=cu_seqlens, max_seqlen=SEQ)
    reference = _reference(sdpa, x)
    torch.testing.assert_close(got.float(), reference, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=1e-2)
    assert torch.equal(got, without_boundaries)


@torch.inference_mode()
def test_sdpa_refuses_a_packed_varlen_batch():
    """**Validates: Requirement 3.2** (a wrong layout fails loudly).

    The vision towers pack several sequences into one ``[1, total, dim]``
    row; attending that as one sequence would mix them, so it raises.
    """
    layer = _layer("sdpa", seed=0)
    x = torch.randn(1, 2 * SEQ, EMBED, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, SEQ, 2 * SEQ], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="equal-length"):
        layer(x, cu_seqlens=cu_seqlens, max_seqlen=SEQ)

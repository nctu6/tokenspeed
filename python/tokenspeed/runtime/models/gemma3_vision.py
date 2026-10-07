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

"""Gemma 3 vision tower + multimodal projector (SigLIP path).

Matches HF ``Gemma3ForConditionalGeneration`` / vLLM ``gemma3_mm``:
SigLIP vision encoder, avg-pool soft tokens, RMSNorm + linear projection
into the text embedding space. Designed for TokenSpeed's
``VisionEmbedder`` / ``EncoderSpec`` contract.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


class Gemma3MultiModalProjector(nn.Module):
    """Pool SigLIP patches to ``mm_tokens_per_image`` soft tokens and project."""

    def __init__(self, vision_hidden_size: int, text_hidden_size: int,
                 image_size: int, patch_size: int, mm_tokens_per_image: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(vision_hidden_size, text_hidden_size)
        )
        self.mm_soft_emb_norm = GemmaRMSNorm(vision_hidden_size, eps=eps)
        patches_per_image = image_size // patch_size
        tokens_per_side = int(mm_tokens_per_image**0.5)
        if tokens_per_side * tokens_per_side != mm_tokens_per_image:
            raise ValueError(
                f"mm_tokens_per_image={mm_tokens_per_image} is not a perfect square"
            )
        if patches_per_image % tokens_per_side != 0:
            raise ValueError(
                f"patches_per_image={patches_per_image} not divisible by "
                f"tokens_per_side={tokens_per_side}"
            )
        kernel = patches_per_image // tokens_per_side
        self.patches_per_image = patches_per_image
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel, stride=kernel)

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        # vision_outputs: [B, patches, vision_hidden]
        batch_size, _, seq_length = vision_outputs.shape
        reshaped = vision_outputs.transpose(1, 2).reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        pooled = self.avg_pool(reshaped).flatten(2).transpose(1, 2)
        # GemmaRMSNorm / flashinfer gemma_rmsnorm requires a contiguous 2-D
        # layout ([tokens, hidden]); 3-D [B, soft, H] can fail the stride%8 check.
        b, n, h = pooled.shape
        flat = pooled.reshape(b * n, h).contiguous()
        normed = self.mm_soft_emb_norm(flat).view(b, n, h)
        return torch.matmul(normed, self.mm_input_projection_weight).type_as(
            vision_outputs
        )


def build_gemma3_vision_tower(vision_config):
    """Construct a HF SigLIP vision tower from ``config.vision_config``."""
    try:
        from transformers import SiglipVisionModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Gemma 3 multimodal requires transformers with SiglipVisionModel"
        ) from exc
    return SiglipVisionModel(vision_config)


def gemma3_mm_tokens_per_image(config) -> int:
    return int(getattr(config, "mm_tokens_per_image", 256) or 256)


def gemma3_make_image_warmup_items(
    *,
    image_size: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> list[MultimodalDataItem]:
    """Minimal CHW image batch for encoder cudagraph / warmup."""
    pixels = torch.zeros(1, 3, image_size, image_size, dtype=dtype, device=device)
    return [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixels,
            hash=0,
        )
    ]


def _gemma3_pixel_batch(items: list[MultimodalDataItem]) -> torch.Tensor:
    """Stack CHW views from items; each item may carry ``[P, 3, H, W]`` crops."""
    views: list[torch.Tensor] = []
    for item in items:
        feat = item.feature
        if feat is None:
            raise ValueError("Gemma 3 image item missing feature tensor")
        if feat.ndim == 3:
            views.append(feat.unsqueeze(0))
        elif feat.ndim == 4:
            views.append(feat)
        else:
            raise ValueError(
                f"Gemma 3 image feature must be [3,H,W] or [P,3,H,W], got {tuple(feat.shape)}"
            )
    return torch.cat(views, dim=0)


def encode_gemma3_images(
    vision_tower: nn.Module,
    projector: Gemma3MultiModalProjector,
    items: list[MultimodalDataItem],
) -> torch.Tensor:
    """Encode a list of image items to concatenated soft-token embeddings.

    Returns ``[total_soft_tokens, text_hidden]`` with one contiguous block
    per item (``mm_tokens_per_image`` × num_patches each), matching
    VisionEmbedder scatter order (original then pan-and-scan crops).
    """
    if not items:
        raise ValueError("encode_gemma3_images requires at least one item")
    device = next(vision_tower.parameters()).device
    dtype = next(vision_tower.parameters()).dtype
    pixels = _gemma3_pixel_batch(items).to(device=device, dtype=dtype, non_blocking=True)
    # HF SiglipVisionModel returns BaseModelOutputWithPooling.
    outputs = vision_tower(pixel_values=pixels)
    hidden = outputs.last_hidden_state
    projected = projector(hidden)  # [B, mm_tokens, text_hidden]
    return projected.reshape(-1, projected.shape[-1])

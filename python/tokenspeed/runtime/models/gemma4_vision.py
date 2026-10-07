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

"""Gemma 4 vision tower + multimodal embedder (text+image).

Matches HF ``Gemma4ForConditionalGeneration`` / vLLM ``gemma4_mm``:
``Gemma4VisionModel`` (patch embedder → encoder → pooler) plus
``embed_vision`` (weightless RMSNorm → Linear) into the text embedding
space. Designed for TokenSpeed's ``VisionEmbedder`` / ``EncoderSpec``
contract. Pixel inputs arrive pre-patchified as
``[B, max_patches, patch_pixels]`` with ``image_position_ids`` of shape
``[B, max_patches, 2]`` (``(-1, -1)`` marks padding).
"""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.layernorm import RMSNormNoWeight
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


class Gemma4MultimodalEmbedder(nn.Module):
    """Project pooled vision soft tokens into LM embedding space.

    Checkpoint ships only ``embedding_projection.weight`` (no embedding
    table / pre-projection scale). HF/vLLM apply a weightless RMSNorm
    then a bias-free linear.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embedding_pre_projection_norm = RMSNormNoWeight(
            vision_hidden_size, eps=eps
        )
        self.embedding_projection = nn.Linear(
            vision_hidden_size, text_hidden_size, bias=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


def build_gemma4_vision_tower(vision_config):
    """Construct HF ``Gemma4VisionModel`` from ``config.vision_config``."""
    try:
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Gemma 4 multimodal requires transformers with Gemma4VisionModel"
        ) from exc
    return AutoModel.from_config(vision_config)


def gemma4_soft_tokens_per_image(config) -> int:
    vision = getattr(config, "vision_config", None)
    if vision is not None:
        val = getattr(vision, "default_output_length", None)
        if val is not None:
            return int(val)
    return int(getattr(config, "vision_soft_tokens_per_image", 280) or 280)


def gemma4_make_image_warmup_items(
    *,
    max_soft_tokens: int,
    patch_size: int,
    pooling_kernel_size: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> list[MultimodalDataItem]:
    """Minimal pre-patchified image batch for encoder warmup."""
    max_patches = max_soft_tokens * (pooling_kernel_size**2)
    patch_pixels = (patch_size**2) * 3
    pixels = torch.zeros(1, max_patches, patch_pixels, dtype=dtype, device=device)
    # Fill a square soft-token budget with valid (x, y); rest stay (-1, -1).
    positions = torch.full((1, max_patches, 2), -1, dtype=torch.long, device=device)
    side = int(max_soft_tokens**0.5) * pooling_kernel_size
    idx = 0
    for y in range(side):
        for x in range(side):
            if idx >= max_patches:
                break
            positions[0, idx, 0] = x
            positions[0, idx, 1] = y
            idx += 1
        if idx >= max_patches:
            break
    return [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=pixels,
            hash=0,
            model_specific_data={"image_position_ids": positions},
        )
    ]


def _gemma4_iter_views(
    items: list[MultimodalDataItem],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Expand items into per-view patch tensors.

    Each item may be a single image ``[P, D]`` / ``[1, P, D]`` or a multi-view
    batch ``[V, P, D]`` (pan-and-scan crops or video frames). Position ids
    follow the same leading shape.
    """
    pixel_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    for item in items:
        pv = item.feature
        if pv is None:
            raise ValueError("Gemma 4 image/video item missing feature tensor")
        pp = item.model_specific_data.get("image_position_ids")
        if pp is None:
            raise ValueError("Gemma 4 image/video item missing image_position_ids")
        if pv.ndim == 2:
            pv = pv.unsqueeze(0)
            pp = pp.unsqueeze(0) if pp.ndim == 2 else pp
        elif pv.ndim == 3 and pv.shape[0] == 1:
            pass  # [1, P, D]
        elif pv.ndim == 3:
            pass  # [V, P, D]
        else:
            raise ValueError(
                f"Gemma 4 feature must be [P,D], [1,P,D], or [V,P,D], got {tuple(pv.shape)}"
            )
        if pp.ndim == 2:
            pp = pp.unsqueeze(0)
        if pp.shape[0] != pv.shape[0]:
            raise ValueError(
                f"image_position_ids batch {pp.shape[0]} != feature batch {pv.shape[0]}"
            )
        for v in range(pv.shape[0]):
            pixel_list.append(pv[v])
            pos_list.append(pp[v])
    return pixel_list, pos_list


def encode_gemma4_images(
    vision_tower: nn.Module,
    embed_vision: Gemma4MultimodalEmbedder,
    items: list[MultimodalDataItem],
) -> torch.Tensor:
    """Encode image/video-frame items to concatenated soft-token embeddings.

    Returns ``[total_soft_tokens, text_hidden]`` with one contiguous block
    per item (padding patches stripped, views in order), matching
    VisionEmbedder scatter.
    """
    if not items:
        raise ValueError("encode_gemma4_images requires at least one item")
    device = next(vision_tower.parameters()).device
    dtype = next(vision_tower.parameters()).dtype
    pooling_k2 = int(getattr(vision_tower.config, "pooling_kernel_size", 3)) ** 2

    pixel_list, pos_list = _gemma4_iter_views(items)
    pixel_list = [p.to(device=device, dtype=dtype, non_blocking=True) for p in pixel_list]
    pos_list = [p.to(device=device, dtype=torch.long, non_blocking=True) for p in pos_list]

    # Drop padded patch rows before encode/pool. SMG feature_token_counts use
    # n_valid // pooling_k2; leaving max_patches padding in makes the pooler
    # emit a different soft length (breaks VisionEmbedder split under PaS).
    stripped_pixels: list[torch.Tensor] = []
    stripped_pos: list[torch.Tensor] = []
    for pv, pp in zip(pixel_list, pos_list):
        keep = ~(pp == -1).all(dim=-1)
        if not bool(keep.any()):
            raise ValueError("gemma4 view has no valid patches")
        stripped_pixels.append(pv[keep])
        stripped_pos.append(pp[keep])
    pixel_list, pos_list = stripped_pixels, stripped_pos

    # Bucket by patch count so each encoder call is uniform-shape.
    buckets: dict[int, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
    for idx, (pv, pp) in enumerate(zip(pixel_list, pos_list)):
        buckets.setdefault(pv.shape[0], []).append((idx, pv, pp))

    last_hidden: dict[int, torch.Tensor] = {}
    for _patches, bucket_items in buckets.items():
        pv_tensor = torch.stack([b[1] for b in bucket_items], dim=0)
        pp_tensor = torch.stack([b[2] for b in bucket_items], dim=0)
        pad_tensor = (pp_tensor == -1).all(dim=-1)
        inputs_embeds = vision_tower.patch_embedder(
            pv_tensor, pp_tensor, pad_tensor
        ).to(dtype)
        encoder_outputs = vision_tower.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~pad_tensor,
            pixel_position_ids=pp_tensor,
        )
        for i, (orig_idx, _, _) in enumerate(bucket_items):
            last_hidden[orig_idx] = encoder_outputs.last_hidden_state[i]

    all_valid: list[torch.Tensor] = []
    valid_lens: list[int] = []
    for orig_idx in range(len(pixel_list)):
        hidden = last_hidden[orig_idx]
        pos = pos_list[orig_idx]
        output_length = hidden.shape[0] // pooling_k2
        padding_positions = (pos == -1).all(dim=-1).unsqueeze(0)
        pooled, valid_mask = vision_tower.pooler(
            hidden_states=hidden.unsqueeze(0),
            pixel_position_ids=pos.unsqueeze(0),
            padding_positions=padding_positions,
            output_length=output_length,
        )
        valid = pooled[valid_mask]
        if getattr(vision_tower.config, "standardize", False):
            valid = (valid - vision_tower.std_bias) * vision_tower.std_scale
        all_valid.append(valid)
        valid_lens.append(int(valid.shape[0]))

    flat = torch.cat(all_valid, dim=0).to(dtype)
    projected = embed_vision(flat.unsqueeze(0)).squeeze(0)
    offset = 0
    for length in valid_lens:
        offset += length
    assert offset == projected.shape[0]
    return projected


def gemma4_make_video_warmup_items(
    *,
    max_soft_tokens: int = 70,
    patch_size: int = 16,
    pooling_kernel_size: int = 3,
    num_frames: int = 2,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> list[MultimodalDataItem]:
    """Minimal multi-frame patchified batch for video encoder warmup."""
    max_patches = max_soft_tokens * (pooling_kernel_size**2)
    patch_pixels = (patch_size**2) * 3
    pixels = torch.zeros(
        num_frames, max_patches, patch_pixels, dtype=dtype, device=device
    )
    positions = torch.full(
        (num_frames, max_patches, 2), -1, dtype=torch.long, device=device
    )
    side = max(1, int(max_soft_tokens**0.5) * pooling_kernel_size)
    for f in range(num_frames):
        idx = 0
        for y in range(side):
            for x in range(side):
                if idx >= max_patches:
                    break
                positions[f, idx, 0] = x
                positions[f, idx, 1] = y
                idx += 1
            if idx >= max_patches:
                break
    return [
        MultimodalDataItem(
            modality=Modality.VIDEO,
            feature=pixels,
            hash=0,
            model_specific_data={"image_position_ids": positions},
        )
    ]


def encode_gemma4_audio(
    audio_tower: nn.Module,
    embed_audio: Gemma4MultimodalEmbedder,
    items: list[MultimodalDataItem],
) -> torch.Tensor:
    """Encode audio mel features through the HF audio tower + embed_audio.

    Each item.feature is ``[mel, frames]`` or ``[1, mel, frames]``; optional
    ``input_features_mask`` / ``audio_feature_lengths`` live in
    ``model_specific_data``.
    """
    if not items:
        raise ValueError("encode_gemma4_audio requires at least one item")
    device = next(audio_tower.parameters()).device
    dtype = next(audio_tower.parameters()).dtype

    feats: list[torch.Tensor] = []
    lengths: list[int] = []
    for item in items:
        feat = item.feature
        if feat is None:
            raise ValueError("Gemma 4 audio item missing feature")
        if feat.ndim == 3 and feat.shape[0] == 1:
            feat = feat.squeeze(0)
        if feat.ndim != 2:
            raise ValueError(
                f"Gemma 4 audio feature must be [mel, frames], got {tuple(feat.shape)}"
            )
        mask = item.model_specific_data.get("input_features_mask")
        if mask is not None:
            if mask.ndim == 2 and mask.shape[0] == 1:
                mask = mask.squeeze(0)
            length = int(mask.to(torch.bool).sum().item())
        elif "audio_feature_lengths" in item.model_specific_data:
            length = int(item.model_specific_data["audio_feature_lengths"].reshape(-1)[0])
        else:
            length = int(feat.shape[-1])
        feats.append(feat.to(device=device, dtype=dtype, non_blocking=True))
        lengths.append(max(length, 1))

    max_t = max(f.shape[-1] for f in feats)
    mel = feats[0].shape[0]
    batch = torch.zeros(len(feats), mel, max_t, device=device, dtype=dtype)
    mask_b = torch.zeros(len(feats), max_t, device=device, dtype=torch.bool)
    for i, (feat, length) in enumerate(zip(feats, lengths)):
        t = min(length, feat.shape[-1], max_t)
        batch[i, :, :t] = feat[:, :t]
        mask_b[i, :t] = True

    audio_outputs = audio_tower(batch, mask_b)
    if isinstance(audio_outputs, tuple):
        encodings, out_mask = audio_outputs
    else:
        encodings = audio_outputs.last_hidden_state
        out_mask = getattr(audio_outputs, "attention_mask", mask_b)

    projected = embed_audio(encodings)
    pieces: list[torch.Tensor] = []
    for enc, m in zip(projected, out_mask):
        pieces.append(enc[m.to(dtype=torch.bool)])
    return torch.cat(pieces, dim=0)

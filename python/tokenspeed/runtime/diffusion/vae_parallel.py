# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""Spread the MiniMax-H3 video VAE decode over the context-parallel ranks.

``AutoencoderKLMiniMaxH3._decode`` decodes the latent video as a sequence of
temporal clips that do not depend on each other (``_decode_clip``) and then
cross-fades neighbouring clips. Under context parallel every rank holds the
same final latents and used to decode the whole video redundantly. Here rank
``r`` decodes clips ``r, r + world, ...``, the clips are all-gathered, and the
unchanged Diffusers ``_decode`` runs its blending over the gathered clips, so
the output is the single-GPU decode, assembled from the same per-clip results.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

__all__ = ["ClipParallelVideoDecode"]


class ClipParallelVideoDecode:
    """Wraps one video VAE; ``enabled`` is set per dispatch (CP on / off)."""

    def __init__(self, vae: Any, group: Any):
        self.vae = vae
        self.group = group
        self.enabled = False
        self.original_decode = vae._decode
        # Instance attribute shadows the class method for this VAE only.
        vae._decode = self._decode

    def clip_inputs(self, z: torch.Tensor) -> list[torch.Tensor]:
        """The latent slices Diffusers' ``_decode`` passes to ``_decode_clip``."""
        vae = self.vae
        tokens_chunk_size = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(
            token_drop > 0
        )
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)
        return [
            z[
                :,
                :,
                i * tokens_chunk_size : i * tokens_chunk_size
                + tokens_chunk_size
                + vae.token_overlap,
            ]
            for i in range(num_chunks)
        ]

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        world = dist.get_world_size(self.group) if self.enabled else 1
        clips = self.clip_inputs(z) if world > 1 else []
        if (
            world <= 1
            or len(clips) < world
            or len({tuple(c.shape) for c in clips}) != 1
        ):
            return self.original_decode(z)

        rank = dist.get_rank(self.group)
        decode_clip = type(self.vae)._decode_clip.__get__(self.vae)
        own = {
            i: decode_clip(clips[i]).contiguous()
            for i in range(rank, len(clips), world)
        }
        template = next(iter(own.values()))
        decoded: list[torch.Tensor | None] = [None] * len(clips)
        for base in range(0, len(clips), world):
            mine = own.get(base + rank)
            send = mine if mine is not None else torch.empty_like(template)
            parts = [torch.empty_like(template) for _ in range(world)]
            dist.all_gather(parts, send, group=self.group)
            for offset, part in enumerate(parts):
                if base + offset < len(clips):
                    decoded[base + offset] = part

        served = iter(range(len(clips)))

        def _gathered_clip(z_clip: torch.Tensor) -> torch.Tensor:
            index = next(served)
            # Fail closed if Diffusers ever slices the clips differently.
            if z_clip.shape != clips[index].shape or not torch.equal(
                z_clip, clips[index]
            ):
                raise RuntimeError("clip-parallel decode: clip layout mismatch")
            return decoded[index]

        self.vae._decode_clip = _gathered_clip
        try:
            return self.original_decode(z)
        finally:
            del self.vae._decode_clip

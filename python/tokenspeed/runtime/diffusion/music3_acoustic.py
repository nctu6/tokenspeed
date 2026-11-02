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

"""Windowed acoustic stage of MiniMax-Music3, batched across songs.

Same math as Diffusers ``MiniMaxMusic3ChunkDenoiseStep`` + the vocoder crop of
``MiniMaxMusic3VocoderDecodeStep``: 200-frame windows on a 100-frame hop, each
window flow-matched with the previous window's trailing latents as an overlap
prompt, then vocoded and cropped so the kept spans tile the song. A window
only depends on its own song's previous window, so windows of different songs
that are ready at the same time run through the DiT as one batch (conditional
and unconditional passes in the same forward).

A window can start as soon as the AR stage has produced its frames, which is
what lets acoustic work overlap the AR stage.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from tokenspeed.runtime.diffusion.music3_stream import (
    ACOUSTIC_CHUNK_FRAMES as CHUNK_FRAMES,
)
from tokenspeed.runtime.diffusion.music3_stream import ACOUSTIC_CHUNK_HOP as CHUNK_HOP
from tokenspeed.runtime.diffusion.music3_stream import (
    CROP_LEFT_LATENT,
    CROP_RIGHT_LATENT,
)
from tokenspeed.runtime.diffusion.music3_stream import (
    acoustic_chunk_starts as window_starts,
)

__all__ = ["AcousticSong", "Music3Acoustic"]

# Diffusers minimax_music3/denoise.py: the carried overlap and the guider scale.
OVERLAP_LATENT_LENGTH = 172
CFG_GUIDANCE_SCALE = 1.7


@dataclass
class AcousticSong:
    """Acoustic state of one song on one acoustic device."""

    seed: int | None
    num_inference_steps: int
    # Noise generator for the window draws. A CPU generator keeps the song
    # movable between acoustic devices (the draws do not depend on where the
    # window runs).
    generator: torch.Generator | None
    frames: list[torch.Tensor] = field(
        default_factory=list
    )  # [1, 8 * hidden] each, AR device
    frames_available: int = 0
    total_frames: int | None = None  # set when the AR stage finished the song
    next_window: int = 0
    # Overlap carry of the previous window (on the device that ran it).
    previous_latent: torch.Tensor | None = None
    previous_condition: torch.Tensor | None = None
    finished: bool = False
    # A window of this song is running; its next window waits for it.
    in_flight: bool = False

    def ready_window(self) -> tuple[int, int, bool] | None:
        """``(start, end, is_last)`` of the next window if its frames exist."""
        if self.finished or self.in_flight:
            return None
        start = self.next_window * CHUNK_HOP
        if self.total_frames is not None:
            starts = window_starts(self.total_frames)
            if self.next_window >= len(starts):
                return None
            end = min(start + CHUNK_FRAMES, self.total_frames)
            return start, end, self.next_window == len(starts) - 1
        # More frames than the window covers: the song continues past it, so
        # it is neither the last window nor shorter than CHUNK_FRAMES.
        if self.frames_available > start + CHUNK_FRAMES:
            return start, start + CHUNK_FRAMES, False
        return None


class Music3Acoustic:
    """Condition encoder + DiT + vocoder on one device."""

    def __init__(
        self,
        condition_encoder,
        transformer,
        vocoder,
        scheduler,
        *,
        device: str,
        latent_hop: int,
        low_priority: bool,
    ):
        """``low_priority`` runs the windows on the device's lowest-priority
        stream, so a latency-critical stage sharing the GPU (the AR stage) gets
        the SMs first."""
        self.condition_encoder = condition_encoder
        self.transformer = transformer
        self.vocoder = vocoder
        self.scheduler = copy.deepcopy(scheduler)
        self.device = torch.device(device)
        self.latent_hop = int(latent_hop)
        self.num_channels_latents = int(transformer.config.in_channels)
        self.stream = None
        if low_priority:
            lowest, _ = torch.cuda.Stream.priority_range()
            self.stream = torch.cuda.Stream(self.device, priority=lowest)

    def run_windows(
        self, items: list[tuple["AcousticSong", int, int, bool]]
    ) -> list[tuple[np.ndarray, bool]]:
        """Denoise + vocode one window per song (see ``_run_windows``)."""
        if self.stream is None:
            return self._run_windows(items)
        with torch.cuda.stream(self.stream):
            return self._run_windows(items)

    @torch.no_grad()
    def _run_windows(
        self, items: list[tuple[AcousticSong, int, int, bool]]
    ) -> list[tuple[np.ndarray, bool]]:
        """Denoise + vocode one window per song.

        Returns ``(float32 [2, samples] cropped waveform, is_last)`` per item.

        Items must share the condition length and step count (caller groups).
        """
        steps = items[0][0].num_inference_steps
        conditions, latents, overlaps, noise_prompts = [], [], [], []
        for song, start, end, _ in items:
            hidden = torch.cat(song.frames[start:end], dim=0)[None].to(self.device)
            condition = self.condition_encoder(hidden.to(self.condition_encoder.dtype))
            condition = condition.to(self.transformer.dtype)
            if song.previous_latent is not None:
                # The previous window may have run on another acoustic device.
                song.previous_latent = song.previous_latent.to(self.device)
                song.previous_condition = song.previous_condition.to(self.device)
            overlap = 0
            if song.previous_latent is not None:
                overlap = min(song.previous_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = song.previous_condition[:, :overlap]
            noise = _randn(
                (1, self.num_channels_latents, condition.shape[1]),
                song.generator,
                self.device,
                condition.dtype,
            )
            noise_prompts.append(noise[..., :overlap].clone() if overlap > 0 else None)
            conditions.append(condition)
            latents.append(noise)
            overlaps.append(overlap)
        condition = torch.cat(conditions, dim=0)
        x = torch.cat(latents, dim=0)
        batch = x.shape[0]
        guidance_condition = torch.cat((condition, torch.zeros_like(condition)), dim=0)

        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=self.device)
        for t in self.scheduler.timesteps:
            for b, (song, *_rest) in enumerate(items):
                overlap = overlaps[b]
                if overlap > 0:
                    time_value = t.to(x.dtype)
                    x[b : b + 1, :, :overlap] = (
                        1.0 - (1.0 - 1e-6) * time_value
                    ) * noise_prompts[b] + time_value * song.previous_latent[
                        ..., :overlap
                    ]
            timestep = t.expand(2 * batch).to(x.dtype)
            prediction = self.transformer(
                hidden_states=torch.cat((x, x), dim=0),
                timestep=timestep,
                encoder_hidden_states=guidance_condition,
                return_dict=False,
            )[0]
            cond, uncond = prediction[:batch], prediction[batch:]
            velocity = uncond + CFG_GUIDANCE_SCALE * (cond - uncond)
            x = self.scheduler.step(velocity, t, x, return_dict=False)[0]

        # Update step: restore the previous window over the overlap (the
        # vocoder sees the restored latents), then carry this window's tail.
        for b, (song, *_rest) in enumerate(items):
            if overlaps[b] > 0:
                x[b : b + 1, :, : overlaps[b]] = song.previous_latent[
                    ..., : overlaps[b]
                ]
        length = x.shape[-1]
        lo = max(0, length - 2 * OVERLAP_LATENT_LENGTH)
        hi = max(lo, length - OVERLAP_LATENT_LENGTH)
        for b, (song, *_rest) in enumerate(items):
            song.previous_latent = x[b : b + 1, :, lo:hi]
            song.previous_condition = condition[b : b + 1, lo:hi]

        waveforms = self.vocoder(x.to(self.vocoder.dtype))
        outputs = []
        for b, (song, _start, _end, is_last) in enumerate(items):
            waveform = waveforms[b]
            left = 0 if song.next_window == 0 else CROP_LEFT_LATENT * self.latent_hop
            right = 0 if is_last else CROP_RIGHT_LATENT * self.latent_hop
            piece = (
                waveform[..., left : waveform.shape[-1] - right]
                .float()
                .clamp(-1.0, 1.0)
            )
            outputs.append((piece.cpu().numpy(), is_last))
            song.next_window += 1
            if is_last:
                song.finished = True
                song.frames = []
        return outputs


def _randn(shape, generator, device, dtype) -> torch.Tensor:
    from diffusers.utils.torch_utils import randn_tensor

    return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

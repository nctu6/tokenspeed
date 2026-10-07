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

"""MiniMax-Music3 serve knobs and duration / frame helpers.

Diffusers ``MiniMaxMusic3ModularPipeline`` generates 44.1 kHz stereo; the
reference Omni / SGLang servers additionally resample to 32 kHz. TokenSpeed
keeps the Diffusers native rate by default and optionally resamples when
``output_sample_rate=32000``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed.runtime.diffusion.family import resolve_modular_root

__all__ = [
    "Music3Config",
    "AUDIO_FRAME_RATE",
    "MAX_AUDIO_FRAMES",
    "NATIVE_SAMPLE_RATE",
    "REFERENCE_SAMPLE_RATE",
    "frames_for_duration",
    "duration_for_frames",
    "resolve_music3_checkpoint",
]

# AR stage emits 25 semantic frames per second of audio (Diffusers docs).
AUDIO_FRAME_RATE = 25
# Checkpoint ceiling: 9000 frames ~= six minutes.
MAX_AUDIO_FRAMES = 9000
MIN_AUDIO_FRAMES = 25  # 1 second
NATIVE_SAMPLE_RATE = 44100
REFERENCE_SAMPLE_RATE = 32000
DEFAULT_AUDIO_DURATION_S = 30.0
DEFAULT_NUM_INFERENCE_STEPS = 30


@dataclass(frozen=True)
class Music3Config:
    """Serve-time knobs for one MiniMax-Music3 process."""

    model_path: str
    served_model_name: str = "test"
    num_gpus: int = 1
    dtype: str = "bfloat16"
    memory_reserve_margin: str = "8GB"
    enable_cpu_offload: bool = True
    # device_split: semantic_generator on cuda:1, denoise+decode on cuda:0
    # when num_gpus>=2. Auto picks split when num_gpus>=2.
    residency: str = "auto"  # auto | single | device_split
    # WAV sample rate written to clients. None / 0 → Diffusers native 44100.
    output_sample_rate: int = REFERENCE_SAMPLE_RATE

    def resolve_residency(self) -> str:
        mode = (self.residency or "auto").lower()
        if mode == "auto":
            return "device_split" if int(self.num_gpus) >= 2 else "single"
        if mode in {"single", "device_split"}:
            if mode == "device_split" and int(self.num_gpus) < 2:
                raise ValueError("residency=device_split requires num_gpus>=2")
            return mode
        raise ValueError(
            f"residency={self.residency!r} must be auto, single, or device_split"
        )


def frames_for_duration(duration_s: float) -> int:
    """Map a seconds upper-bound onto an AR frame budget (25 fps)."""
    if duration_s <= 0:
        raise ValueError(f"audio_duration={duration_s} must be positive")
    frames = int(round(float(duration_s) * AUDIO_FRAME_RATE))
    frames = max(MIN_AUDIO_FRAMES, min(MAX_AUDIO_FRAMES, frames))
    return frames


def duration_for_frames(frames: int) -> float:
    frames = max(MIN_AUDIO_FRAMES, min(MAX_AUDIO_FRAMES, int(frames)))
    return frames / float(AUDIO_FRAME_RATE)


def resolve_music3_checkpoint(model_path: str) -> str:
    """Validate and return the Music3 modular root."""
    from tokenspeed.runtime.diffusion.family import detect_diffusion_family

    root = resolve_modular_root(model_path)
    family = detect_diffusion_family(root)
    if family != "music3":
        raise ValueError(
            f"{root} is diffusion family {family!r}, not music3 "
            "(expected MiniMaxMusic3ModularPipeline)"
        )
    return root

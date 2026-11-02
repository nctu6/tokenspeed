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

"""CPU-only: the MP4 of one H3 clip has a stereo track as long as the video."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")
pytest.importorskip("diffusers")

from tokenspeed.runtime.diffusion.pipeline import (  # noqa: E402
    mux_mp4,
    single_clip_outputs,
)

FPS, FRAMES, RATE = 24, 24, 32000  # one second of video and audio


def _pipeline_outputs() -> dict:
    # Shapes as the Diffusers MiniMax-H3 decode step returns them:
    # videos is a batch of [frames, H, W, 3]; audio is (1, 2, num_samples).
    rng = np.random.default_rng(0)
    video = rng.integers(0, 255, (FRAMES, 64, 96, 3), dtype=np.uint8)
    t = torch.arange(RATE) / RATE
    left, right = torch.sin(2 * torch.pi * 440 * t), torch.sin(2 * torch.pi * 660 * t)
    audio = 0.5 * torch.stack([left, right])[None]
    return {"videos": [video], "audio": audio, "sampling_rate": RATE}


def test_single_clip_drops_the_batch_axis():
    video, audio, rate = single_clip_outputs(_pipeline_outputs())
    assert video.shape == (FRAMES, 64, 96, 3)
    assert tuple(audio.shape) == (2, RATE)
    assert rate == RATE


def test_mux_keeps_audio_duration_and_stereo():
    # Regression: the batched (1, 2, N) tensor produced a 2x-long track.
    video, audio, rate = single_clip_outputs(_pipeline_outputs())
    data = mux_mp4(video, audio, rate, fps=FPS)
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        samples = sum(frame.samples for frame in container.decode(stream))
        channels = stream.codec_context.channels
    assert channels == 2
    # AAC adds up to a couple of frames of priming/padding, never 2x.
    assert abs(samples / RATE - FRAMES / FPS) < 0.1


def test_mono_or_unbatched_soundtrack_fails_closed():
    outputs = _pipeline_outputs()
    outputs["audio"] = outputs["audio"][:, :1]  # (1, 1, N): not stereo
    with pytest.raises(ValueError, match="stereo"):
        single_clip_outputs(outputs)

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

"""TokenSpeed diffusion runtime (image/video generation, not LLM decode).

Small parallel surface to the generative engine: no paged KV, no autoregressive
decode. The first vertical slice wraps released ``diffusers`` ModularPipeline
for MiniMax-H3. Diffusers deps ship via ``tokenspeed[diffusion]``; ``./scripts/install.sh`` installs them by default.

Heavy imports (pipeline / job) are lazy so ``config`` stays CPU-importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "H3Config",
    "H3Pipeline",
    "VideoJobQueue",
    "resolve_diffusion_checkpoint",
    "AUDIO_SIGMA_SHIFT",
    "VIDEO_SIGMA_SHIFT",
]

from tokenspeed.runtime.diffusion.config import (
    AUDIO_SIGMA_SHIFT,
    H3Config,
    VIDEO_SIGMA_SHIFT,
    resolve_diffusion_checkpoint,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.diffusion.job import VideoJobQueue
    from tokenspeed.runtime.diffusion.pipeline import H3Pipeline


def __getattr__(name: str):
    if name == "H3Pipeline":
        from tokenspeed.runtime.diffusion.pipeline import H3Pipeline

        return H3Pipeline
    if name == "VideoJobQueue":
        from tokenspeed.runtime.diffusion.job import VideoJobQueue

        return VideoJobQueue
    raise AttributeError(name)

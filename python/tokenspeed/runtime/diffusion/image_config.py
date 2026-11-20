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
"""MiniMax-H3-Image serve-time configuration and request validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "H3ImageConfig",
    "IMAGE_FORMATS",
    "MAX_REFERENCE_IMAGES",
    "parse_image_size",
]

# The README documents "up to 16 reference images".
MAX_REFERENCE_IMAGES = 16
IMAGE_FORMATS = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}

_SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")


@dataclass(frozen=True)
class H3ImageConfig:
    """Serve-time knobs for one MiniMax-H3-Image process.

    Every field is explicit: residency and offload choose the execution path.

    ``residency``:
      * ``single``: every component on ``cuda:0`` (auto CPU offload when
        ``enable_cpu_offload``).
      * ``device_split``: the Qwen3-VL conditioner on ``cuda:1``, DiT + VAE on
        ``cuda:0``. The two weights sets are ~62 GB each, so this is the
        layout that keeps both resident on two 96 GB GPUs.
      * ``auto``: ``device_split`` when ``num_gpus >= 2``, else ``single``.
    """

    model_path: str
    served_model_name: str
    num_gpus: int
    dtype: str
    enable_cpu_offload: bool
    memory_reserve_margin: str
    residency: str

    def resolve_residency(self) -> str:
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus={self.num_gpus} must be >= 1")
        mode = self.residency.lower()
        if mode == "auto":
            return "device_split" if self.num_gpus >= 2 else "single"
        if mode == "single":
            return mode
        if mode == "device_split":
            if self.num_gpus < 2:
                raise ValueError("residency=device_split requires num_gpus>=2")
            return mode
        raise ValueError(
            f"residency={self.residency!r} must be auto, single, or device_split"
        )


def parse_image_size(size: str, multiple: int) -> tuple[int, int]:
    """Parse ``"WIDTHxHEIGHT"`` into ``(width, height)``.

    ``multiple`` is the checkpoint's canvas granularity (VAE 16x times the 2x2
    patch = 32): a side that is not a multiple is rejected rather than rounded,
    so the returned image has exactly the requested size.
    """
    match = _SIZE_RE.match(size or "")
    if match is None:
        raise ValueError(f"size={size!r} must look like '4096x4096' (WIDTHxHEIGHT)")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError(f"size={size!r} must be positive")
    if width % multiple or height % multiple:
        raise ValueError(
            f"size={size!r}: width and height must be multiples of {multiple}"
        )
    return width, height

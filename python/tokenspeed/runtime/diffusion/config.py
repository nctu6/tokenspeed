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

"""MiniMax-H3 constants and checkpoint layout resolution.

Two incompatible packagings ship in the HF repo:

* **root** ``model_index.json`` -- ``MiniMaxH3ModularPipeline`` (released
  diffusers >= 0.40). Prefer this.
* **FL2VA/** / **Ref2VA/** -- SGLang-oriented indexes whose class names are
  not in released diffusers. Pointing ``ts serve`` at a partition alone is
  redirected to the parent root when that root looks modular.

Audio sigma shift ``3.0`` is only in FL2VA metadata; the root scheduler
config only carries video ``12.0``. Record it here so it cannot silently
fall back to the video shift (that still yields a video-with-sound, with
no timing number that would catch the bug).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

__all__ = [
    "H3Config",
    "AUDIO_SIGMA_SHIFT",
    "VIDEO_SIGMA_SHIFT",
    "SUPPORTED_FPS",
    "resolve_diffusion_checkpoint",
]

VIDEO_SIGMA_SHIFT = 12.0
AUDIO_SIGMA_SHIFT = 3.0  # from FL2VA/_minimax_h3.sigma_shift_scales; not in root scheduler
SUPPORTED_FPS = 24
MIN_DURATION_S = 4.0
MAX_DURATION_S = 15.0
DEFAULT_SHORT_EDGE = 768


@dataclass(frozen=True)
class H3Config:
    """Serve-time knobs for one MiniMax-H3 process."""

    model_path: str
    workflow: str = "fl2va"  # fl2va | ref2va | t2va | all
    served_model_name: str = "test"
    num_gpus: int = 2
    dtype: str = "bfloat16"
    memory_reserve_margin: str = "12GB"
    enable_cpu_offload: bool = True
    # Parallelism (vLLM-shaped flag names; TokenSpeed residency semantics).
    # device_split: HF ModularPipeline conditioner@cuda:1 / rest@cuda:0 when
    # num_gpus>=2 and USP/ring == 1.
    # ulysses: DiT Context Parallel (Diffusers CP + TokenSpeed torchrun sync);
    #   replicates DiT weights across ranks, shards sequence in attention.
    # Weight-TP (text_encoder_tp / DiT Linear shard) is fail-closed: H3 has no
    # Diffusers _tp_plan yet.
    ulysses_degree: int = 1
    ring_degree: int = 1
    text_encoder_tp_size: int = 1
    residency: str = "auto"  # auto | single | device_split | ulysses
    diffusion_attention_backend: str | None = None

    def load_components_workflow(self) -> str | None:
        """Workflow passed to ``load_components`` (not ``from_pretrained``).

        Diffusers docs: select the DiT partition at ``load_components``.
        ``workflow="t2va"`` fetches ``transformer/`` and serves both ``t2va``
        and ``fl2va`` calls. Selecting ``fl2va`` at ``from_pretrained`` prunes
        to keyframe-only blocks and breaks pure text ``t2va`` (empty
        ``condition_rows`` cat). ``all`` loads both DiT partitions.
        """
        if self.workflow in {"fl2va", "t2va"}:
            return "t2va"
        if self.workflow == "ref2va":
            return "ref2va"
        if self.workflow == "all":
            return None
        raise ValueError(
            f"workflow={self.workflow!r} must be fl2va, t2va, ref2va, or all"
        )

    def modular_workflow(self) -> str | None:
        """Deprecated alias kept for callers; prefer ``load_components_workflow``."""
        return self.load_components_workflow()

    def resolve_residency(self) -> str:
        """Pick residency layout; fail closed on unsupported weight-TP asks."""
        if self.text_encoder_tp_size > 1:
            raise ValueError(
                "TokenSpeed diffusion does not shard the Qwen3-VL text encoder with "
                f"weight-TP (text_encoder_tp_size={self.text_encoder_tp_size}). "
                "Use residency=device_split (full conditioner on cuda:1) or "
                "ulysses_degree>1 (DiT sequence parallel with replicated weights)."
            )
        usp = max(1, int(self.ulysses_degree))
        ring = max(1, int(self.ring_degree))
        if usp > 1 and ring > 1:
            raise ValueError(
                "hybrid ulysses*ring is not enabled in TokenSpeed H3 serve yet "
                f"(ulysses={usp}, ring={ring}); pick one of --usp or --ring"
            )
        mode = (self.residency or "auto").lower()
        if mode == "auto":
            if usp > 1 or ring > 1:
                return "ulysses"
            return "device_split" if int(self.num_gpus) >= 2 else "single"
        if mode in {"single", "device_split", "ulysses"}:
            if mode == "device_split" and int(self.num_gpus) < 2:
                raise ValueError("residency=device_split requires num_gpus>=2")
            if mode == "ulysses" and usp <= 1 and ring <= 1:
                raise ValueError(
                    "residency=ulysses requires --ulysses-degree/--usp > 1 or --ring > 1"
                )
            return mode
        raise ValueError(
            f"residency={self.residency!r} must be auto, single, device_split, or ulysses"
        )

    def context_parallel_world(self) -> int:
        return max(1, int(self.ulysses_degree)) * max(1, int(self.ring_degree))


def _is_modular_root(path: str) -> bool:
    index = os.path.join(path, "model_index.json")
    if not os.path.isfile(index):
        return False
    try:
        with open(index) as handle:
            meta = json.load(handle)
    except (OSError, ValueError):
        return False
    name = str(meta.get("_class_name") or "")
    return "Modular" in name or meta.get("_blocks_class_name") is not None


def resolve_diffusion_checkpoint(model_path: str) -> str:
    """Return the directory ``ModularPipeline.from_pretrained`` should open.

    Accepts the HF repo root or a ``FL2VA`` / ``Ref2VA`` partition path. The
    latter is remapped to the parent when the parent is the modular root, so
    scripts that historically pointed at ``.../FL2VA`` still work.
    """
    path = os.path.abspath(os.path.expanduser(model_path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"diffusion checkpoint not found: {path}")

    if _is_modular_root(path):
        return path

    base = os.path.basename(path.rstrip(os.sep))
    parent = os.path.dirname(path)
    if base in {"FL2VA", "Ref2VA"} and _is_modular_root(parent):
        return parent

    # Partition-only tree without a modular sibling: still require model_index
    # so runtime_select keeps classifying it as diffusion, but warn callers.
    if os.path.isfile(os.path.join(path, "model_index.json")):
        return path

    raise FileNotFoundError(
        f"{path} has no model_index.json (need MiniMax-H3 repo root or FL2VA/)"
    )


def frames_for_duration(duration_s: float, fps: int = SUPPORTED_FPS) -> int:
    """Snap duration onto the video VAE's ``17*n + 5`` latent-friendly frame count."""
    if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
        raise ValueError(
            f"duration={duration_s}s out of MiniMax-H3 range "
            f"[{MIN_DURATION_S}, {MAX_DURATION_S}]"
        )
    raw = int(round(float(duration_s) * int(fps)))
    # Smallest n such that 17*n + 5 >= raw.
    n = max(0, (raw - 5 + 16) // 17)
    frames = 17 * n + 5
    while frames / fps > MAX_DURATION_S and n > 0:
        n -= 1
        frames = 17 * n + 5
    while frames / fps < MIN_DURATION_S:
        n += 1
        frames = 17 * n + 5
        if frames / fps > MAX_DURATION_S:
            raise ValueError(
                f"cannot snap duration={duration_s}s at fps={fps} into "
                f"[{MIN_DURATION_S}, {MAX_DURATION_S}] with 17*n+5 frames"
            )
    return frames

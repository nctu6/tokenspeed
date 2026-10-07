# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI

"""CPU-only MiniMax-H3 config / checkpoint resolution tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenspeed.runtime.diffusion.config import (
    AUDIO_SIGMA_SHIFT,
    VIDEO_SIGMA_SHIFT,
    frames_for_duration,
    resolve_diffusion_checkpoint,
)


def test_sigma_shifts_are_pinned():
    assert VIDEO_SIGMA_SHIFT == 12.0
    assert AUDIO_SIGMA_SHIFT == 3.0


def test_frames_for_duration_snaps():
    frames = frames_for_duration(5.0, fps=24)
    assert frames % 17 == 5 or (frames - 5) % 17 == 0
    assert 5.0 <= frames / 24 <= 15.0


def test_frames_rejects_out_of_range():
    with pytest.raises(ValueError):
        frames_for_duration(1.0)
    with pytest.raises(ValueError):
        frames_for_duration(30.0)


def test_resolve_remaps_fl2va_partition(tmp_path: Path):
    root = tmp_path / "MiniMax-H3"
    fl2va = root / "FL2VA"
    fl2va.mkdir(parents=True)
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "_blocks_class_name": "MiniMaxH3Blocks",
            }
        )
    )
    (fl2va / "model_index.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Pipeline"})
    )
    assert resolve_diffusion_checkpoint(str(fl2va)) == str(root.resolve())
    assert resolve_diffusion_checkpoint(str(root)) == str(root.resolve())

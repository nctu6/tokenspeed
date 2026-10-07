# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI

"""CPU-only multipart reference classification / limit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tokenspeed.runtime.diffusion.config import H3Config
from tokenspeed.runtime.diffusion.references import (
    MAX_REF_IMAGES,
    classify_media,
    cleanup_staged,
    stage_uploads,
    validate_ref2va_counts,
)


def test_classify_by_mime_and_suffix():
    assert classify_media(filename="a.PNG", content_type=None) == "image"
    assert classify_media(filename="b.mp4", content_type="video/mp4") == "video"
    assert classify_media(filename="c.bin", content_type="audio/wav") == "audio"
    with pytest.raises(ValueError):
        classify_media(filename="x.bin", content_type="application/octet-stream")


def test_validate_ref2va_limits():
    validate_ref2va_counts(["image", "video", "audio"])
    with pytest.raises(ValueError):
        validate_ref2va_counts(["audio"])
    with pytest.raises(ValueError):
        validate_ref2va_counts(["image"] * (MAX_REF_IMAGES + 1))
    with pytest.raises(ValueError):
        validate_ref2va_counts([])


def test_stage_and_cleanup(tmp_path: Path):
    staged = stage_uploads(
        [
            (b"\x89PNG", "a.png", "image/png"),
            (b"\x00\x00", "b.mp4", "video/mp4"),
        ],
        directory=str(tmp_path / "refs"),
    )
    assert [s.kind for s in staged] == ["image", "video"]
    assert all(Path(s.path).is_file() for s in staged)
    cleanup_staged(staged)
    assert not any(Path(s.path).exists() for s in staged)


def test_residency_device_split_default_and_usp_rejected():
    cfg = H3Config(model_path="/tmp", num_gpus=2)
    assert cfg.resolve_residency() == "device_split"
    cfg1 = H3Config(model_path="/tmp", num_gpus=1)
    assert cfg1.resolve_residency() == "single"
    with pytest.raises(ValueError, match="Ulysses"):
        H3Config(model_path="/tmp", num_gpus=2, ulysses_degree=2).resolve_residency()
    with pytest.raises(ValueError, match="text encoder"):
        H3Config(model_path="/tmp", num_gpus=2, text_encoder_tp_size=2).resolve_residency()

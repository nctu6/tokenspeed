# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI

"""CPU-only MiniMax-Music3 config / family / request-shape tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tokenspeed.runtime.diffusion.family import (
    detect_diffusion_family,
    is_modular_checkpoint,
    resolve_modular_root,
)
from tokenspeed.runtime.diffusion.music3_config import (
    AUDIO_FRAME_RATE,
    MAX_AUDIO_FRAMES,
    duration_for_frames,
    frames_for_duration,
    resolve_music3_checkpoint,
)
from tokenspeed.runtime.entrypoints.music3_http import _request_from_body
from tokenspeed.runtime_select import resolve_serving_runtime


def _write_music3(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxMusic3ModularPipeline",
                "_blocks_class_name": "MiniMaxMusic3Blocks",
            }
        )
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxMusic3ForConditionalGeneration"],
                "model_type": "minimax_music3",
            }
        )
    )


def test_frames_for_duration_clamps():
    assert frames_for_duration(1.0) == AUDIO_FRAME_RATE
    assert frames_for_duration(30.0) == 30 * AUDIO_FRAME_RATE
    assert frames_for_duration(9999.0) == MAX_AUDIO_FRAMES
    assert duration_for_frames(750) == 30.0


def test_detect_music3_from_modular_index(tmp_path: Path):
    root = tmp_path / "MiniMax-Music3"
    _write_music3(root)
    assert is_modular_checkpoint(str(root))
    assert detect_diffusion_family(str(root)) == "music3"
    assert resolve_music3_checkpoint(str(root)) == str(root.resolve())
    assert resolve_serving_runtime(str(root)) == "diffusion"


def test_detect_h3_still_works(tmp_path: Path):
    root = tmp_path / "MiniMax-H3"
    root.mkdir()
    (root / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "_blocks_class_name": "MiniMaxH3Blocks",
            }
        )
    )
    assert detect_diffusion_family(str(root)) == "h3"
    assert resolve_serving_runtime(str(root)) == "diffusion"


def test_resolve_modular_root_music3(tmp_path: Path):
    root = tmp_path / "MiniMax-Music3"
    _write_music3(root)
    assert resolve_modular_root(str(root)) == str(root.resolve())


def test_speech_request_requires_lyrics_and_caption():
    with pytest.raises(ValueError, match="lyrics"):
        _request_from_body({"instructions": "jazz"})
    with pytest.raises(ValueError, match="instructions"):
        _request_from_body({"input": "[verse]\nhello"})


def test_speech_request_maps_openai_fields():
    req = _request_from_body(
        {
            "model": "test",
            "input": "[verse]\nhello",
            "instructions": "Genre: jazz. Soft piano.",
            "seed": 7,
            "max_new_tokens": 250,
            "num_inference_steps": 4,
            "response_format": "wav",
        }
    )
    assert req.lyrics.startswith("[verse]")
    assert "jazz" in req.prompt
    assert req.seed == 7
    assert req.max_new_tokens == 250
    assert req.num_inference_steps == 4


def test_speech_rejects_voice_and_temperature():
    with pytest.raises(ValueError, match="voice"):
        _request_from_body(
            {
                "input": "hi",
                "instructions": "jazz",
                "voice": "alloy",
            }
        )
    with pytest.raises(ValueError, match="temperature"):
        _request_from_body(
            {
                "input": "hi",
                "instructions": "jazz",
                "temperature": 0.7,
            }
        )


def test_residency_auto_split():
    from tokenspeed.runtime.diffusion.music3_config import Music3Config

    assert Music3Config(model_path="x", num_gpus=1).resolve_residency() == "single"
    assert Music3Config(model_path="x", num_gpus=2).resolve_residency() == "device_split"
    with pytest.raises(ValueError):
        Music3Config(model_path="x", num_gpus=1, residency="device_split").resolve_residency()


def test_speech_stream_false_still_ok():
    req = _request_from_body(
        {
            "input": "[verse]\nhello",
            "instructions": "jazz",
            "stream": False,
        }
    )
    assert getattr(req, "stream") is False


def test_speech_stream_defaults_to_sse():
    req = _request_from_body(
        {
            "input": "[verse]\nhello",
            "instructions": "jazz",
            "stream": True,
            "response_format": "pcm",
        }
    )
    assert req.stream is True
    assert req.stream_format == "sse"
    assert req.response_format == "pcm"


def test_speech_stream_format_audio():
    req = _request_from_body(
        {
            "input": "[verse]\nhello",
            "instructions": "jazz",
            "stream_format": "audio",
            "response_format": "wav",
        }
    )
    assert req.stream is True
    assert req.stream_format == "audio"


def test_speech_stream_rejects_non_pcm_wav():
    with pytest.raises(ValueError, match="streaming"):
        _request_from_body(
            {
                "input": "[verse]\nhello",
                "instructions": "jazz",
                "stream": True,
                "response_format": "mp3",
            }
        )


def test_acoustic_chunk_starts_and_sse_framing():
    from tokenspeed.runtime.diffusion.music3_stream import (
        Music3StreamChunk,
        acoustic_chunk_starts,
        iter_speech_sse,
        streaming_wav_header,
    )

    assert acoustic_chunk_starts(50) == [0]
    assert acoustic_chunk_starts(200) == [0]
    assert acoustic_chunk_starts(250) == [0, 100]
    assert acoustic_chunk_starts(400) == [0, 100, 200]

    header = streaming_wav_header(32000, num_channels=2)
    assert len(header) == 44
    assert header[:4] == b"RIFF"

    chunks = [
        Music3StreamChunk(
            pcm_s16le=b"\x01\x02\x03\x04",
            sample_rate=32000,
            channels=2,
            chunk_index=0,
            num_chunks=2,
            num_samples=1,
        ),
        Music3StreamChunk(
            pcm_s16le=b"\x05\x06",
            sample_rate=32000,
            channels=2,
            chunk_index=1,
            num_chunks=2,
            num_samples=1,
        ),
    ]
    frames = list(iter_speech_sse(iter(chunks), response_format="pcm"))
    joined = b"".join(frames).decode("utf-8")
    assert "event: speech.audio.delta" in joined
    assert "speech.audio.done" in joined
    assert frames[-1].decode("utf-8").startswith("event: speech.audio.done")

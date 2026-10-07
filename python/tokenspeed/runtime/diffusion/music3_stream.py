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

"""TokenSpeed-native MiniMax-Music3 streaming helpers.

Diffusers' Music3 ModularPipeline already denoises in 200-frame windows with a
100-frame hop, but the stock ``__call__`` buffers every window and returns one
finished waveform. TokenSpeed owns the progressive driver: after the AR stage
emits ``frame_hiddens``, each acoustic window is denoised, vocoded, cropped,
and yielded as PCM so ``/v1/audio/speech`` can stream real audio (OpenAI
``speech.audio.*`` SSE or raw ``audio`` bytes) instead of labeling a late
full-song buffer as a stream.

Window / crop sizes match the released Diffusers modular contract so a
concatenated stream matches the non-streaming path for the same seed.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass
from typing import Any, Iterator

__all__ = [
    "ACOUSTIC_CHUNK_FRAMES",
    "ACOUSTIC_CHUNK_HOP",
    "CROP_LEFT_LATENT",
    "CROP_RIGHT_LATENT",
    "Music3StreamChunk",
    "acoustic_chunk_starts",
    "pcm16le_bytes",
    "streaming_wav_header",
    "iter_speech_sse",
    "iter_speech_audio_bytes",
    "resample_poly_pcm",
]

# Acoustic windowing (Diffusers MiniMaxMusic3PrepareChunksStep contract).
ACOUSTIC_CHUNK_FRAMES = 200
ACOUSTIC_CHUNK_HOP = 100
# Vocoder stitch crops (Diffusers MiniMaxMusic3VocoderDecodeStep contract).
# Neighboring windows overlap by ~344 latent frames; keep the mid span so
# concatenated crops reconstruct the song once.
CROP_LEFT_LATENT = 86
CROP_RIGHT_LATENT = 344 - 86  # 258


@dataclass(frozen=True)
class Music3StreamChunk:
    """One progressive PCM window ready for HTTP framing."""

    pcm_s16le: bytes
    sample_rate: int
    channels: int
    chunk_index: int
    num_chunks: int
    num_samples: int

    @property
    def is_first(self) -> bool:
        return self.chunk_index == 0

    @property
    def is_last(self) -> bool:
        return self.chunk_index + 1 >= self.num_chunks


def acoustic_chunk_starts(num_frames: int) -> list[int]:
    """Frame starts for 200-wide / 100-hop acoustic windows."""
    n = int(num_frames)
    if n <= 0:
        return []
    if n <= ACOUSTIC_CHUNK_FRAMES:
        return [0]
    return list(range(0, n - ACOUSTIC_CHUNK_HOP, ACOUSTIC_CHUNK_HOP))


def streaming_wav_header(
    sample_rate: int, *, num_channels: int = 2, bits_per_sample: int = 16
) -> bytes:
    """44-byte WAV header with ``0xFFFFFFFF`` size placeholders (OpenAI-style)."""
    byte_rate = int(sample_rate) * int(num_channels) * int(bits_per_sample) // 8
    block_align = int(num_channels) * int(bits_per_sample) // 8
    placeholder = 0xFFFFFFFF
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        placeholder,
        b"WAVE",
        b"fmt ",
        16,
        1,
        int(num_channels),
        int(sample_rate),
        byte_rate,
        block_align,
        int(bits_per_sample),
        b"data",
        placeholder,
    )


def pcm16le_bytes(waveform, *, native_sr: int, out_sr: int) -> tuple[bytes, int, int, int]:
    """``waveform`` (channels, samples) float → (pcm_s16le, sr, samples, channels)."""
    import numpy as np

    if hasattr(waveform, "detach"):
        audio = waveform.detach().float().cpu().numpy()
    else:
        audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim != 2:
        raise RuntimeError(f"stream encode expected (C, T), got {audio.shape}")
    pcm = np.clip(audio.T, -1.0, 1.0).astype(np.float32)
    sample_rate = int(native_sr)
    if out_sr and int(out_sr) > 0 and int(out_sr) != sample_rate:
        pcm = resample_poly_pcm(pcm, sample_rate, int(out_sr))
        sample_rate = int(out_sr)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)
    return pcm_i16.tobytes(order="C"), sample_rate, int(pcm_i16.shape[0]), int(pcm_i16.shape[1])


def resample_poly_pcm(pcm, src_sr: int, dst_sr: int):
    """Lightweight polyphase resample; (T, C) float32 → (T', C)."""
    import numpy as np

    if src_sr == dst_sr:
        return pcm
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src_sr, dst_sr)
        up, down = dst_sr // g, src_sr // g
        channels = [
            resample_poly(pcm[:, c], up, down).astype(np.float32)
            for c in range(pcm.shape[1])
        ]
        return np.stack(channels, axis=1)
    except Exception:
        duration = pcm.shape[0] / float(src_sr)
        new_len = max(1, int(round(duration * dst_sr)))
        x_old = np.linspace(0.0, 1.0, pcm.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
        channels = [
            np.interp(x_new, x_old, pcm[:, c]).astype(np.float32)
            for c in range(pcm.shape[1])
        ]
        return np.stack(channels, axis=1)


def iter_speech_audio_bytes(
    chunks: Iterator[Music3StreamChunk], *, response_format: str
) -> Iterator[bytes]:
    """Raw byte stream: optional streaming WAV header + PCM16LE windows."""
    fmt = (response_format or "wav").lower()
    if fmt in {"wave", "audio/wav"}:
        fmt = "wav"
    if fmt not in {"wav", "pcm"}:
        raise ValueError("streaming response_format must be 'wav' or 'pcm'")
    for chunk in chunks:
        payload = chunk.pcm_s16le
        if fmt == "wav" and chunk.is_first:
            payload = (
                streaming_wav_header(chunk.sample_rate, num_channels=chunk.channels)
                + payload
            )
        yield payload


def iter_speech_sse(
    chunks: Iterator[Music3StreamChunk],
    *,
    response_format: str,
    usage: dict[str, Any] | None = None,
) -> Iterator[bytes]:
    """OpenAI ``speech.audio.*`` SSE frames (TokenSpeed chat-shaped streaming)."""
    fmt = (response_format or "pcm").lower()
    if fmt in {"wave", "audio/wav"}:
        fmt = "wav"
    if fmt not in {"wav", "pcm"}:
        raise ValueError("streaming response_format must be 'wav' or 'pcm'")
    emitted = False
    try:
        for raw in iter_speech_audio_bytes(chunks, response_format=fmt):
            payload = {
                "type": "speech.audio.delta",
                "audio": base64.b64encode(raw).decode("ascii"),
                "response_format": fmt,
            }
            data = json.dumps(payload, separators=(",", ":"))
            emitted = True
            yield f"event: speech.audio.delta\ndata: {data}\n\n".encode("utf-8")
        done: dict[str, Any] = {"type": "speech.audio.done"}
        if usage is not None:
            done["usage"] = usage
        yield f"event: speech.audio.done\ndata: {json.dumps(done, separators=(',', ':'))}\n\n".encode(
            "utf-8"
        )
    except Exception as exc:
        error = {
            "message": str(exc),
            "type": "server_error",
            "param": None,
            "code": 500,
        }
        if emitted:
            error["partial_audio"] = True
            error["action"] = "discard"
        payload = {"type": "speech.audio.error", "error": error}
        yield f"event: speech.audio.error\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode(
            "utf-8"
        )

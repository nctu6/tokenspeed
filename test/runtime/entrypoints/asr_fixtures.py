# Copyright (c) 2026 LightSeek Foundation
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""The ASR app over a stand-in scheduler, for ``/v1/audio/*`` tests.

The app, the feature extractor and the engine's sync ``LLM`` bridge are the
real ones. Only the scheduler behind ``AsyncLLM`` (processes and IPC) and the
checkpoint tokenizer are stand-ins: the stand-in ``AsyncLLM`` yields the
cumulative text an engine streams, one piece per step, when the request asks
to stream, and the whole transcript once otherwise. It keeps every request it
received, so a test can check what the endpoint asked the engine for.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from collections.abc import Iterator, Sequence

import numpy as np

PIECES = [" Hello", " there,", " world."]

# Whisper's prompt tokens, plus the language tokens the tests ask for.
KNOWN_TOKENS = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|de|>",
    "<|zh|>",
    "<|transcribe|>",
    "<|translate|>",
    "<|notimestamps|>",
]


class Tokenizer:
    """Whisper's prompt tokens only (the checkpoint tokenizer needs its files)."""

    def __init__(self, known: Sequence[str]) -> None:
        self._known = {token: index for index, token in enumerate(known)}
        self._unk = len(known)

    def convert_tokens_to_ids(self, tokens):
        return [self._known.get(token, self._unk) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        back = {index: token for token, index in self._known.items()}
        return [back.get(index, "<|unk|>") for index in ids]


class ScriptedAsyncLLM:
    """Stands in for the scheduler-backed ``AsyncLLM.generate_request``."""

    def __init__(self, pieces: Sequence[str]) -> None:
        self.pieces = list(pieces)
        self.requests: list = []

    async def generate_request(self, obj):
        self.requests.append(obj)
        if not obj.stream:
            yield {"text": "".join(self.pieces), "meta_info": {"finish_reason": None}}
            return
        text = ""
        for piece in self.pieces:
            text += piece
            yield {"text": text, "meta_info": {"finish_reason": None}}


@dataclasses.dataclass
class Engine:
    llm: object


@dataclasses.dataclass
class Processor:
    feature_extractor: object
    tokenizer: Tokenizer


def wav_bytes(
    seconds: float, rate: int, channels: int, rng: np.random.Generator
) -> bytes:
    """A WAV upload: one random-frequency tone per channel."""
    import soundfile as sf

    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    tones = [
        0.1 * np.sin(2 * np.pi * float(rng.uniform(200.0, 2000.0)) * t)
        for _ in range(channels)
    ]
    data = tones[0] if channels == 1 else np.stack(tones, axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV")
    return buffer.getvalue()


@contextlib.contextmanager
def asr_client(pieces: Sequence[str]) -> Iterator[tuple[object, ScriptedAsyncLLM]]:
    """A ``TestClient`` over the real app, and the stand-in scheduler behind it."""
    from fastapi.testclient import TestClient
    from transformers import WhisperFeatureExtractor

    from tokenspeed.runtime.engine.llm import LLM
    from tokenspeed.runtime.entrypoints.asr_http import create_app

    scheduler = ScriptedAsyncLLM(pieces)
    llm = LLM(scheduler)
    app = create_app(
        Engine(llm=llm),
        Processor(
            feature_extractor=WhisperFeatureExtractor(),
            tokenizer=Tokenizer(KNOWN_TOKENS),
        ),
        served_model_name="test",
    )
    try:
        with TestClient(app) as test_client:
            yield test_client, scheduler
    finally:
        llm.shutdown()

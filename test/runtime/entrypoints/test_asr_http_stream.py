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

"""Streamed ``/v1/audio/transcriptions``: the OpenAI/vLLM SSE chunk format.

The app, the feature extractor and the engine's sync ``LLM`` bridge are the
real ones. Only the scheduler behind ``AsyncLLM`` (processes and IPC) and the
checkpoint tokenizer are stand-ins: the stand-in ``AsyncLLM`` yields the
cumulative text an engine streams, one piece per step, when the request asks
to stream, and the whole transcript once otherwise.
"""

import dataclasses
import io
import json

import numpy as np
import pytest

PIECES = [" Hello", " there,", " world."]

KNOWN_TOKENS = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|transcribe|>",
    "<|translate|>",
    "<|notimestamps|>",
]


class _Tokenizer:
    """Whisper's prompt tokens only (the checkpoint tokenizer needs its files)."""

    def __init__(self, known):
        self._known = {token: index for index, token in enumerate(known)}
        self._unk = len(known)

    def convert_tokens_to_ids(self, tokens):
        return [self._known.get(token, self._unk) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        back = {index: token for token, index in self._known.items()}
        return [back.get(index, "<|unk|>") for index in ids]


class _ScriptedAsyncLLM:
    """Stands in for the scheduler-backed ``AsyncLLM.generate_request``."""

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.requests = []

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
class _Engine:
    llm: object


@dataclasses.dataclass
class _Processor:
    feature_extractor: object
    tokenizer: _Tokenizer


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    import soundfile as sf

    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    buffer = io.BytesIO()
    sf.write(buffer, 0.1 * np.sin(2 * np.pi * 440.0 * t), rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from transformers import WhisperFeatureExtractor

    from tokenspeed.runtime.engine.llm import LLM
    from tokenspeed.runtime.entrypoints.asr_http import create_app

    llm = LLM(_ScriptedAsyncLLM(PIECES))
    app = create_app(
        _Engine(llm=llm),
        _Processor(
            feature_extractor=WhisperFeatureExtractor(),
            tokenizer=_Tokenizer(KNOWN_TOKENS),
        ),
        served_model_name="test",
    )
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        llm.shutdown()


@pytest.mark.parametrize("seed", [0, 1])
def test_streamed_transcription_sends_incremental_sse_chunks(client, seed):
    """**Validates: Requirements 1.6, 2.6** (Property 1, noStream)."""
    rng = np.random.default_rng(seed)
    audio = _wav_bytes(seconds=float(rng.uniform(0.5, 2.0)))
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data={
            "model": "test",
            "language": "en",
            "temperature": "0",
            "response_format": "json",
            "stream": "true",
        },
    )
    assert response.status_code == 200, response.text
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/event-stream"), (
        f"stream=true answered with {content_type!r} and body "
        f"{response.text[:200]!r}, not an SSE stream"
    )
    events = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events and events[-1] == "[DONE]", events
    chunks = [json.loads(event) for event in events[:-1]]
    assert len(chunks) > 1, f"one chunk is not a stream: {chunks}"
    for chunk in chunks:
        assert chunk["object"] == "transcription.chunk", chunk
        assert isinstance(chunk["choices"][0]["delta"]["content"], str), chunk
    streamed = "".join(chunk["choices"][0]["delta"]["content"] for chunk in chunks)
    assert streamed == "".join(PIECES)

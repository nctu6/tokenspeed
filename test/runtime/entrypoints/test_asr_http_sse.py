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

"""Streamed ``/v1/audio/*``: the parts of vLLM 0.32.0's SSE schema past the
deltas -- the finishing chunk, usage, translations, and errors.

The app, the feature extractor and the engine's sync ``LLM`` bridge are the
real ones; the scheduler behind ``AsyncLLM`` replays scripted engine outputs
(cumulative text, request-total token counts, a finish reason on the last).
"""

import contextlib
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runtime.entrypoints.asr_fixtures import (  # noqa: E402
    KNOWN_TOKENS,
    Engine,
    Processor,
    Tokenizer,
    wav_bytes,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

PROMPT_TOKENS = 4
EOS = 50257

# Cumulative text per engine step. The repeated "Grüße" step is a step whose
# token completed no character (the engine's detokenizer held it back), so it
# adds no text and must not become a chunk.
STEPS = [" Grüße", " Grüße", " Grüße aus", " Grüße aus 東京."]


class _ReplayAsyncLLM:
    """Replays ``STEPS`` as the engine's streamed outputs for each request."""

    def __init__(self, fail_after: int | None) -> None:
        self.fail_after = fail_after
        self.requests: list = []

    async def generate_request(self, obj):
        self.requests.append(obj)
        for step, text in enumerate(STEPS):
            if self.fail_after is not None and step == self.fail_after:
                raise RuntimeError("engine went away")
            last = step == len(STEPS) - 1
            yield {
                "text": text,
                "meta_info": {
                    "finish_reason": (
                        {"type": "stop", "matched": EOS} if last else None
                    ),
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": step + 1,
                },
            }


@contextlib.contextmanager
def _client(fail_after: int | None):
    from fastapi.testclient import TestClient
    from transformers import WhisperFeatureExtractor

    from tokenspeed.runtime.engine.llm import LLM
    from tokenspeed.runtime.entrypoints.asr_http import create_app

    scheduler = _ReplayAsyncLLM(fail_after)
    llm = LLM(scheduler)
    app = create_app(
        Engine(llm=llm),
        Processor(
            feature_extractor=WhisperFeatureExtractor(),
            tokenizer=Tokenizer(KNOWN_TOKENS),
        ),
        served_model_name="served",
    )
    try:
        with TestClient(app) as test_client:
            yield test_client, scheduler
    finally:
        llm.shutdown()


def _events(response) -> list:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        line[len("data: ") :]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]", events
    return [json.loads(event) for event in events[:-1]]


def _audio(seed: int) -> bytes:
    return wav_bytes(1.0, 16000, 1, np.random.default_rng(seed))


@pytest.mark.parametrize(
    "route, obj, task_token",
    [
        ("/v1/audio/transcriptions", "transcription.chunk", "<|transcribe|>"),
        ("/v1/audio/translations", "translation.chunk", "<|translate|>"),
    ],
)
def test_stream_matches_vllm_schema_with_usage(route, obj, task_token):
    with _client(fail_after=None) as (client, scheduler):
        response = client.post(
            route,
            files={"file": ("clip.wav", _audio(0), "audio/wav")},
            data={
                "stream": "True",
                "stream_include_usage": "True",
                "stream_continuous_usage_stats": "True",
            },
        )
        chunks = _events(response)

    (request,) = scheduler.requests
    assert request.stream is True
    assert request.input_ids[2] == KNOWN_TOKENS.index(task_token)

    *text_chunks, usage_chunk = chunks
    # One chunk per step that added text; the held-back step sent nothing.
    assert [c["choices"][0]["delta"]["content"] for c in text_chunks] == [
        " Grüße",
        " aus",
        " 東京.",
    ]
    for chunk in chunks:
        assert chunk["object"] == obj
        assert chunk["model"] == "served"
        assert chunk["id"] == chunks[0]["id"]
        assert chunk["created"] == chunks[0]["created"]
    for chunk in text_chunks[:-1]:
        assert set(chunk["choices"][0]) == {"delta"}
    assert text_chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert text_chunks[-1]["choices"][0]["stop_reason"] is None
    # Continuous usage: request totals so far, on every text chunk.
    assert [c["usage"]["completion_tokens"] for c in text_chunks] == [1, 3, 4]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {
        "prompt_tokens": PROMPT_TOKENS,
        "total_tokens": PROMPT_TOKENS + len(STEPS),
        "completion_tokens": len(STEPS),
    }


def test_stream_without_usage_sends_no_usage():
    with _client(fail_after=None) as (client, _):
        chunks = _events(
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", _audio(1), "audio/wav")},
                data={"model": "asked", "stream": "true"},
            )
        )
    assert all("usage" not in chunk for chunk in chunks)
    assert all(chunk["model"] == "asked" for chunk in chunks)
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == STEPS[-1]


@pytest.mark.parametrize(
    "audio, language",
    [(b"", "en"), (None, "zzz")],
    ids=["empty-audio", "unknown-language"],
)
def test_stream_rejects_before_streaming(audio, language):
    with _client(fail_after=None) as (client, scheduler):
        response = client.post(
            "/v1/audio/transcriptions",
            files={
                "file": ("clip.wav", _audio(2) if audio is None else audio, "audio/wav")
            },
            data={"language": language, "stream": "true"},
        )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert "error" in response.json()
    assert scheduler.requests == []


def test_engine_error_mid_stream_is_an_error_event_then_done():
    with _client(fail_after=2) as (client, _):
        chunks = _events(
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("clip.wav", _audio(3), "audio/wav")},
                data={"stream": "true"},
            )
        )
    assert [c["choices"][0]["delta"]["content"] for c in chunks[:-1]] == [" Grüße"]
    assert "engine went away" in chunks[-1]["error"]["message"]

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

"""Non-streamed ``/v1/audio/*``: what the endpoints answer and ask the engine for.

Preservation property:
for any non-streamed request -- either route, any language code, any sample
rate, mono or stereo, empty or not, JSON or text -- the response body, the
status, the decoder prompt and the log-mel features handed to the engine are
what the pre-fix server produces. The expected values were observed on the
unfixed code and are written out here.
"""

import io
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
    PIECES,
    asr_client,
    wav_bytes,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

TRANSCRIPT = "".join(PIECES)
RATES = (8000, 16000, 22050, 44100)
LANGUAGES = (None, "en", "de", "zh", "zzz")


def _token_id(token: str) -> int:
    return KNOWN_TOKENS.index(token)


def _expected_features(audio: bytes):
    """The log-mel window the pre-fix endpoint computes for an upload.

    Read as float32, stereo averaged to mono, resampled to the extractor's
    16 kHz by the reduced polyphase ratio, then HF's feature extractor.
    """
    from math import gcd

    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor()
    wav, rate = sf.read(io.BytesIO(audio), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    target = int(extractor.sampling_rate)
    if rate != target:
        divisor = gcd(int(rate), target)
        wav = resample_poly(wav, target // divisor, int(rate) // divisor)
        rate = target
    features = extractor(wav, sampling_rate=rate, return_tensors="pt").input_features
    return torch.as_tensor(features[0])


def _draw_case(seed: int) -> dict:
    """One random non-streamed request."""
    rng = np.random.default_rng(seed)
    route = ("transcriptions", "translations")[int(rng.integers(2))]
    return {
        "rng": rng,
        "route": route,
        "language": LANGUAGES[int(rng.integers(len(LANGUAGES)))],
        "rate": RATES[int(rng.integers(len(RATES)))],
        "channels": int(rng.integers(1, 3)),
        "empty": bool(rng.random() < 0.15),
        "response_format": ("json", "text")[int(rng.integers(2))],
        "temperature": (0.0, 0.7)[int(rng.integers(2))],
        "seconds": float(rng.uniform(0.3, 2.0)),
    }


@pytest.mark.parametrize("seed", range(24))
def test_non_streamed_requests_answer_as_before(seed):
    """**Validates: Requirements 3.2, 3.3** (Property 3, API preservation)."""
    import torch

    case = _draw_case(seed)
    audio = (
        b""
        if case["empty"]
        else wav_bytes(case["seconds"], case["rate"], case["channels"], case["rng"])
    )
    data = {
        "model": "test",
        "response_format": case["response_format"],
        "temperature": str(case["temperature"]),
    }
    if case["language"] is not None:
        # The translations route takes no language field; a client may still send one.
        data["language"] = case["language"]

    with asr_client(PIECES) as (client, scheduler):
        response = client.post(
            f"/v1/audio/{case['route']}",
            files={"file": ("clip.wav", audio, "audio/wav")},
            data=data,
        )

    print(
        f"seed {seed}: {case['route']} lang={case['language']} rate={case['rate']} "
        f"ch={case['channels']} empty={case['empty']} fmt={case['response_format']} "
        f"-> {response.status_code}"
    )
    if case["empty"]:
        assert response.status_code == 400
        assert response.json() == {"error": "empty audio file"}
        assert scheduler.requests == []
        return

    task = "transcribe" if case["route"] == "transcriptions" else "translate"
    language = case["language"] if task == "transcribe" else None
    resolved = language or "en"
    if resolved == "zzz":
        assert response.status_code == 400
        assert response.json() == {
            "error": "this checkpoint's tokenizer has no ['<|zzz|>'] -- check the "
            "`language` code (Whisper wants 'en', 'zh', 'ja', ...)"
        }
        assert scheduler.requests == []
        return

    assert response.status_code == 200, response.text
    if case["response_format"] == "text":
        assert response.headers["content-type"].startswith("text/plain")
        assert response.text == TRANSCRIPT
    else:
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"text": TRANSCRIPT, "language": resolved}

    (request,) = scheduler.requests
    assert not request.stream
    assert request.input_ids == [
        _token_id("<|startoftranscript|>"),
        _token_id(f"<|{resolved}|>"),
        _token_id(f"<|{task}|>"),
        _token_id("<|notimestamps|>"),
    ]
    assert request.sampling_params == {
        "temperature": case["temperature"],
        "max_new_tokens": 448,
    }
    (item,) = request.precomputed_multimodal_inputs.mm_items
    assert item.modality.name == "AUDIO"
    torch.testing.assert_close(
        item.feature, _expected_features(audio), atol=0.0, rtol=0.0
    )

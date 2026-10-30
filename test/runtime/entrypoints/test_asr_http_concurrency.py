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

"""``/v1/audio/*`` under more concurrent requests than may extract at once.

The endpoint bounds how many requests compute log-mel features at a time
(``asr_http`` module docstring). These tests check that the bound changes
nothing a client can see: every concurrent request still gets its own
features, bit-exact, and a request rejected while extracting gives its turn
back.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

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
    PIECES,
    asr_client,
    wav_bytes,
)
from test.runtime.entrypoints.test_asr_http_preservation import (  # noqa: E402
    _expected_features,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

TRANSCRIPT = "".join(PIECES)
CALLERS = 16


def _post(client, audio: bytes, language: str):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data={"model": "test", "language": language},
    )


@pytest.mark.parametrize("seed", range(3))
def test_concurrent_requests_each_get_their_own_features(seed):
    """**Validates: Requirements 3.2, 3.3** (Property 3 under concurrency)."""
    import torch

    rng = np.random.default_rng(seed)
    clips = [
        wav_bytes(
            float(rng.uniform(0.3, 2.0)),
            (8000, 16000, 44100)[int(rng.integers(3))],
            int(rng.integers(1, 3)),
            rng,
        )
        for _ in range(CALLERS)
    ]
    with asr_client(PIECES) as (client, scheduler):
        with ThreadPoolExecutor(CALLERS) as pool:
            responses = list(pool.map(lambda a: _post(client, a, "en"), clips))

    assert [r.status_code for r in responses] == [200] * CALLERS
    assert all(r.json() == {"text": TRANSCRIPT, "language": "en"} for r in responses)
    # Requests reach the engine in completion order; match each clip to the
    # one request carrying exactly its features.
    sent = [
        r.precomputed_multimodal_inputs.mm_items[0].feature for r in scheduler.requests
    ]
    assert len(sent) == CALLERS
    for audio in clips:
        expected = _expected_features(audio)
        matches = [f for f in sent if torch.equal(f, expected)]
        assert matches, "a request's features were not handed to the engine"


def test_rejected_requests_release_their_turn():
    """A 400 raised during extraction must not keep the extraction slot."""
    rng = np.random.default_rng(0)
    audio = wav_bytes(0.5, 16000, 1, rng)
    with asr_client(PIECES) as (client, scheduler):
        with ThreadPoolExecutor(CALLERS) as pool:
            rejected = list(
                pool.map(lambda _: _post(client, audio, "zzz"), range(2 * CALLERS))
            )
        accepted = _post(client, audio, "en")

    assert {r.status_code for r in rejected} == {400}
    assert accepted.status_code == 200
    assert len(scheduler.requests) == 1

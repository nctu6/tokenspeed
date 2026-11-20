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

"""HTTP contract of the OpenAI-compatible ``/v1/embeddings`` server."""

import base64
import os
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.entrypoints.embedding_server import (
    create_app,
    parse_embedding_input,
)

register_cuda_ci(est_time=5, suite="runtime-1gpu")

MODEL = "test-embed"


class FakeEngine:
    """Records ``encode`` calls; one vector per input, sized by prompt length."""

    def __init__(self):
        self.calls: list[dict] = []

    def encode(self, prompt=None, input_ids=None):
        self.calls.append({"prompt": prompt, "input_ids": input_ids})
        items = prompt if prompt is not None else input_ids
        outputs = [
            {
                "embedding": [float(i), 0.5, -1.0],
                "meta_info": {"prompt_tokens": len(item)},
            }
            for i, item in enumerate(items)
        ]
        # Engine.encode returns a dict for a single input.
        return outputs[0] if len(outputs) == 1 else outputs


@pytest.fixture
def engine():
    return FakeEngine()


def _client(engine, api_key=None):
    return TestClient(create_app(engine, served_model_name=MODEL, api_key=api_key))


def test_parse_embedding_input_forms():
    assert parse_embedding_input("hi") == (["hi"], None)
    assert parse_embedding_input(["a", "b"]) == (["a", "b"], None)
    assert parse_embedding_input([1, 2, 3]) == (None, [[1, 2, 3]])
    assert parse_embedding_input([[1, 2], [3]]) == (None, [[1, 2], [3]])
    for bad in ([], None, [True, False], [[1], []], ["a", 1]):
        with pytest.raises(ValueError):
            parse_embedding_input(bad)


def test_embeddings_texts_and_usage(engine):
    resp = _client(engine).post(
        "/v1/embeddings", json={"model": MODEL, "input": ["abc", "de"]}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert body["data"][1]["embedding"] == [1.0, 0.5, -1.0]
    assert body["usage"] == {"prompt_tokens": 5, "total_tokens": 5}
    assert engine.calls == [{"prompt": ["abc", "de"], "input_ids": None}]


def test_embeddings_single_token_list_is_one_input(engine):
    resp = _client(engine).post("/v1/embeddings", json={"input": [7, 8, 9]})
    assert resp.status_code == 200
    assert len(resp.json()["data"]) == 1
    assert engine.calls == [{"prompt": None, "input_ids": [[7, 8, 9]]}]


def test_embeddings_base64_round_trips(engine):
    resp = _client(engine).post(
        "/v1/embeddings", json={"input": "x", "encoding_format": "base64"}
    )
    encoded = resp.json()["data"][0]["embedding"]
    decoded = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
    np.testing.assert_array_equal(decoded, [0.0, 0.5, -1.0])


@pytest.mark.parametrize(
    "body, status",
    [
        ({"model": "other", "input": "x"}, 404),
        ({"input": "x", "encoding_format": "int8"}, 400),
        ({"input": "x", "dimensions": 64}, 400),
        ({"input": []}, 400),
    ],
)
def test_embeddings_rejects_bad_requests(engine, body, status):
    resp = _client(engine).post("/v1/embeddings", json=body)
    assert resp.status_code == status
    assert engine.calls == []


def test_api_key_guards_v1_but_not_health(engine):
    client = _client(engine, api_key="secret")
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    wrong = {"Authorization": "Bearer nope"}
    assert (
        client.post("/v1/embeddings", json={"input": "x"}, headers=wrong).status_code
        == 401
    )
    ok = {"Authorization": "Bearer secret"}
    models = client.get("/v1/models", headers=ok)
    assert [m["id"] for m in models.json()["data"]] == [MODEL]
    assert (
        client.post("/v1/embeddings", json={"input": "x"}, headers=ok).status_code
        == 200
    )

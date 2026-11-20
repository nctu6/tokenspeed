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

"""``ts serve`` routes pooling checkpoints to the in-process embedding server."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from tokenspeed.cli._argsplit import split_argv
from tokenspeed.cli.serve_smg import _embedding_server_args, _run_embedding_server
from tokenspeed.runtime.entrypoints import embedding_server
from tokenspeed.runtime.utils import server_args as server_args_module


def _write_pooling_checkpoint(path) -> str:
    (path / "1_Pooling").mkdir()
    (path / "1_Pooling" / "config.json").write_text(
        json.dumps({"pooling_mode_lasttoken": True})
    )
    return str(path)


@pytest.fixture
def parsed(monkeypatch):
    """Stub ServerArgs parsing; records the engine argv it was given."""
    seen: list[list[str]] = []

    def fake_prepare(argv):
        seen.append(list(argv))
        return SimpleNamespace(
            model=argv[argv.index("--model") + 1],
            is_embedding="--is-embedding" in argv or None,
            pooling_type="last" if "--pooling-type" in argv else None,
            pooling_normalize=None,
            nnodes=1,
        )

    monkeypatch.setattr(server_args_module, "prepare_server_args", fake_prepare)
    return seen


def test_generative_checkpoint_skips_engine_parsing(tmp_path, parsed):
    assert _embedding_server_args(split_argv([str(tmp_path)])) is None
    assert parsed == []


def test_pooling_checkpoint_selects_embedding_server(tmp_path, parsed):
    model = _write_pooling_checkpoint(tmp_path)
    assert _embedding_server_args(split_argv([model])).model == model
    assert _embedding_server_args(split_argv([model, "--no-is-embedding"])) is None


def test_is_embedding_flag_on_plain_checkpoint(tmp_path, parsed):
    argv = [str(tmp_path), "--is-embedding", "--pooling-type", "last"]
    assert _embedding_server_args(split_argv(argv)) is not None


def test_run_embedding_server_forwards_gateway_flags(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        embedding_server, "serve", lambda args, **kw: calls.append((args, kw))
    )
    args = SimpleNamespace(nnodes=1)
    split = split_argv(
        [str(tmp_path), "--host", "127.0.0.1", "--port", "9001", "--api-key", "k"]
    )
    _run_embedding_server(args, split)
    assert calls == [(args, {"host": "127.0.0.1", "port": 9001, "api_key": "k"})]


def test_run_embedding_server_refuses_gateway_only_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_server, "serve", lambda *a, **kw: None)
    split = split_argv([str(tmp_path), "--policy", "round_robin"])
    with pytest.raises(SystemExit, match="--policy"):
        _run_embedding_server(SimpleNamespace(nnodes=1), split)
    with pytest.raises(SystemExit, match="single node"):
        _run_embedding_server(SimpleNamespace(nnodes=2), split_argv([str(tmp_path)]))

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

"""--prefill-coalesce-ms: the input rank holds new requests while others decode."""

import pytest

from tokenspeed.runtime.engine import request_handler
from tokenspeed.runtime.engine.request_handler import RequestHandler


class _Generate:
    """Stands in for TokenizedGenerateReqInput: _coalesce only checks the type."""

    def __init__(self, rid: str) -> None:
        self.rid = rid


class _Abort:
    pass


@pytest.fixture(autouse=True)
def _generate_type(monkeypatch):
    monkeypatch.setattr(request_handler, "TokenizedGenerateReqInput", _Generate)


def _handler(window_s: float, busy: bool) -> RequestHandler:
    handler = RequestHandler.__new__(RequestHandler)
    handler.prefill_coalesce_s = window_s
    handler._held_reqs = []
    handler._held_since = 0.0
    handler.engine_busy = busy
    return handler


def _generate(rid: str) -> _Generate:
    return _Generate(rid)


def _clock(monkeypatch, now: list[float]) -> None:
    monkeypatch.setattr(request_handler.time, "monotonic", lambda: now[0])


def test_requests_pass_through_while_nothing_decodes(monkeypatch):
    now = [10.0]
    _clock(monkeypatch, now)
    handler = _handler(0.05, busy=False)
    a = _generate("a")
    assert handler._coalesce([a]) == [a]
    assert handler._coalesce([]) == []


def test_requests_wait_for_the_window_and_release_in_order(monkeypatch):
    now = [10.0]
    _clock(monkeypatch, now)
    handler = _handler(0.05, busy=True)
    a, b = _generate("a"), _generate("b")
    assert handler._coalesce([a]) == []
    now[0] = 10.03
    assert handler._coalesce([b]) == []
    # The window runs from the oldest held request.
    now[0] = 10.05
    assert handler._coalesce([]) == [a, b]
    assert handler._held_reqs == []


def test_held_requests_release_when_decoding_stops(monkeypatch):
    now = [10.0]
    _clock(monkeypatch, now)
    handler = _handler(0.05, busy=True)
    a = _generate("a")
    assert handler._coalesce([a]) == []
    handler.engine_busy = False
    assert handler._coalesce([]) == [a]


def test_a_control_message_never_overtakes_held_requests(monkeypatch):
    now = [10.0]
    _clock(monkeypatch, now)
    handler = _handler(0.05, busy=True)
    a = _generate("a")
    abort = _Abort()
    assert handler._coalesce([a]) == []
    assert handler._coalesce([abort]) == [a, abort]

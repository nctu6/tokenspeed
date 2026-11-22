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
"""The default request log (level 0) records metadata, not payloads.

A pooled vector is ``hidden_size`` floats; logging it at level 0 wrote a 74 KB
line per embedding request on the engine's event-loop thread.
"""
import os
import sys

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.utils.common import dataclass_to_string_truncated  # noqa: E402


class _LoggingAsyncLLM(AsyncLLM):
    """Skips ZMQ, ModelConfig and tokenizer bring-up; only the logging knobs."""

    def __init__(self, level: int) -> None:
        self.log_requests = True
        self.log_requests_level = level


def test_level0_output_log_omits_the_pooled_vector():
    llm = _LoggingAsyncLLM(level=0)
    max_length, _, out_skip_names = llm.get_log_request_metadata()
    out = {
        "embedding": [0.0123456789] * 4096,
        "meta_info": {"id": "r1", "prompt_tokens": 392},
    }
    line = dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)
    assert "0.0123" not in line
    assert "'prompt_tokens': 392" in line
    assert len(line) < 200


def test_level2_output_log_keeps_the_pooled_vector():
    llm = _LoggingAsyncLLM(level=2)
    max_length, _, out_skip_names = llm.get_log_request_metadata()
    out = {"embedding": [0.5] * 8, "meta_info": {"prompt_tokens": 3}}
    line = dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)
    assert line.count("0.5") == 8

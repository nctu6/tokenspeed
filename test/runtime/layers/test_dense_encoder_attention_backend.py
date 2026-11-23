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
"""The attention backend a dense-window encoder (Whisper) gets by default."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_kernel.platform import ArchVersion, PlatformInfo  # noqa: E402

from tokenspeed.runtime.layers.attention.mm_encoder_attention import (  # noqa: E402
    dense_encoder_attention_backend,
)


def _platform(vendor: str, major: int, minor: int) -> PlatformInfo:
    return PlatformInfo(
        vendor=vendor,
        arch_version=ArchVersion(major, minor),
        device_name="test",
        device_count=1,
        total_memory=96 * 2**30,
        memory_bandwidth=1792.0,
        sm_count=188,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=101376,
    )


@pytest.mark.parametrize("minor", [0, 1])
def test_workstation_blackwell_takes_sdpa(minor):
    assert (
        dense_encoder_attention_backend(None, _platform("nvidia", 12, minor)) == "sdpa"
    )


@pytest.mark.parametrize(
    "platform",
    [
        _platform("nvidia", 9, 0),
        _platform("nvidia", 10, 0),
        _platform("amd", 12, 5),
    ],
    ids=["hopper", "datacenter-blackwell", "amd"],
)
def test_other_platforms_keep_the_platform_default(platform):
    assert dense_encoder_attention_backend(None, platform) is None


def test_explicit_backend_wins():
    sm120 = _platform("nvidia", 12, 0)
    assert dense_encoder_attention_backend("triton_attn", sm120) == "triton_attn"

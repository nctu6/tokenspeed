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
"""Which attention backend consumes the hybrid history cache on each platform."""

import pytest
from tokenspeed_kernel.platform import ArchVersion, Platform, PlatformInfo

from tokenspeed.runtime.layers.attention import registry as attention_registry


def _platform(vendor: str, major: int, minor: int) -> PlatformInfo:
    return PlatformInfo(
        vendor=vendor,
        arch_version=ArchVersion(major, minor),
        device_name=f"{vendor} {major}.{minor}",
        device_count=1,
        total_memory=96 * (1024**3),
        memory_bandwidth=1792.0,
        sm_count=188,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=102400,
    )


@pytest.fixture
def platform():
    """Pin the detected platform for one test and restore detection after."""
    original = Platform.get()

    def pin(vendor: str, major: int, minor: int) -> None:
        Platform.override(_platform(vendor, major, minor))

    yield pin
    Platform.override(original)


def _resolve(requested: str | None) -> str | None:
    return attention_registry._resolve_hybrid_full_backend_name(
        requested, is_kda=True, is_dsa=False, is_qsa=False
    )


@pytest.mark.parametrize(
    ("vendor", "major", "minor", "expected"),
    [
        ("nvidia", 10, 0, "tokenspeed_mla"),  # SM100: tcgen05 CuTe kernels
        ("nvidia", 10, 3, "tokenspeed_mla"),
        ("nvidia", 12, 0, None),  # SM120 has no tcgen05: generic MLA backend
        ("amd", 9, 5, None),
    ],
)
def test_k3_default_history_consumer_follows_the_platform(
    platform, vendor, major, minor, expected
):
    platform(vendor, major, minor)
    assert _resolve(None) == expected


def test_explicit_choice_is_kept_on_every_platform(platform):
    for vendor, major, minor in (("nvidia", 10, 0), ("nvidia", 12, 0), ("amd", 9, 5)):
        platform(vendor, major, minor)
        assert _resolve("flashmla") == "flashmla"
        assert _resolve("tokenspeed_mla") == "tokenspeed_mla"

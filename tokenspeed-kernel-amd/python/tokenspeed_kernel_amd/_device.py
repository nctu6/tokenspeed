# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Device geometry of the visible gfx950 parts (CUs and XCD chiplets).

Kernels tuned on MI355X (256 CUs, 8 XCDs) size grids and chiplet swizzles from
these instead of fixed constants, so MI350P (128 CUs, 4 XCDs) gets one-wave
split-K and chiplet-local program ids too. Each gfx950 XCD has 32 CUs.
"""

from __future__ import annotations

import functools
import os

import torch

_CUS_PER_XCD = 32


@functools.cache
def num_cus() -> int:
    """Compute units of device 0 (all visible GPUs are the same part)."""
    override = os.environ.get("TOKENSPEED_AMD_NUM_CUS")
    if override:
        return int(override)
    try:
        return int(torch.cuda.get_device_properties(0).multi_processor_count)
    except (AssertionError, RuntimeError):
        return 256


@functools.cache
def num_xcds() -> int:
    """XCD chiplets of device 0."""
    override = os.environ.get("TOKENSPEED_AMD_NUM_XCDS")
    if override:
        return int(override)
    return max(1, num_cus() // _CUS_PER_XCD)

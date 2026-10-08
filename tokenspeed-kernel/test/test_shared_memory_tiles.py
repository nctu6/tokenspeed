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

"""Shared-memory budget detection and Triton MHA prefill tile selection.

``PlatformInfo.max_shared_memory_per_sm`` must carry the CUDA opt-in per-block
ceiling, and the prefill tile choice must follow that budget (sm_120 ~100 KB
takes small tiles despite sorting above sm_90) while sm_86/sm_89 keep small
tiles even when the budget is unknown. Pure-Python checks; one optional test
reads the real device.
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha._triton.prefill import (
    _LARGE_SHARED_MEMORY_BYTES,
    _has_large_shared_memory,
    _select_prefill_tiles,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    PlatformInfo,
    _cuda_shared_memory_budget,
)

_SM120_BUDGET = 101376  # RTX PRO 6000 / workstation Blackwell opt-in limit
_SM90_BUDGET = 232448
_SM80_BUDGET = 167936

# --------------------------------------------------------------------------
# _cuda_shared_memory_budget
# --------------------------------------------------------------------------


def test_budget_prefers_optin_limit() -> None:
    props = SimpleNamespace(
        shared_memory_per_block_optin=_SM120_BUDGET,
        shared_memory_per_block=48 * 1024,
        max_shared_memory_per_block=7,  # not a torch attribute; must be ignored
    )
    assert _cuda_shared_memory_budget(props) == _SM120_BUDGET


@pytest.mark.parametrize("optin", [None, 0], ids=["missing", "zero"])
def test_budget_falls_back_to_default_limit(optin) -> None:
    props = SimpleNamespace(shared_memory_per_block=48 * 1024)
    if optin is not None:
        props.shared_memory_per_block_optin = optin
    assert _cuda_shared_memory_budget(props) == 48 * 1024


def test_budget_unknown_returns_zero_and_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="tokenspeed_kernel.platform"):
        assert _cuda_shared_memory_budget(SimpleNamespace()) == 0
    assert any(
        "no shared-memory limit" in r.getMessage() for r in caplog.records
    ), caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_detected_budget_matches_device_optin() -> None:
    from tokenspeed_kernel.platform import current_platform

    platform = current_platform()
    if not platform.is_nvidia:
        pytest.skip("NVIDIA-only")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    assert platform.max_shared_memory_per_sm == props.shared_memory_per_block_optin
    assert platform.max_shared_memory_per_sm > 0
    expected_large = props.shared_memory_per_block_optin >= _LARGE_SHARED_MEMORY_BYTES
    assert _has_large_shared_memory(platform) is expected_large


# --------------------------------------------------------------------------
# _has_large_shared_memory
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (0, True),  # unknown: keep the tiles validated on datacenter parts
        (_SM120_BUDGET, False),
        (_LARGE_SHARED_MEMORY_BYTES - 1, False),
        (_LARGE_SHARED_MEMORY_BYTES, True),
        (_SM80_BUDGET, True),
        (_SM90_BUDGET, True),
    ],
)
def test_large_shared_memory_threshold(budget: int, expected: bool) -> None:
    platform = SimpleNamespace(max_shared_memory_per_sm=budget)
    assert _has_large_shared_memory(platform) is expected


# --------------------------------------------------------------------------
# _select_prefill_tiles
# --------------------------------------------------------------------------

_HOPPER_TILES = {64: (128, 64), 128: (128, 64), 256: (128, 64), 512: (32, 64)}
_SMALL_AMPERE_TILES = {64: (64, 128), 128: (64, 128), 256: (64, 64), 512: (32, 32)}
_SM80_TILES = {64: (128, 128), 128: (128, 128), 256: (64, 64), 512: (32, 64)}
_PRE_AMPERE_TILES = {64: (64, 64), 128: (64, 64), 256: (32, 32), 512: (32, 32)}


def _nvidia(base: PlatformInfo, major: int, minor: int, budget: int) -> PlatformInfo:
    return dataclasses.replace(
        base,
        arch_version=ArchVersion(major, minor),
        max_shared_memory_per_sm=budget,
    )


@pytest.mark.parametrize(
    ("major", "minor", "budget", "expected"),
    [
        (9, 0, _SM90_BUDGET, _HOPPER_TILES),
        (10, 0, 262144, _HOPPER_TILES),
        (12, 0, _SM120_BUDGET, _SMALL_AMPERE_TILES),
        # Unknown budget on sm_120 keeps the pre-detection (datacenter) tiles.
        (12, 0, 0, _HOPPER_TILES),
        (8, 0, _SM80_BUDGET, _SM80_TILES),
        (8, 0, 0, _SM80_TILES),
        (8, 6, _SM120_BUDGET, _SMALL_AMPERE_TILES),
        # sm_86/sm_89 stay small even if detection yields no budget.
        (8, 6, 0, _SMALL_AMPERE_TILES),
        (8, 9, 0, _SMALL_AMPERE_TILES),
        (7, 5, 65536, _PRE_AMPERE_TILES),
    ],
    ids=[
        "sm90",
        "sm100",
        "sm120",
        "sm120-unknown",
        "sm80",
        "sm80-unknown",
        "sm86",
        "sm86-unknown",
        "sm89-unknown",
        "sm75",
    ],
)
@pytest.mark.parametrize("Lq", [64, 128, 256, 512])
def test_select_prefill_tiles(
    h100_platform, major, minor, budget, expected, Lq
) -> None:
    platform = _nvidia(h100_platform, major, minor, budget)
    assert _select_prefill_tiles(platform, Lq) == expected[Lq]


def test_select_prefill_tiles_conftest_platforms(
    h100_platform, a100_platform, b200_platform
) -> None:
    assert _select_prefill_tiles(h100_platform, 128) == (128, 64)
    assert _select_prefill_tiles(b200_platform, 128) == (128, 64)
    assert _select_prefill_tiles(a100_platform, 128) == (128, 128)

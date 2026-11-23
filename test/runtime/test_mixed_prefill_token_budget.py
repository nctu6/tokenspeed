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
"""``--mixed-prefill-token-budget``: the flag, its validation, and the
scheduler config it becomes."""

from __future__ import annotations

from test.ci_system.ci_register import register_cuda_ci

import pytest

register_cuda_ci(est_time=10, suite="runtime-1gpu")
ts = pytest.importorskip("tokenspeed_engine.scheduler")

from tokenspeed.runtime.engine.scheduler_utils import make_config
from tokenspeed.runtime.utils.server_args import prepare_server_args

BASE = ["--model", "x"]


def test_budget_defaults_to_the_decode_count_aware_bound() -> None:
    assert prepare_server_args(BASE).mixed_prefill_token_budget == 0


def test_budget_turns_mixed_batches_on() -> None:
    args = prepare_server_args(BASE + ["--mixed-prefill-token-budget", "1024"])
    assert args.mixed_prefill_token_budget == 1024
    assert args.enable_mixed_batch


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--mixed-prefill-token-budget", "-1"], "non-negative"),
        (
            ["--mixed-prefill-token-budget", "1024", "--disable-mixed-batch"],
            "disable-mixed-batch",
        ),
        (
            [
                "--mixed-prefill-token-budget",
                "2048",
                "--chunked-prefill-size",
                "1024",
            ],
            "must not exceed --chunked-prefill-size",
        ),
        (
            [
                "--mixed-prefill-token-budget",
                "512",
                "--disaggregation-mode",
                "prefill",
            ],
            "disaggregation-mode null",
        ),
    ],
)
def test_budget_rejects_unusable_combinations(extra: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_server_args(BASE + extra)


def _config(*, budget: int, mixed: bool, max_scheduled_tokens: int = 64):
    config = make_config(
        num_device_pages=32,
        max_scheduled_tokens=max_scheduled_tokens,
        max_batch_size=8,
        max_resident_requests=8,
        prefix_granularity=2,
        num_host_pages=0,
        disable_l2_cache=True,
        enable_l3_storage=False,
        role="fused",
        num_snapshot_pages=1,
        host_group_packing=(),
        max_retracted_requests=0,
        host_write_policy="write_through",
        prefill_decode_interval=0,
        prefill_decode_min_interval=0,
        mixed_prefill_token_budget=budget,
        enable_mixed_prefill_decode=mixed,
        storage_config=None,
    )
    config.cache_groups = [ts.CacheGroupConfig("full", 2, 32, ts.AttnKind.Full)]
    return config


def test_make_config_carries_the_budget_to_the_scheduler_config() -> None:
    assert _config(budget=48, mixed=True).mixed_prefill_token_budget == 48
    assert _config(budget=0, mixed=True).mixed_prefill_token_budget == 0


@pytest.mark.parametrize(("budget", "mixed"), [(16, False), (-1, True), (65, True)])
def test_scheduler_config_rejects_a_budget_it_cannot_honor(
    budget: int, mixed: bool
) -> None:
    with pytest.raises(ValueError, match="mixed_prefill_token_budget"):
        ts.Scheduler(_config(budget=budget, mixed=mixed))

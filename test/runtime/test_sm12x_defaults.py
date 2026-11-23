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
"""Startup defaults ServerArgs resolves from the detected device.

Workstation Blackwell (compute capability 12.x) trades KV pool against CUDA-graph
memory differently from datacenter parts: the prefill graph ladder is capped and
a checkpoint that fills the card caps the decode ladder. Explicit flags always
win and other architectures keep the engine defaults.
"""

import argparse
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed_kernel.platform import ArchVersion, InterconnectInfo, PlatformInfo

from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import ServerArgs

GPU_MIB = 96 * 1024


def _platform(major: int, minor: int) -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(major, minor),
        device_name="test",
        device_count=2,
        total_memory=GPU_MIB * 2**20,
        memory_bandwidth=1792.0,
        sm_count=188,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=101376,
        interconnect=InterconnectInfo(topology="pcie"),
    )


SM120 = _platform(12, 0)
SM90 = _platform(9, 0)


def _checkpoint(root, gib: int) -> str:
    """A model directory whose ``.safetensors`` files add up to ``gib`` GiB (sparse)."""
    root.mkdir(exist_ok=True)
    with open(root / "model-00001.safetensors", "wb") as f:
        f.truncate(gib * 2**30)
    return str(root)


def _resolve(argv: list[str], platform: PlatformInfo) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    with patch.object(ServerArgs, "__post_init__"):
        sa = ServerArgs.from_cli_args(parser.parse_args(argv))
    sa.resolve_basic_defaults()
    sa.resolve_parallelism()
    with (
        patch.object(server_args_module, "current_platform", return_value=platform),
        patch.object(
            server_args_module, "get_nvgpu_memory_capacity", return_value=GPU_MIB
        ),
    ):
        sa.resolve_memory_and_scheduling()
    return sa


class TestPrefillGraphCap:
    def test_sm120_caps_the_ladder_at_1024(self):
        sa = _resolve(["--model", "test/model"], SM120)
        assert sa.prefill_graph_max_tokens == 1024

    def test_other_architectures_keep_the_engine_default(self):
        sa = _resolve(["--model", "test/model"], SM90)
        assert sa.prefill_graph_max_tokens is None

    def test_the_cap_follows_a_smaller_chunk(self):
        sa = _resolve(["--model", "test/model", "--chunked-prefill-size", "512"], SM120)
        assert sa.prefill_graph_max_tokens == 512

    @pytest.mark.parametrize("value", ["0", "4096"])
    def test_explicit_value_wins(self, value):
        sa = _resolve(
            ["--model", "test/model", "--prefill-graph-max-tokens", value], SM120
        )
        assert sa.prefill_graph_max_tokens == int(value)

    @pytest.mark.parametrize(
        "flags",
        [["--enforce-eager"], ["--disable-prefill-graph"]],
        ids=["eager", "disabled"],
    )
    def test_no_cap_when_prefill_graphs_are_off(self, flags):
        assert (
            _resolve(["--model", "test/model", *flags], SM120).prefill_graph_max_tokens
            is None
        )

    def test_explicit_bucket_list_is_not_clamped(self):
        sa = _resolve(
            [
                "--model",
                "test/model",
                "--prefill-graph-capture-token-sizes",
                "2048",
                "4096",
            ],
            SM120,
        )
        assert sa.prefill_graph_max_tokens is None


class TestDecodeGraphCap:
    def test_checkpoint_filling_the_card_caps_the_ladder(self, tmp_path):
        model = _checkpoint(tmp_path / "big", 60)
        sa = _resolve(["--model", model, "--max-num-seqs", "452"], SM120)
        assert sa.max_cudagraph_capture_size == 128

    def test_small_checkpoint_keeps_the_full_ladder(self, tmp_path):
        model = _checkpoint(tmp_path / "small", 16)
        sa = _resolve(["--model", model, "--max-num-seqs", "452"], SM120)
        assert sa.max_cudagraph_capture_size == 452

    def test_batch_within_the_cap_is_unchanged(self, tmp_path):
        model = _checkpoint(tmp_path / "big", 60)
        sa = _resolve(["--model", model, "--max-num-seqs", "64"], SM120)
        assert sa.max_cudagraph_capture_size == 64

    def test_explicit_value_wins(self, tmp_path):
        model = _checkpoint(tmp_path / "big", 60)
        sa = _resolve(
            [
                "--model",
                model,
                "--max-num-seqs",
                "452",
                "--max-cudagraph-capture-size",
                "256",
            ],
            SM120,
        )
        assert sa.max_cudagraph_capture_size == 256

    def test_other_architectures_keep_the_full_ladder(self, tmp_path):
        model = _checkpoint(tmp_path / "big", 60)
        sa = _resolve(["--model", model, "--max-num-seqs", "452"], SM90)
        assert sa.max_cudagraph_capture_size == 452

    def test_two_gpus_keep_the_full_ladder(self, tmp_path):
        model = _checkpoint(tmp_path / "big", 60)
        sa = _resolve(
            ["--model", model, "--max-num-seqs", "452", "--tensor-parallel-size", "2"],
            SM120,
        )
        assert sa.max_cudagraph_capture_size == 452

    def test_hub_id_without_a_local_checkpoint_is_not_guessed(self):
        sa = _resolve(
            ["--model", "org/not-a-local-dir", "--max-num-seqs", "452"], SM120
        )
        assert sa.max_cudagraph_capture_size == 452

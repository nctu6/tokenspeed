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


"""The decode graph ladder ``--cudagraph-capture-sizes`` builds, and the graph a
batch pads to."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

import pytest  # noqa: E402

from tokenspeed.runtime.execution.forward_step import (  # noqa: E402
    get_batch_sizes_to_capture,
)
from tokenspeed.runtime.execution.model_executor import (  # noqa: E402
    ModelExecutorConfig,
)
from tokenspeed.runtime.utils.common import get_nvgpu_memory_capacity  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _config(
    sizes: list[int] | None,
    *,
    max_num_seqs: int,
    max_capture: int,
    disable_padding: bool = False,
):
    return ModelExecutorConfig(
        max_req_pool_size=max_num_seqs + 1,
        output_length=1,
        enforce_eager=False,
        prefix_granularity=128,
        max_num_seqs=max_num_seqs,
        chunked_prefill_size=4096,
        vocab_size=32,
        context_len=4096,
        physical_context_len=4096,
        device="cpu",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=sizes,
        disable_cuda_graph_padding=disable_padding,
        spec_topk=1,
        max_cudagraph_capture_size=max_capture,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        input_logprob_chunk_tokens=1024,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        prefill_graph_max_tokens=256,
        enable_memory_saver=False,
    )


@pytest.mark.parametrize(
    "sizes,max_num_seqs,max_capture,ladder",
    [
        # Given out of order, with a repeat: the ladder still ascends.
        ([8, 4, 16, 4, 1], 64, 64, [1, 4, 8, 16]),
        # Above max_num_seqs: the cap and the size below it join the ladder.
        ([64, 8, 2], 32, 64, [2, 8, 31, 32]),
        # The capture cap drops the larger sizes.
        ([32, 1, 16, 8], 64, 16, [1, 8, 16]),
    ],
)
def test_the_ladder_ascends_whatever_order_the_sizes_come_in(
    sizes, max_num_seqs, max_capture, ladder
) -> None:
    config = _config(sizes, max_num_seqs=max_num_seqs, max_capture=max_capture)
    assert get_batch_sizes_to_capture(config) == ladder


def test_default_ladder_below_the_old_cap_is_unchanged() -> None:
    config = _config(None, max_num_seqs=160, max_capture=160)
    assert get_batch_sizes_to_capture(config) == [1, 2, 4] + list(range(8, 161, 8))


def test_default_ladder_reaches_max_num_seqs() -> None:
    config = _config(None, max_num_seqs=512, max_capture=512)
    ladder = get_batch_sizes_to_capture(config)
    assert ladder[-1] == 512
    assert ladder[: len(range(8, 161, 8)) + 3] == [1, 2, 4] + list(range(8, 161, 8))
    assert ladder[-9:] == [256] + list(range(288, 513, 32))
    # From 81 requests up, a batch pads to a graph with under 11% waste.
    for bs in range(81, 513):
        top = min(size for size in ladder if size >= bs)
        assert (top - bs) / top < 0.11, (bs, top)


def test_default_ladder_ends_at_an_unaligned_max_num_seqs() -> None:
    config = _config(None, max_num_seqs=300, max_capture=300)
    assert get_batch_sizes_to_capture(config)[-3:] == [256, 288, 300]


def test_default_ladder_honours_an_explicit_cap() -> None:
    config = _config(None, max_num_seqs=512, max_capture=160)
    assert get_batch_sizes_to_capture(config)[-1] == 160
    config = _config(None, max_num_seqs=512, max_capture=200)
    assert get_batch_sizes_to_capture(config)[-3:] == [160, 176, 192]


def test_explicit_sizes_are_not_extended() -> None:
    config = _config([1, 8, 160], max_num_seqs=512, max_capture=512)
    assert get_batch_sizes_to_capture(config) == [1, 8, 160]


def test_unpadded_default_ladder_reaches_max_num_seqs() -> None:
    config = _config(None, max_num_seqs=256, max_capture=256, disable_padding=True)
    ladder = get_batch_sizes_to_capture(config)
    assert ladder[-4:] == [160, 192, 224, 256]
    assert ladder[:32] == list(range(1, 33))


def _server_args(**kwargs) -> ServerArgs:
    return ServerArgs(
        model="unused", disable_kvstore=True, enable_prefix_caching=False, **kwargs
    )


@pytest.mark.parametrize("max_num_seqs", [64, 160, 512])
def test_server_args_capture_size_follows_max_num_seqs(max_num_seqs) -> None:
    if (get_nvgpu_memory_capacity() or 0) < 25_000:
        pytest.skip("small GPUs clamp the capture size by memory")
    assert _server_args(max_num_seqs=max_num_seqs).max_cudagraph_capture_size == (
        max_num_seqs
    )


def test_server_args_keep_an_explicit_capture_size() -> None:
    args = _server_args(max_num_seqs=512, max_cudagraph_capture_size=160)
    assert args.max_cudagraph_capture_size == 160


def test_server_args_without_the_memory_reserve_keep_the_old_default() -> None:
    if (get_nvgpu_memory_capacity() or 0) < 25_000:
        pytest.skip("small GPUs clamp the capture size by memory")
    args = _server_args(max_num_seqs=512, disable_cudagraph_memory_reserve=True)
    assert args.max_cudagraph_capture_size == 160

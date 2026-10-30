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

"""The cost of a Whisper prefill step that carries fresh audio.

Properties of the fixed prefill step:

* The forward performs no device-to-host synchronization.
* Its kernel launch count does not grow with the number of fresh requests
  (the cross-KV write is batched, not one launch sequence per request per
  layer).
* The decoder prefill is eligible for the prefill CUDA graph.
"""

import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from test.runtime.models.whisper_fixtures import (  # noqa: E402
    Request,
    WhisperRuntime,
    audio_features,
    build_whisper,
    cuda_kernels,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Whisper runtime needs CUDA"
)

PROMPT = [1, 2, 3, 4]
NUM_SLOTS = 17


def _fresh_requests(count: int, seed: int, tag: str) -> list[Request]:
    """``count`` new requests in random slots; ``tag`` names this admission.

    A warm-up and the measured forward use different tags: they are
    different requests landing in the same slots, so both run the encoder
    (the model skips it for a request whose slot already holds its audio).
    """
    generator = torch.Generator().manual_seed(seed)
    slots = torch.randperm(NUM_SLOTS, generator=generator)[:count].tolist()
    return [
        Request(slot, row, PROMPT, audio_features(seed * 100 + row), rid=f"{tag}{row}")
        for row, slot in enumerate(slots)
    ]


@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_fresh_audio_prefill_forward_never_syncs_the_host(seed):
    """**Validates: Requirements 1.5, 2.5** (Property 1, prefill: host syncs)."""
    model = build_whisper(NUM_SLOTS, seed=seed)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    # warm-up: compilation may sync
    runtime.extend(_fresh_requests(4, seed, "warm"), [0] * 4)()
    run = runtime.extend(_fresh_requests(4, seed, "run"), [0] * 4)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_fresh_audio_prefill_launches_do_not_grow_with_the_request_count(seed):
    """**Validates: Requirements 1.5, 2.5** (Property 1, prefill: launches).

    The whole prefill forward launches the same number of kernels for 2, 3
    and 8 fresh requests: no launch sequence runs once per request (the
    unfixed per-request x per-layer write grew 4 kernels per request on this
    2-layer model, {1: 85, 3: 91, 8: 111}).

    A single request is left out on purpose: at batch 1 the libraries pick
    different kernels for the same work (cuDNN conv algorithm without the
    NCHW/NHWC transforms, cuBLAS GEMV instead of GEMM, a different embedding
    gather), which changes the total by a constant, not per request. The
    write alone is compared down to one request in
    ``test_whisper_cross_kv_write.py``.
    """
    model = build_whisper(NUM_SLOTS, seed=seed)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    counts = {}
    for count in (2, 3, 8):
        warm = _fresh_requests(count, seed, f"warm{count}-")
        runtime.extend(warm, [0] * count)()  # warm-up
        measured = _fresh_requests(count, seed, f"run{count}-")
        counts[count] = len(cuda_kernels(runtime.extend(measured, [0] * count)))
    print(f"prefill kernel launches by fresh request count: {counts}")
    assert (
        len(set(counts.values())) == 1
    ), f"kernel launches per prefill step grow with fresh requests: {counts}"


@dataclasses.dataclass
class _ModelConfig:
    """The one model-config field PrefillGraph reads."""

    requires_request_token_history: bool


@dataclasses.dataclass
class _ModelRunner:
    """The ModelRunner fields PrefillGraph reads (the real one loads weights)."""

    model: torch.nn.Module
    model_config: _ModelConfig
    is_generation: bool
    is_multimodal: bool


def _executor_config(prefill_graph_max_tokens: int = 256):
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    return ModelExecutorConfig(
        max_req_pool_size=NUM_SLOTS,
        output_length=1,
        enforce_eager=False,
        prefix_granularity=16,
        max_num_seqs=NUM_SLOTS - 1,
        chunked_prefill_size=1024,
        vocab_size=128,
        context_len=64,
        physical_context_len=64,
        device="cuda",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=16,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        input_logprob_chunk_tokens=1024,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=None,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        prefill_graph_max_tokens=prefill_graph_max_tokens,
    )


@torch.inference_mode()
def test_whisper_audio_prefill_is_eligible_for_the_prefill_graph():
    """**Validates: Requirements 1.5, 2.1, 2.5** (Property 1, prefill graph).

    The prefill graph is built for Whisper (not disabled for the model), and a
    forward carrying audio is not refused for being multimodal: it gets the
    same answer as the same forward without audio.
    """
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    model = build_whisper(NUM_SLOTS, seed=0)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    graph = PrefillGraph(
        model_runner=_ModelRunner(
            model=model,
            model_config=_ModelConfig(requires_request_token_history=False),
            is_generation=True,
            is_multimodal=True,
        ),
        attn_backend=runtime.router,
        token_to_kv_pool=runtime.pool,
        input_buffers=None,
        config=_executor_config(),
    )
    assert not graph.disable, (
        "the prefill CUDA graph is disabled for Whisper: its decoder prefill "
        "always runs eager"
    )
    probe = _ForwardProbe(runtime, _fresh_requests(2, seed=0, tag="probe"))
    assert graph.can_run(probe.ctx, probe.multimodal_context) == graph.can_run(
        probe.ctx, None
    ), "a forward carrying Whisper audio is refused the prefill graph"


class _ForwardProbe:
    """The ForwardContext and multimodal context of one Whisper audio prefill."""

    def __init__(self, runtime: WhisperRuntime, requests: list[Request]) -> None:
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
        from tokenspeed.runtime.multimodal.inputs import (
            Modality,
            MultimodalDataItem,
            MultimodalForwardContext,
            MultimodalInputs,
        )

        bs = len(requests)
        tokens = sum(len(r.tokens) for r in requests)
        self.ctx = ForwardContext(
            attn_backend=runtime.router,
            token_to_kv_pool=runtime.pool,
            bs=bs,
            num_extends=bs,
            input_num_tokens=tokens,
            forward_mode=ForwardMode.EXTEND,
            output_layout=ForwardOutputLayout(bs, bs, 0, 1),
        )
        self.multimodal_context = MultimodalForwardContext(
            mm_inputs=[
                MultimodalInputs(
                    mm_items=[
                        MultimodalDataItem(modality=Modality.AUDIO, feature=r.audio)
                    ]
                )
                for r in requests
            ],
            request_ids=[r.rid for r in requests],
            request_pool_indices=[r.slot for r in requests],
            extend_prefix_lens=[0] * bs,
            extend_seq_lens=[len(r.tokens) for r in requests],
            max_encoder_tokens=4096,
        )


class _LiveExtend:
    """One fresh-audio extend laid into the executor's persistent input buffers.

    What ``ModelExecutor`` does before a prefill forward, for the fields the
    Whisper forward and the prefill graph read: ids, positions and slots in
    the static buffers, the attention metadata, the context and the
    multimodal context.
    """

    def __init__(self, runtime: WhisperRuntime, ib, requests: list[Request]) -> None:
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
        from tokenspeed.runtime.multimodal.inputs import (
            Modality,
            MultimodalDataItem,
            MultimodalForwardContext,
            MultimodalInputs,
        )

        self.runtime = runtime
        self.ib = ib
        self.requests = requests
        bs = len(requests)
        lens = [len(r.tokens) for r in requests]
        num_tokens = sum(lens)
        self.bs = bs
        self.num_tokens = num_tokens
        ib.input_ids_buf[:num_tokens].copy_(
            torch.tensor([t for r in requests for t in r.tokens], dtype=torch.int32)
        )
        ib.positions_buf[:num_tokens].copy_(
            torch.tensor([i for n in lens for i in range(n)], dtype=torch.int64)
        )
        ib.req_pool_indices_buf[:bs].copy_(
            torch.tensor([r.slot for r in requests], dtype=torch.int32)
        )
        lens_cpu = torch.tensor(lens, dtype=torch.int32)
        self._lens_cpu = lens_cpu
        self.ctx = ForwardContext(
            attn_backend=runtime.router,
            token_to_kv_pool=runtime.pool,
            bs=bs,
            num_extends=bs,
            input_num_tokens=num_tokens,
            forward_mode=ForwardMode.EXTEND,
            output_layout=ForwardOutputLayout(bs, bs, 0, 1),
            gather_ids=torch.cumsum(lens_cpu, 0).cuda().to(torch.int64) - 1,
        )
        self.multimodal_context = MultimodalForwardContext(
            mm_inputs=[
                MultimodalInputs(
                    mm_items=[
                        MultimodalDataItem(modality=Modality.AUDIO, feature=r.audio)
                    ]
                )
                for r in requests
            ],
            request_ids=[r.rid for r in requests],
            request_pool_indices=[r.slot for r in requests],
            extend_prefix_lens=[0] * bs,
            extend_seq_lens=lens,
            max_encoder_tokens=4096,
        )

    @property
    def input_ids(self) -> torch.Tensor:
        return self.ib.input_ids_buf[: self.num_tokens]

    @property
    def positions(self) -> torch.Tensor:
        return self.ib.positions_buf[: self.num_tokens]

    def init_metadata(self) -> None:
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        lens = self._lens_cpu
        zeros = torch.zeros_like(lens)
        tables = self.runtime._tables([r.page_row for r in self.requests])
        self.runtime.router.init_forward_metadata(
            bs=self.bs,
            num_extends=self.bs,
            req_pool_indices=self.ib.req_pool_indices_buf[: self.bs],
            seq_lens=lens.cuda(),
            forward_mode=ForwardMode.EXTEND,
            block_tables=tables,
            extend_seq_lens=lens.cuda(),
            extend_seq_lens_cpu=lens,
            extend_prefix_lens=zeros.cuda(),
            extend_prefix_lens_cpu=zeros,
            extend_replay_lens_cpu=zeros,
            extend_prompt_lens_cpu=lens,
            extend_with_prefix=False,
            query_shard=None,
            block_tables_cpu={gid: t.cpu() for gid, t in tables.items()},
        )

    def eager(self) -> torch.Tensor:
        """The executor's eager route: ``model_runner.forward``."""
        self.init_metadata()
        out = self.runtime.model(
            self.ctx,
            self.input_ids,
            self.positions,
            req_pool_indices=self.ib.req_pool_indices_buf[: self.bs],
            multimodal_context=self.multimodal_context,
        )
        return out.next_token_logits.float().clone()

    def replay(self, graph) -> torch.Tensor:
        """The executor's graph route: ``PrefillGraph.replay``."""
        self.init_metadata()
        assert graph.can_run(self.ctx, self.multimodal_context)
        out = graph.replay(self.ctx, self.input_ids, self.multimodal_context)
        return out.next_token_logits.float().clone()


def test_whisper_audio_prefill_graph_replay_equals_eager():
    """**Validates: Requirements 1.5, 2.5, 3.6** (Property 1, prefill graph;
    Property 4, graph == eager).

    A real ``PrefillGraph`` captures Whisper's decoder per token bucket from
    a one-request dummy batch; live batches of other request counts and
    slots, whose audio is encoded in the seam, replay it. The replayed
    logits equal the eager forward's on the same inputs: the cross-attention
    read follows the live slots and request count, not the capture's.
    """
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    max_tokens = 32
    model = build_whisper(NUM_SLOTS, seed=3)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    # As ForwardStepRunner sizes it: the padded extend write span a captured
    # prologue records must cover the largest prefill bucket.
    runtime.router.init_cuda_graph_state(NUM_SLOTS, max_extend_tokens=max_tokens)
    ib = InputBuffers(
        max_bs=NUM_SLOTS,
        max_num_tokens=64,
        state_write_padding_pool_index=0,
        device="cuda",
    )
    graph = PrefillGraph(
        model_runner=_ModelRunner(
            model=model,
            model_config=_ModelConfig(requires_request_token_history=False),
            is_generation=True,
            is_multimodal=True,
        ),
        attn_backend=runtime.router,
        token_to_kv_pool=runtime.pool,
        input_buffers=ib,
        config=_executor_config(prefill_graph_max_tokens=max_tokens),
    )
    assert not graph.disable
    graph.capture(None, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    # Buckets 16 and 32 tokens; at context 64 their dummy batches hold 1 request.
    assert graph.capture_buckets == [16, 32]

    generator = torch.Generator().manual_seed(11)
    batches = [
        # (slots, prompt lengths): bucket 16 with 1 and 3 requests, bucket 32
        # with 5.
        ([12], [7]),
        ([5, 0, 16], [4, 3, 5]),
        ([2, 9, 14, 7, 11], [4, 6, 5, 3, 4]),
    ]
    for index, (slots, lens) in enumerate(batches):
        requests = [
            Request(
                slot,
                row,
                torch.randint(0, 128, (n,), generator=generator).tolist(),
                audio_features(1000 * index + row),
                rid=f"b{index}r{row}",
            )
            for row, (slot, n) in enumerate(zip(slots, lens))
        ]
        live = _LiveExtend(runtime, ib, requests)
        with torch.inference_mode():
            # Replay first: its seam encodes the fresh audio, so the eager
            # forward after it reads the same cross KV without re-encoding.
            replayed = live.replay(graph)
            eager = live.eager()
        torch.cuda.synchronize()
        assert replayed.shape == (len(requests), 128)
        assert torch.isfinite(replayed).all()
        diff = (replayed - eager).abs().max().item()
        print(f"batch {index}: max |replay - eager| = {diff:.3e}")
        torch.testing.assert_close(replayed, eager, atol=2e-2, rtol=2e-2)

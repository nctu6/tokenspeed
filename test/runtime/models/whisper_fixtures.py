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

"""A tiny Whisper driven through the real attention runtime.

The model is the production ``WhisperForConditionalGeneration`` at test
dimensions with random weights (the unit environment has no checkpoint). The
decoder self-attention runs on the production paged stack: an
``MHATokenToKVPool`` over a real arena, the ``CacheGroupRouter`` built by
``create_paged_router`` and bound with ``set_cache_pool``, and layer cache
groups bound by ``bind_cache_groups``. The encoder window is the real
``encoder_len = 1500``, because the cross-attention read cost scales with it.

Each request owns a fixed row of ``PAGES_PER_REQUEST`` pages, so a test names
requests by (cross-attention slot, page row) the way the scheduler would.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import torch

# Test dimensions. head_dim 64 matches whisper-large-v3 (d_model 1280 / 20).
D_MODEL = 128
HEADS = 2
HEAD_DIM = D_MODEL // HEADS
DECODER_LAYERS = 2
ENCODER_LAYERS = 1
NUM_MEL_BINS = 16
ENCODER_LEN = 1500
# The encoder's conv2 has stride 2, so the feature window is twice the encoder length.
FEATURE_FRAMES = 2 * ENCODER_LEN
VOCAB = 128
PAGE = 16
PAGES_PER_REQUEST = 4
CONTEXT_LEN = PAGE * PAGES_PER_REQUEST


def whisper_config(num_slots: int):
    """A tiny HF ``WhisperConfig`` stamped with the cross-attention slot count."""
    from transformers import WhisperConfig

    config = WhisperConfig(
        vocab_size=VOCAB,
        num_mel_bins=NUM_MEL_BINS,
        encoder_layers=ENCODER_LAYERS,
        encoder_attention_heads=HEADS,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=HEADS,
        encoder_ffn_dim=2 * D_MODEL,
        decoder_ffn_dim=2 * D_MODEL,
        d_model=D_MODEL,
        max_source_positions=ENCODER_LEN,
        max_target_positions=CONTEXT_LEN,
        activation_function="gelu",
    )
    # ModelConfig stamps this from the request pool size in production.
    config.cross_attn_slots = num_slots
    return config


def build_whisper(
    num_slots: int, seed: int, *, mm_attention_backend: str = "triton_attn"
):
    """The production model at test dimensions, with seeded random weights.

    Two calls with the same seed give identical weights, whatever ``num_slots``
    and encoder attention backend.
    """
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.whisper import WhisperForConditionalGeneration

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = WhisperForConditionalGeneration(
                whisper_config(num_slots),
                Mapping(rank=0, world_size=1),
                mm_attention_backend=mm_attention_backend,
            )
    finally:
        torch.set_default_dtype(previous)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "layer_norm" in name:
                param.fill_(1.0 if name.endswith("weight") else 0.0)
            else:
                param.normal_(0.0, 0.05, generator=generator)
        model.encoder.embed_positions.normal_(0.0, 0.05, generator=generator)
    return model.eval()


def audio_features(seed: int) -> torch.Tensor:
    """One request's log-mel window, on the host as the request carries it."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_MEL_BINS, FEATURE_FRAMES, generator=generator)


@dataclasses.dataclass
class Request:
    """One request as a forward sees it."""

    slot: int
    page_row: int
    tokens: list[int]
    audio: torch.Tensor
    # The scheduler's request id: the model encodes a request's audio iff its
    # slot does not already hold this request's encoder output.
    rid: str


class WhisperRuntime:
    """The paged decoder-attention runtime the model forwards against."""

    def __init__(self, model, *, max_bs: int) -> None:
        from test.runtime.cache_pool_test_utils import make_mha_memory_plan, make_pool

        from tokenspeed.runtime.configs.model_config import AttentionArch
        from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
        from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
        from tokenspeed.runtime.layers.attention.kv_cache.mha import (
            MHATokenToKVPool,
        )
        from tokenspeed.runtime.layers.attention.registry import create_paged_router
        from tokenspeed.runtime.layers.paged_attention import bind_cache_groups

        self.model = model
        self.max_bs = max_bs
        spec = MHAConfig(
            backend_name="triton",
            num_attention_heads=HEADS,
            num_kv_heads=HEADS,
            head_dim=HEAD_DIM,
            attn_tp_size=1,
        )
        attn_config = AttnConfig(
            device="cuda",
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            prefix_granularity=PAGE,
            kernel_page_size=PAGE,
            context_len=CONTEXT_LEN,
            max_bs=max_bs,
            components=(spec,),
        )
        # Page 0 is the null page; every request row needs its own pages after it.
        num_pages = 1 + max_bs * PAGES_PER_REQUEST
        _, self.pool = make_pool(
            MHATokenToKVPool,
            make_mha_memory_plan(
                size=num_pages * PAGE,
                prefix_granularity=PAGE,
                layer_num=DECODER_LAYERS,
                kv_heads=HEADS,
                head_dim=HEAD_DIM,
                dtype=torch.bfloat16,
            ),
            device="cuda",
            dtype=torch.bfloat16,
            head_num=HEADS,
            head_dim=HEAD_DIM,
            layer_num=DECODER_LAYERS,
            rank=0,
        )
        self.router = create_paged_router(
            attn_config, AttentionArch.MHA, backend_name="triton"
        )
        self.router.set_cache_pool(self.pool)
        bind_cache_groups(model, self.pool)
        self.router.init_cuda_graph_state(max_bs)
        (self.group_id,) = self.router.group_ids

    def _tables(self, page_rows: Sequence[int]) -> dict[str, torch.Tensor]:
        rows = [
            [1 + row * PAGES_PER_REQUEST + page for page in range(PAGES_PER_REQUEST)]
            for row in page_rows
        ]
        return {self.group_id: torch.tensor(rows, dtype=torch.int32, device="cuda")}

    def extend(
        self, requests: Sequence[Request], prefix_lens: Sequence[int]
    ) -> Callable[[], object]:
        """Prepare one extend forward and return the call that runs it.

        Request ``i`` feeds ``tokens[prefix_lens[i]:]`` with its audio attached,
        as ``multimodal_context_for_forward`` attaches a request's inputs on
        every chunk of its prefill.
        """
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
        extend_lens = [len(r.tokens) - p for r, p in zip(requests, prefix_lens)]
        seq_lens = [len(r.tokens) for r in requests]
        prefix_cpu = torch.tensor(prefix_lens, dtype=torch.int32)
        extend_cpu = torch.tensor(extend_lens, dtype=torch.int32)
        seq_cpu = torch.tensor(seq_lens, dtype=torch.int32)
        slots = torch.tensor(
            [r.slot for r in requests], dtype=torch.int32, device="cuda"
        )
        tables = self._tables([r.page_row for r in requests])
        self.router.init_forward_metadata(
            bs=bs,
            num_extends=bs,
            req_pool_indices=slots,
            seq_lens=seq_cpu.cuda(),
            forward_mode=ForwardMode.EXTEND,
            block_tables=tables,
            extend_seq_lens=extend_cpu.cuda(),
            extend_seq_lens_cpu=extend_cpu,
            extend_prefix_lens=prefix_cpu.cuda(),
            extend_prefix_lens_cpu=prefix_cpu,
            extend_replay_lens_cpu=torch.zeros_like(prefix_cpu),
            extend_prompt_lens_cpu=seq_cpu,
            extend_with_prefix=any(prefix_lens),
            query_shard=None,
            block_tables_cpu={gid: t.cpu() for gid, t in tables.items()},
        )
        input_ids = torch.tensor(
            [t for r, p in zip(requests, prefix_lens) for t in r.tokens[p:]],
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.tensor(
            [i for r, p in zip(requests, prefix_lens) for i in range(p, len(r.tokens))],
            dtype=torch.int64,
            device="cuda",
        )
        ctx = ForwardContext(
            attn_backend=self.router,
            token_to_kv_pool=self.pool,
            bs=bs,
            num_extends=bs,
            input_num_tokens=int(input_ids.shape[0]),
            forward_mode=ForwardMode.EXTEND,
            output_layout=ForwardOutputLayout(bs, bs, 0, 1),
            gather_ids=torch.cumsum(extend_cpu, 0).cuda().to(torch.int64) - 1,
        )
        multimodal_context = MultimodalForwardContext(
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
            extend_prefix_lens=list(prefix_lens),
            extend_seq_lens=extend_lens,
            max_encoder_tokens=4096,
        )

        def run():
            return self.model(
                ctx,
                input_ids,
                positions,
                req_pool_indices=slots,
                multimodal_context=multimodal_context,
            )

        return run

    def decode(
        self,
        slots: Sequence[int],
        page_rows: Sequence[int],
        seq_lens: Sequence[int],
        tokens: Sequence[int] | None = None,
    ) -> "DecodeStep":
        """Prepare one decode forward (one new token per request) and return its call.

        ``tokens`` are the tokens fed at position ``seq_len - 1``; omitted, request
        ``i`` feeds token ``i``.
        """
        step = DecodeStep(self, len(slots))
        step.refresh(slots, page_rows, seq_lens, tokens)
        return step


class DecodeStep:
    """One decode forward over fixed-address inputs, as a CUDA graph replays it.

    :meth:`refresh` rewrites the slots, tokens and positions in place and
    refreshes the decode metadata in place, so a graph captured over
    ``__call__`` replays a new batch of the same size.
    """

    def __init__(self, runtime: WhisperRuntime, bs: int) -> None:
        from tokenspeed.runtime.execution.context import ForwardContext
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
        from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

        self.runtime = runtime
        self.bs = bs
        self.slots = torch.zeros(bs, dtype=torch.int32, device="cuda")
        self.seq_lens = torch.zeros(bs, dtype=torch.int32, device="cuda")
        self.input_ids = torch.zeros(bs, dtype=torch.int64, device="cuda")
        self.positions = torch.zeros(bs, dtype=torch.int64, device="cuda")
        self.ctx = ForwardContext(
            attn_backend=runtime.router,
            token_to_kv_pool=runtime.pool,
            bs=bs,
            num_extends=0,
            input_num_tokens=bs,
            forward_mode=ForwardMode.DECODE,
            output_layout=ForwardOutputLayout(0, 0, bs, 1),
        )

    def refresh(
        self,
        slots: Sequence[int],
        page_rows: Sequence[int],
        seq_lens: Sequence[int],
        tokens: Sequence[int] | None,
    ) -> None:
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        if tokens is None:
            tokens = [i % VOCAB for i in range(self.bs)]
        assert len(slots) == len(page_rows) == len(seq_lens) == len(tokens) == self.bs
        self.slots.copy_(torch.tensor(slots, dtype=torch.int32))
        self.seq_lens.copy_(torch.tensor(seq_lens, dtype=torch.int32))
        self.input_ids.copy_(torch.tensor(tokens, dtype=torch.int64))
        self.positions.copy_(torch.tensor(seq_lens, dtype=torch.int64) - 1)
        self.runtime.router.refresh_decode_metadata(
            self.bs,
            self.bs,
            self.slots,
            self.seq_lens,
            forward_mode=ForwardMode.DECODE,
            block_tables=self.runtime._tables(page_rows),
        )

    def __call__(self):
        return self.runtime.model(
            self.ctx, self.input_ids, self.positions, req_pool_indices=self.slots
        )


def cuda_kernels(run: Callable[[], object]) -> list[str]:
    """The names of the device kernels ``run`` launches (copies and fills excluded)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    return sorted(
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
    )


def peak_bytes(run: Callable[[], object]) -> int:
    """Device memory ``run`` allocates above what was live before it."""
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    run()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base

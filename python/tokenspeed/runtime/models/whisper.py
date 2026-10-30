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

"""Whisper (encoder-decoder ASR).

Structure, and why it differs from the other ASR model in this tree:

``qwen3_asr.py`` is decoder-only with an audio prefix -- the audio tower's
output is spliced into the LM's input embeddings and the rest is ordinary
causal decode. Whisper is a real encoder-decoder: the decoder cross-attends
to a fixed encoder output that is computed once and read by every decode
step. That fixed KV is the whole design question, and the four reference
engines disagree about it (``unieinfra/xref/15-encoder-embedding.md`` §3.3):
vLLM gives it its own KV spec, manager and cache group; TRT-LLM gives it a
separate resource manager with its own slice of the KV budget; SGLang folds
it into the ordinary paged pool by offsetting the layer id so cross-attention
looks like ``encoder_layers + decoder_layers + i``.

We are on vLLM/TRT-LLM's side -- cross KV lives outside the paged arena --
but implemented as a plain preallocated per-slot buffer rather than a fourth
cache family, because cross KV needs nothing the arena provides:

  * it never grows (one write of ``encoder_len`` rows per request),
  * it is never prefix-matched (encoder output is unique per audio, which is
    why vLLM's CrossAttentionManager raises on ``find_longest_cache_hit``),
  * it is never evicted independently of its request.

The cost is bounded: ``decoder_layers x padded_len x d_model x 2`` per slot
(``padded_len`` is ``encoder_len`` rounded up to whole pages, see below) --
251.7 MB per request on large-v3 (32 decoder layers, 1500 frames padded to
1536). Keeping it out of the arena also keeps this port off the shared C++
scheduler.

Cross-attention reads that buffer in place, through the ordinary paged MHA
ops of ``tokenspeed-kernel``. Each slot's rows are a fixed run of
``pages_per_slot`` pages, so a page table derived on device from
``req_pool_indices`` points every request at its own rows, and
``cache_seqlens = encoder_len`` masks the padding rows. That is one read
path for decode and extend, eager and CUDA graph: it touches exactly the
batch's ``bs x encoder_len`` rows per layer, copies nothing, and its cost does
not depend on how many slots the pool has.

Prefill uses the breakable prefill CUDA graph like any causal LM
(``execution/prefill_graph.py``): the decoder stack is the captured inner
model, the encoder and the cross-KV write run in the
``multimodal_input_embeds`` seam before it, and the cross-attention read is a
``break_point`` next to the self-attention one, so it reads the live batch's
slots and query boundaries on every replay. The seam's encoder pass has one
fixed shape per request count (Whisper's window is fixed), so with the
multimodal encoder graphs enabled it replays a CUDA graph captured per
request count (:class:`AudioEncodeRunner`); the eager encoder is host-bound,
and the replay lets the host enqueue the decoder while the encoder runs.

Because the buffer is keyed by ``req_pool_index`` and lives in no cache
group, it is per-slot state a retraction snapshot must carry
(``execution/slot_state.py``): :class:`CrossAttentionCache` is a
``SlotStateExporter`` whose image is the slot's ``encoder_len`` rows of every
layer behind a request-keyed prepared marker, so a suspended request resumes
reading its own encoder output from whatever slot it is restored into.

The decoder carries the residual stream next to each sublayer's output and
adds them inside the next LayerNorm (``tokenspeed-kernel``'s
``add_layernorm``: one launch per pre-norm boundary instead of an add plus a
norm, same rounding as the eager pair). Its dense projections take
``tokenspeed-kernel``'s ``mm`` selection like any other linear; on SM120 a
decode step's 1-16 rows select the skinny BF16 GEMM there.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
)
from tokenspeed_kernel.ops.layernorm import add_layernorm
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature
from torch import nn
from transformers.activations import ACT2FN

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.slot_state import (
    PREPARED_MARKER_BYTES,
    pack_slot_rows,
    read_prepared_marker,
    slot_state_image_bytes,
    unpack_slot_rows,
    write_prepared_marker,
)
from tokenspeed.runtime.layers.attention.mm_encoder_attention import (
    MultimodalEncoderAttention,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalForwardContext
from tokenspeed.runtime.utils import add_prefix, get_colorful_logger

logger = get_colorful_logger(__name__)


def _add_layer_norm(
    hidden_states: torch.Tensor, residual: torch.Tensor, norm: nn.LayerNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """``norm(residual + hidden_states)`` and the sum, in one kernel launch.

    The decoder's pre-norm boundary; the sum and the norm round as the eager
    ``residual + hidden_states`` and ``nn.LayerNorm`` do. The module only
    holds the checkpoint's weight, bias and eps.
    """
    return add_layernorm(hidden_states, residual, norm.weight, norm.bias, eps=norm.eps)


class WhisperEncoderLayer(nn.Module):
    """One encoder block: bidirectional attention, no KV cache.

    Reuses ``MultimodalEncoderAttention`` -- the same non-causal, pool-free
    layer the vision and audio towers use. Whisper's encoder is exactly that
    shape, so it needs no encoder-specific attention code.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.self_attn = MultimodalEncoderAttention(
            embed_dim=embed_dim,
            num_heads=config.encoder_attention_heads,
            mapping=mapping,
            # Whisper ships no bias for k_proj. The fused QKV bias is
            # allocated anyway and the k slice is zero-filled at load time
            # (see load_weights) -- a zero bias is exactly "no bias".
            qkv_bias=True,
            proj_bias=True,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]

        vision = mapping.vision
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.encoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        residual = hidden_states
        normed = self.self_attn_layer_norm(hidden_states)
        # The rows as the dense ``[batch, max_seqlen, d]`` batch they are
        # (every window has the encoder's fixed length), so a backend that
        # needs equal-length sequences (``sdpa``) can take them; the varlen
        # backends read the same rows through ``cu_seqlens``.
        batch = cu_seqlens.shape[0] - 1
        attended = self.self_attn(
            normed.view(batch, max_seqlen, -1),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        ).view(hidden_states.shape)
        hidden_states = residual + attended

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return residual + hidden_states


class WhisperEncoder(nn.Module):
    """Log-mel frames -> a fixed-length sequence of encoder states.

    Runs once per request. The output is what every decode step cross-attends
    to; nothing here touches the paged KV pool.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        # Whisper's encoder positions are fixed sinusoids shipped as weights,
        # not a learned table; registered as a buffer so they load like one
        # and never appear in the optimizer's eyes.
        self.register_buffer(
            "embed_positions",
            torch.zeros(config.max_source_positions, embed_dim),
            persistent=True,
        )
        self.layers = nn.ModuleList(
            [
                WhisperEncoderLayer(
                    config,
                    mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for i in range(config.encoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.conv1.weight.device

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """``input_features``: [batch, num_mel_bins, frames] -> [batch*T, d]."""
        features = input_features.to(device=self.device, dtype=self.dtype)
        hidden = F.gelu(self.conv1(features))
        hidden = F.gelu(self.conv2(hidden))
        hidden = hidden.permute(0, 2, 1)

        batch, seq_len, _ = hidden.shape
        if seq_len > self.embed_positions.shape[0]:
            raise ValueError(
                f"Whisper encoder got {seq_len} frames but the position table "
                f"holds {self.embed_positions.shape[0]}; the feature extractor "
                "must pad or trim to the model's window."
            )
        hidden = hidden + self.embed_positions[:seq_len].to(hidden.dtype)

        # One packed varlen sequence per batch row, all the same length: the
        # encoder window is fixed, so cu_seqlens is a plain arange.
        # ``contiguous``: the permute above leaves the rows feature-strided and
        # the add keeps that layout; at batch 1 ``reshape`` is then a view, and
        # every residual add and LayerNorm of the 32 layers would run on
        # transposed rows (non-vectorized adds plus a copy per norm, ~2 ms per
        # request on large-v3). At batch > 1 ``reshape`` already copies.
        flat = hidden.reshape(batch * seq_len, -1).contiguous()
        cu_seqlens = torch.arange(
            0,
            (batch + 1) * seq_len,
            seq_len,
            device=flat.device,
            dtype=torch.int32,
        )
        for layer in self.layers:
            flat = layer(flat, cu_seqlens=cu_seqlens, max_seqlen=seq_len)
        return self.layer_norm(flat)


class CrossAttentionCache:
    """Per-request encoder K/V for the decoder's cross-attention.

    Deliberately not a paged-arena cache group. Cross KV is written once,
    never grows, is never prefix-matched and is never evicted apart from its
    request, so none of the arena's machinery applies to it (see the module
    docstring, and §3.3 of the xref).

    Layout is ``[layer, slot, padded_len, kv_heads, head_dim]`` with
    ``padded_len = pages_per_slot * page_size`` the smallest whole number of
    pages that holds ``encoder_len`` rows. Layer first, so ``k[layer]`` is
    contiguous and :meth:`paged_layer` views it as the
    ``[slot * pages_per_slot, page_size, kv_heads, head_dim]`` page pool the
    paged MHA ops take, without a copy. Slot ``s`` owns pages
    ``s * pages_per_slot .. (s + 1) * pages_per_slot - 1``. Rows
    ``encoder_len..padded_len`` are never written; the read masks them with
    ``cache_seqlens = encoder_len``.

    Slots are indexed by ``req_pool_index``, which the scheduler bounds at
    ``per_rank_max_batch + 1`` (``event_loop.py:369``); ModelConfig stamps
    that bound on the HF config as ``cross_attn_slots``.

    The read-side tensors are allocated once and refreshed in place, so a
    CUDA graph captured over a forward replays against the same addresses
    (``docs/design/unified_path.md``):

      * ``page_table`` -- ``[num_slots, pages_per_slot]`` int32; a forward
        with ``bs`` requests fills rows ``[:bs]`` from its
        ``req_pool_indices`` in one device op (:meth:`fill_page_table`);
      * ``cache_seqlens`` -- ``[num_slots]`` int32, constant ``encoder_len``;
      * ``cu_seqlens_kv`` -- ``[num_slots + 1]`` int32, constant
        ``arange * encoder_len`` (the extend op's KV boundaries);
      * ``cu_seqlens_q`` -- ``[num_slots + 1]`` int32; an extend forward
        with ``bs`` requests fills ``[1:bs + 1]`` from its ``gather_ids``
        (row ``0`` stays ``0``).

    :meth:`refresh_read` rewrites the page table and the query boundaries
    once per forward, and :meth:`read_plan` hands each layer views of them
    sized by the live forward context. Both read only the live context and
    the persistent buffers, never a capture-time shape, so the same code
    serves the eager forward, a decode graph and a breakable prefill graph
    whose breaks run them on every replay.

    **Slot-state image** (``SlotStateExporter``, ``execution/slot_state.py``).
    A retraction suspends a request and restores it, possibly into another
    slot, without recomputing anything (``docs/design/scheduler.md`` §2, §4),
    so the slot's cross KV must travel with its pages. The image is
    ``k[:, slot, :encoder_len]`` then ``v[:, slot, :encoder_len]`` -- the
    padding rows are never written and stay out -- behind a prepared marker.

    The cache is a request-keyed owner: the host-side ``owner`` of a slot is
    the request whose encoder output its rows hold, recorded when the audio is
    encoded (:meth:`claim_slot`). An extend for request ``r`` in slot ``s``
    runs the encoder iff ``owner_of(s) != r`` (``_audio_features_to_encode``),
    so a request that resumes into a slot another request encoded since --
    whatever its prefix length -- encodes its own audio again.

    Leaving a slot is detected the way the request-keyed sampling backends
    detect it (``SamplingBackend.prepare_step``): the data plane receives no
    finish or abort event, so a finished request's claim stays until the next
    occupant's extend sees a different request id and encodes over it. That
    relies on request ids being unique over the server's lifetime, as the
    sampling backends' per-slot state does. A victim whose slot it does not
    own (retracted before its encoder ran, or a later occupant's rows) images
    the marker as not-prepared and no rows, and the restore leaves the new
    slot unowned (:meth:`release_slot`) so the request's next extend encodes
    its own audio; a prepared image is restored and the new slot is claimed
    for the request.

    The image is large: ``2 x decoder_layers x encoder_len x d_model`` bytes
    of bf16 per slot (plus the 16-byte marker) -- 245.8 MB on large-v3
    (32 x 1500 x 1280 x 2 B x 2). The snapshot executor pins
    ``(max_retracted_requests + 1)`` such rows (``cache/l2/sizing.py``), so
    a Whisper deployment's slot-state arena is ~4.2 GB at 16 retractable
    requests; ``--retraction-snapshot-max-requests`` bounds it.

    The read-side tensors are not imaged: ``page_table`` and ``cu_seqlens_q``
    are rewritten at the start of every forward (token-derived in the
    exporter's sense -- the restored request's next forward reseeds it), and
    ``cache_seqlens``, ``cu_seqlens_kv`` and ``page_offsets`` hold the same
    values whatever slot reads them.
    """

    #: Rewritten from the batch's ``req_pool_indices`` and ``gather_ids`` by
    #: every forward (:meth:`refresh_read`), so a restore has nothing to copy.
    token_derived_slot_state: tuple[str, ...] = ("page_table", "cu_seqlens_q")
    #: Identical for every slot (``encoder_len`` lengths and boundaries, the
    #: per-slot page offsets).
    constant_slot_state: tuple[str, ...] = (
        "cache_seqlens",
        "cu_seqlens_kv",
        "page_offsets",
    )

    def __init__(
        self,
        num_slots: int,
        num_layers: int,
        encoder_len: int,
        kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"cross-attention page_size must be > 0, got {page_size}")
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.encoder_len = encoder_len
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.pages_per_slot = -(-encoder_len // page_size)
        self.padded_len = self.pages_per_slot * page_size
        shape = (num_layers, num_slots, self.padded_len, kv_heads, head_dim)
        # Allocated once, up front. Growing it lazily would mean reading
        # ``slots.max()`` on the read path, and that is a device-to-host sync
        # on every decode step -- illegal under CUDA graph capture and a
        # per-step stall otherwise.
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

        self.page_table = torch.zeros(
            (num_slots, self.pages_per_slot), dtype=torch.int32, device=device
        )
        self.page_offsets = torch.arange(
            self.pages_per_slot, dtype=torch.int32, device=device
        )
        self.cache_seqlens = torch.full(
            (num_slots,), encoder_len, dtype=torch.int32, device=device
        )
        self.cu_seqlens_kv = torch.arange(
            0,
            (num_slots + 1) * encoder_len,
            encoder_len,
            dtype=torch.int32,
            device=device,
        )
        self.cu_seqlens_q = torch.zeros(
            (num_slots + 1,), dtype=torch.int32, device=device
        )
        # Host-side: which request's encoder output each slot's rows hold;
        # None while the slot holds nobody's (fresh, released, or restored
        # from a not-prepared image).
        self.owner_per_slot: list[str | None] = [None] * num_slots

    def bytes_total(self) -> int:
        return self.k.numel() * self.k.element_size() * 2

    # ------------------------------------------------------------------
    # Slot ownership (host-side, request-keyed)
    # ------------------------------------------------------------------

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self.num_slots:
            raise IndexError(
                f"cross-attention slot {slot} outside the pool of "
                f"{self.num_slots}; the buffer is sized from max_num_seqs"
            )

    def owner_of(self, slot: int) -> str | None:
        """The request whose encoder output ``slot`` holds, or None."""
        self._check_slot(slot)
        return self.owner_per_slot[slot]

    def claim_slot(self, slot: int, request_id: str) -> None:
        """Record that ``slot`` now holds ``request_id``'s cross KV."""
        self._check_slot(slot)
        self.owner_per_slot[slot] = request_id

    def release_slot(self, slot: int) -> None:
        """Forget ``slot``'s owner: its rows are nobody's from now on."""
        self._check_slot(slot)
        self.owner_per_slot[slot] = None

    # ------------------------------------------------------------------
    # Slot-state image (SlotStateExporter)
    # ------------------------------------------------------------------

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        """The slot's cross KV rows a snapshot images: K, then V.

        Views of the written ``encoder_len`` rows of every layer
        (``[layers, encoder_len, heads, head_dim]``, strided over the slot
        dimension); the pack helpers copy non-contiguous views.
        """
        self._check_slot(slot)
        return [
            self.k[:, slot, : self.encoder_len],
            self.v[:, slot, : self.encoder_len],
        ]

    def slot_state_bytes(self) -> int:
        return PREPARED_MARKER_BYTES + slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        # The rows are ``request_id``'s only once its audio was encoded into
        # this slot; otherwise they are a previous occupant's (or zeros) and
        # the image carries the marker alone.
        prepared = self.owner_of(slot) == request_id
        write_prepared_marker(out, prepared)
        if prepared:
            pack_slot_rows(
                self.slot_state_rows(slot), out[PREPARED_MARKER_BYTES:], stream
            )

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        if read_prepared_marker(src):
            unpack_slot_rows(
                self.slot_state_rows(slot), src[PREPARED_MARKER_BYTES:], stream
            )
            # The restored rows are the request's own: its next extend must
            # not re-encode over them.
            self.claim_slot(slot, request_id)
        else:
            # Nobody's: the request's next extend encodes its own audio, as
            # the unretracted request's first extend would have.
            self.release_slot(slot)

    def check_write_slots(self, slots: list[int]) -> None:
        """Validate the slots one step writes, on the host.

        In range, and no slot twice: ``index_copy_`` with a repeated index is
        nondeterministic. The device index itself is staged by
        :class:`AudioEncodeRunner`.
        """
        for slot in slots:
            self._check_slot(slot)
        if len(set(slots)) != len(slots):
            raise ValueError(
                f"cross-attention write names a slot twice in one step: {slots}"
            )

    def write_layer(
        self,
        layer_id: int,
        slot_index: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Store one decoder layer's encoder K/V for every request of a step.

        ``k`` / ``v`` are ``[num_requests, encoder_len, kv_heads, head_dim]``;
        row ``i`` lands in slot ``slot_index[i]``. One ``index_copy_`` per
        tensor, whatever the request count. Slots not in ``slot_index`` are
        untouched.
        """
        if k.shape[1] != self.encoder_len or v.shape[1] != self.encoder_len:
            raise ValueError(
                f"cross-attention KV has {k.shape[1]} rows per request but "
                f"the cache was built for {self.encoder_len}; Whisper's encoder "
                "window is fixed, so a different length means the feature "
                "extractor did not pad or trim to it."
            )
        rows = slice(0, self.encoder_len)
        self.k[layer_id, :, rows].index_copy_(0, slot_index, k)
        self.v[layer_id, :, rows].index_copy_(0, slot_index, v)

    def paged_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's K/V as the page pool the paged MHA ops read (a view)."""
        pages = (
            self.num_slots * self.pages_per_slot,
            self.page_size,
            self.kv_heads,
            self.head_dim,
        )
        return self.k[layer_id].view(pages), self.v[layer_id].view(pages)

    def fill_page_table(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        """Point rows ``[:bs]`` of the page table at the batch's slots.

        Row ``i`` becomes ``slot_i * pages_per_slot + arange(pages_per_slot)``,
        computed on device in one op, so there is no host read of the slots
        and a captured graph recomputes it from the replayed
        ``req_pool_indices``. Returns the ``[:bs]`` view, whose address is the
        same for every ``bs``.
        """
        bs = req_pool_indices.shape[0]
        if bs > self.num_slots:
            raise ValueError(
                f"cross-attention read for {bs} requests but the pool has "
                f"{self.num_slots} slots"
            )
        table = self.page_table[:bs]
        torch.add(
            self.page_offsets,
            req_pool_indices.unsqueeze(1),
            alpha=self.pages_per_slot,
            out=table,
        )
        return table

    def refresh_read(self, ctx: ForwardContext, req_pool_indices: torch.Tensor) -> None:
        """Rewrite this forward's read tables in place, once for every layer.

        ``req_pool_indices`` holds at least ``ctx.bs`` slots: the executor
        passes exactly the batch's, a breakable prefill graph passes its
        whole persistent buffer (the capture-time slice would freeze the
        dummy batch's request count), so the live count comes from ``ctx``.
        On a forward carrying an extend, the query boundaries are
        ``[0, gather_ids + 1]`` (``gather_ids`` is each request's last row on
        an extend or a non-speculative mixed forward). Device ops only.
        """
        bs = ctx.bs
        self.fill_page_table(req_pool_indices[:bs])
        if ctx.num_extends == 0 or bs == 0:
            return
        if ctx.gather_ids is None or ctx.gather_ids.shape[0] != bs:
            raise ValueError(
                "Whisper cross-attention on an extend needs gather_ids with one "
                f"last-row index per request ({bs}), got "
                f"{None if ctx.gather_ids is None else tuple(ctx.gather_ids.shape)}"
            )
        # ``copy_`` casts the int64 last-row indices to the int32 boundaries.
        self.cu_seqlens_q[1 : bs + 1].copy_(ctx.gather_ids + 1)

    def read_plan(self, ctx: ForwardContext, num_rows: int) -> "CrossAttentionPlan":
        """One layer's read of the tables :meth:`refresh_read` wrote.

        Views only -- no device work. ``num_rows`` is the query row count the
        layer feeds (under a padded prefill-graph replay, the bucket): a
        decode reads ``num_rows // bs`` rows per request, an extend uses it
        as the grid bound of the extend op, which only sizes its launch with
        it (a runtime argument, not a compile-time one).
        """
        bs = ctx.bs
        page_table = self.page_table[:bs]
        cache_seqlens = self.cache_seqlens[:bs]
        cu_seqlens_kv = self.cu_seqlens_kv[: bs + 1]
        if bs == 0:
            return CrossAttentionPlan(
                cache=self,
                num_requests=0,
                uniform_q_len=True,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q=None,
                max_seqlen_q=0,
            )
        if ctx.num_extends == 0:
            if num_rows % bs != 0:
                raise ValueError(
                    f"decode forward has {num_rows} rows for {bs} requests; "
                    "cross-attention expects the same row count per request"
                )
            return CrossAttentionPlan(
                cache=self,
                num_requests=bs,
                uniform_q_len=True,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q=None,
                max_seqlen_q=num_rows // bs,
            )
        return CrossAttentionPlan(
            cache=self,
            num_requests=bs,
            uniform_q_len=False,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q=self.cu_seqlens_q[: bs + 1],
            # A python int, never ``.max().item()``: reading a device tensor
            # here is a host sync. The forward's row count bounds every
            # request's query rows.
            max_seqlen_q=num_rows,
        )


class WhisperDecoderSelfAttention(nn.Module):
    """Causal self-attention over the decoded tokens; ordinary paged KV."""

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            embed_dim,
            self.head_dim,
            total_heads,
            total_heads,
            # Whisper's k_proj carries no bias; the fused bias exists and its
            # k slice is zero-filled at load time.
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # The reference implementations pre-multiply q by the scale and build
        # the attention with scaling=1.0. Passing the scale instead is the
        # same arithmetic with one fewer elementwise pass.
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            rotary_emb=None,
            qk_norm=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Whisper has no RoPE on decoder self-attn; positions still required by
        # the nctu6 PagedAttention prologue signature.
        attended = self.attn(q, k, v, positions, ctx)
        output, _ = self.out_proj(attended)
        return output


class WhisperCrossAttention(nn.Module):
    """Decoder attends to the fixed encoder output.

    K/V are projected once, at the request's first prefill
    (:meth:`project_encoder_kv`), and stored in :class:`CrossAttentionCache`.
    Every forward after that -- including the rest of that prefill -- reads
    them back from the cache through the paged MHA ops (:meth:`attend`).
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        cross_cache: CrossAttentionCache,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        total_heads = config.decoder_attention_heads
        self.layer_id = layer_id
        # Shared by every layer; a plain object, not a submodule.
        self.cross_cache = cross_cache
        self.tp_size = mapping.attn.tp_size
        self.num_heads = total_heads // self.tp_size
        self.head_dim = embed_dim // total_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # The checkpoint's k_proj and v_proj, fused so the cross-KV write is
        # one GEMM per layer. Shards are [k, v]; under TP each rank holds its
        # heads of both, so the local output splits into k then v. Whisper's
        # k_proj has no bias: the fused bias exists and its k slice is
        # zero-filled at load time, as for the self-attention QKV.
        self.kv_size = self.num_heads * self.head_dim
        self.kv_proj = MergedColumnParallelLinear(
            embed_dim,
            [embed_dim, embed_dim],
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("kv_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
            reduce_results=False,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )

    def project_encoder_kv(
        self, encoder_states: torch.Tensor, num_requests: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a step's encoder output into this layer's K/V in one GEMM.

        ``encoder_states`` is ``[num_requests * encoder_len, d_model]``;
        returns K and V as ``[num_requests, encoder_len, heads, head_dim]``
        views of the one fused output (no copy).
        """
        kv, _ = self.kv_proj(encoder_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        shape = (num_requests, -1, self.num_heads, self.head_dim)
        return k.view(shape), v.view(shape)

    def project_query(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        return q.view(-1, self.num_heads, self.head_dim)

    @break_point
    def attend(
        self,
        q: torch.Tensor,
        ctx: ForwardContext,
        req_pool_indices: torch.Tensor,
        refresh_read: bool,
    ) -> torch.Tensor:
        """Attend ``q`` (``[rows, heads, head_dim]``) to each request's own cross KV.

        A break point of the breakable prefill graph, like the decoder's
        self-attention: the read depends on the live batch (its slots, its
        request count, its query boundaries), so under a prefill-graph
        replay it runs eager and reads them from the live ``ctx`` and the
        persistent ``req_pool_indices`` buffer (``[>= bs]``). The first layer
        passes ``refresh_read=True`` and rewrites the per-forward read tables
        for every layer (:meth:`CrossAttentionCache.refresh_read`); the flag
        is a per-layer constant, so a captured break may freeze it. Under a
        padded replay the rows past the live ones are filler: the extend op
        computes only the rows ``cu_seqlens_q`` covers, and the break clears
        the rest of its output.
        """
        cache = self.cross_cache
        if refresh_read:
            cache.refresh_read(ctx, req_pool_indices)
        cross = cache.read_plan(ctx, int(q.shape[0]))
        if cross.num_requests == 0:
            # An idle forward (e.g. a DP rank with nothing to run) has no rows.
            return q
        k_cache, v_cache = cache.paged_layer(self.layer_id)
        if cross.uniform_q_len:
            return mha_decode_with_kvcache(
                q,
                k_cache,
                v_cache,
                page_table=cross.page_table,
                cache_seqlens=cross.cache_seqlens,
                max_seqlen_k=cache.padded_len,
                max_seqlen_q=cross.max_seqlen_q,
                softmax_scale=self.scaling,
            )
        return mha_extend_with_kvcache(
            q,
            cu_seqlens_q=cross.cu_seqlens_q,
            cu_seqlens_kv=cross.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=cross.page_table,
            cache_seqlens=cross.cache_seqlens,
            max_seqlen_q=cross.max_seqlen_q,
            max_seqlen_k=cache.padded_len,
            # Every query row sees the request's whole encoder output.
            is_causal=False,
            softmax_scale=self.scaling,
        )

    def project_out(self, attended: torch.Tensor) -> torch.Tensor:
        output, _ = self.out_proj(attended.reshape(-1, self.num_heads * self.head_dim))
        return output


@dataclasses.dataclass
class CrossAttentionPlan:
    """Everything one decoder layer needs to run cross-attention this step.

    Views of the cache's persistent read tables, which the first layer
    rewrites once per forward (:meth:`CrossAttentionCache.refresh_read`) and
    every layer reads: rebuilding them per layer would be pure launch
    overhead. Building a plan is host work only (:meth:`~CrossAttentionCache.read_plan`).

    ``uniform_q_len`` selects the op from the batch's shape, not from a mode:
    when every request feeds exactly ``max_seqlen_q`` rows (a decode step)
    the read is ``mha_decode_with_kvcache``; otherwise (any step that
    carries an extend, mixed or not) it is ``mha_extend_with_kvcache`` with
    per-request ``cu_seqlens_q``, which also covers the decode rows of a
    mixed step as one-row segments. Both read the same pages.
    """

    cache: "CrossAttentionCache"
    num_requests: int
    uniform_q_len: bool
    # ``[bs, pages_per_slot]`` view of the cache's persistent page table.
    page_table: torch.Tensor
    # ``[bs]`` view of the cache's constant ``encoder_len`` lengths.
    cache_seqlens: torch.Tensor
    # ``[bs + 1]`` constant KV boundaries, for the extend op.
    cu_seqlens_kv: torch.Tensor
    # ``[bs + 1]`` query boundaries; None when ``uniform_q_len``.
    cu_seqlens_q: torch.Tensor | None
    # Rows per request when ``uniform_q_len``; an upper bound otherwise.
    max_seqlen_q: int


class WhisperDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        layer_id: int,
        cross_cache: CrossAttentionCache,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.layer_id = layer_id
        self.self_attn = WhisperDecoderSelfAttention(
            config,
            mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.encoder_attn = WhisperCrossAttention(
            config,
            mapping,
            layer_id=layer_id,
            cross_cache=cross_cache,
            quant_config=quant_config,
            prefix=add_prefix("encoder_attn", prefix),
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = ColumnParallelLinear(
            embed_dim,
            config.decoder_ffn_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.fc2 = RowParallelLinear(
            config.decoder_ffn_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
            reduce_results=True,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
        req_pool_indices: torch.Tensor,
        refresh_cross_read: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-norm block with each residual add fused into the next LayerNorm.

        ``hidden_states`` is the previous sublayer's output not yet added to
        the ``residual`` stream; returns this block's FFN output and the stream
        the same way, so the next block (or the final norm) adds them.
        """
        normed, residual = _add_layer_norm(
            hidden_states, residual, self.self_attn_layer_norm
        )
        hidden_states = self.self_attn(positions, normed, ctx)

        normed, residual = _add_layer_norm(
            hidden_states, residual, self.encoder_attn_layer_norm
        )
        query = self.encoder_attn.project_query(normed)
        attended = self.encoder_attn.attend(
            query, ctx, req_pool_indices, refresh_cross_read
        )
        hidden_states = self.encoder_attn.project_out(attended)

        normed, residual = _add_layer_norm(
            hidden_states, residual, self.final_layer_norm
        )
        hidden_states, _ = self.fc1(normed)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states, residual


class WhisperDecoder(nn.Module):
    """The decoder stack: the inner model a prefill graph captures.

    Follows the inner-stack contract of ``execution/prefill_graph.py``:
    ``forward(input_ids, positions, ctx, **model_kwargs)`` returns
    ``(hidden_states, aux_hidden_states)`` and starts from ``input_embeds``
    when given (the graph's static embeds buffer, filled outside the graph
    by ``embed_tokens``). The learned position embedding is added inside,
    from ``positions``, so a captured graph reads it from the static
    positions buffer.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        cross_cache: CrossAttentionCache,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            embed_dim,
            prefix=add_prefix("embed_tokens", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        # Learned, unlike the encoder's fixed sinusoids.
        self.embed_positions = nn.Embedding(config.max_target_positions, embed_dim)
        self.layers = nn.ModuleList(
            [
                WhisperDecoderLayer(
                    config,
                    mapping,
                    layer_id=i,
                    cross_cache=cross_cache,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.decoder_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        *,
        req_pool_indices: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        """``req_pool_indices`` holds at least ``ctx.bs`` request-pool slots;
        ``input_embeds`` omitted means look the tokens up here."""
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        # The first block's norm adds the token and position embeddings.
        hidden_states, residual = input_embeds, self.embed_positions(positions)
        for i, layer in enumerate(self.layers):
            # The first layer's read rewrites the per-forward read tables
            # every layer's read uses.
            hidden_states, residual = layer(
                positions, hidden_states, residual, ctx, req_pool_indices, i == 0
            )
        normed, _ = _add_layer_norm(hidden_states, residual, self.layer_norm)
        return normed, None


# Requests one encoder pass takes (AudioEncodeRunner). A step with more fresh
# requests runs several passes. 16 is the script default max_num_seqs; a pass
# of 16 requests is ~150 ms of GPU time on large-v3, so a larger step is
# GPU-bound and splitting it costs nothing, and the bound caps the encoder
# graphs' memory pool (~1.2 GiB for 1..16 on large-v3).
_ENCODE_PASS_MAX_REQUESTS = 16


class AudioEncodeRunner:
    """The encoder and cross-KV write of one prefill step, eager or graphed.

    A step's fresh requests are encoded in passes of at most ``max_requests``.
    Each pass stages its inputs into persistent device buffers -- the log-mel
    windows into ``features[:n]`` and the target slots into
    ``slot_index[:n]``, both by non-blocking copies from pinned host memory --
    and then runs over those buffers alone: the encoder over the ``n``
    windows, then per decoder layer one ``kv_proj`` GEMM and one
    ``index_copy_`` each for K and V into the slots
    (:meth:`WhisperForConditionalGeneration.encode_into_cross_cache`).

    The pass reads only the buffers and has one fixed shape per ``n``
    (Whisper's window is fixed: ``2 x encoder_len`` frames in,
    ``encoder_len`` rows per request out), so a CUDA graph captured per
    request count replays it exactly, with no padding. ``cuda_graph`` is the
    only difference between the eager runner and the graphed one: staging is
    the same code, and the graphed runner replays the captured pass where
    the eager one runs it. The model starts with the eager runner;
    :meth:`WhisperForConditionalGeneration.make_encoder_cudagraph_wrappers`
    builds the graphed one, which the multimodal encoder-graph install
    (``MultimodalRuntime.install_encoder_graphs``) puts in its place.

    Why a graph: the eager encoder is host-bound. Its ~430 launches take as
    long to enqueue (~12-14 ms) as the GPU takes to run them for one request,
    so the decoder's prefill breaks queue behind it. A replay is one launch,
    and the host enqueues the decoder while the encoder runs.

    Staging needs no host sync. A copy from pageable host memory makes the
    host wait for the stream, so each window is copied from pinned memory
    (features that arrive through shared memory are pinned already; others
    are pinned once here). Pinned tensors come from PyTorch's caching host
    allocator, which records the copy on the stream and does not hand the
    block out again before the copy has run. The next pass's staging is
    queued behind this pass on the same stream, so it cannot overwrite the
    buffers early either.
    """

    def __init__(
        self,
        model: "WhisperForConditionalGeneration",
        *,
        max_requests: int,
        cuda_graph: bool,
    ) -> None:
        cache = model.cross_cache
        if not 0 < max_requests <= cache.num_slots:
            raise ValueError(
                f"encoder pass of {max_requests} requests for a pool of "
                f"{cache.num_slots} slots"
            )
        config = model.config
        self.model = model
        self.max_requests = max_requests
        self.cuda_graph = cuda_graph
        device = cache.k.device
        # The feature extractor's dtype; a window in another dtype is cast on
        # its way in (exactly: the encoder casts to its own dtype on device).
        self.features = torch.zeros(
            (max_requests, config.num_mel_bins, 2 * config.max_source_positions),
            dtype=torch.float32,
            device=device,
        )
        self.slot_index = torch.zeros((max_requests,), dtype=torch.int64, device=device)
        # Request count -> the captured pass (filled by :meth:`capture`).
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def __call__(self, features: Sequence[torch.Tensor], slots: list[int]) -> None:
        """Encode ``features[i]`` (``[mel, frames]`` or ``[1, mel, frames]``,
        on any device) into slot ``slots[i]``'s cross KV."""
        if len(features) != len(slots):
            raise ValueError(
                f"encoder pass got {len(features)} audio window(s) for "
                f"{len(slots)} slot(s)"
            )
        self.model.cross_cache.check_write_slots(slots)
        for start in range(0, len(slots), self.max_requests):
            stop = min(start + self.max_requests, len(slots))
            self._stage(features[start:stop], slots[start:stop])
            self._run_pass(stop - start)

    def _stage(self, features: Sequence[torch.Tensor], slots: list[int]) -> None:
        window = tuple(self.features.shape[1:])
        for row, feature in enumerate(features):
            if feature.ndim == 3 and feature.shape[0] == 1:
                feature = feature[0]
            if tuple(feature.shape) != window:
                raise ValueError(
                    f"Whisper audio window has shape {tuple(feature.shape)}; the "
                    f"encoder takes {window} (the feature extractor pads or "
                    "trims every clip to the model's 30 s window)"
                )
            if feature.dtype != self.features.dtype:
                feature = feature.to(self.features.dtype)
            if feature.device.type == "cpu" and not feature.is_pinned():
                feature = feature.pin_memory()
            self.features[row].copy_(feature, non_blocking=True)
        host = torch.tensor(slots, dtype=torch.int64, pin_memory=True)
        self.slot_index[: len(slots)].copy_(host, non_blocking=True)

    def _run_pass(self, n: int) -> None:
        if not self.cuda_graph:
            self.model.encode_into_cross_cache(self.features[:n], self.slot_index[:n])
            return
        graph = self.graphs.get(n)
        if graph is None:
            raise RuntimeError(
                f"no encoder graph captured for {n} request(s); capture() "
                f"covers 1..{self.max_requests}"
            )
        graph.replay()

    @torch.no_grad()
    def capture(self) -> None:
        """Capture the pass for every request count ``1..max_requests``.

        Largest first, into one memory pool: the passes never overlap, and
        each leaves nothing live in the pool. Each count runs once eagerly
        first, so JIT compilation and library autotuning happen outside the
        capture. The passes run over zero windows into slots ``0..n-1``:
        that overwrites those slots' rows but claims none of them, so the
        next request in any of them encodes its own audio over it.
        """
        if not self.cuda_graph:
            raise RuntimeError("capture() on an eager encoder runner")
        from tokenspeed_kernel.workspace import capture_session

        device = self.features.device
        pool = None
        self.features.zero_()
        for n in range(self.max_requests, 0, -1):
            self.slot_index[:n].copy_(torch.arange(n, device=device))
            features, slot_index = self.features[:n], self.slot_index[:n]
            self.model.encode_into_cross_cache(features, slot_index)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with capture_session(graph), torch.cuda.graph(graph, pool=pool):
                self.model.encode_into_cross_cache(features, slot_index)
            if pool is None:
                pool = graph.pool()
            self.graphs[n] = graph
        torch.cuda.synchronize(device)


class WhisperForConditionalGeneration(nn.Module):
    """Whisper as this engine serves it: one encoder pass, then ordinary decode.

    The request carries its audio the same way a multimodal request carries an
    image. What is different from ``qwen3_asr.py`` is where the encoder output
    goes: not into the decoder's input embeddings, but into every decoder
    layer's cross-attention K/V, written once at prefill and read by every
    step after.
    """

    def __init__(
        self,
        config: Any,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        if not is_multimodal_active:
            # --language-model-only strips the tower from a prefix-style
            # multimodal model and still leaves a usable LM. Whisper's decoder
            # has nothing to attend to without its encoder, so there is no
            # meaningful text-only mode to fall back to.
            raise ValueError(
                "--language-model-only is not meaningful for Whisper: the "
                "decoder cross-attends to the encoder on every step, so "
                "dropping the encoder leaves no model to run."
            )
        self.config = config
        self.mapping = mapping
        # Bounded by the request pool, which the scheduler sizes at
        # per_rank_max_batch + 1 (event_loop.py:369). ModelConfig stamps the
        # bound on the HF config because the model is constructed from that
        # config alone; a config without the stamp was not built by
        # ModelConfig, and guessing a pool size would undersize the buffer.
        if "cross_attn_slots" not in config.to_dict():
            raise ValueError(
                "Whisper config lacks cross_attn_slots; ModelConfig stamps it "
                "from --max-num-seqs (the request pool size)"
            )
        cross_dtype = torch.bfloat16
        head_dim = config.d_model // config.decoder_attention_heads
        # Built before the decoder, whose cross-attention layers read it.
        self.cross_cache = CrossAttentionCache(
            num_slots=int(config.cross_attn_slots),
            num_layers=config.decoder_layers,
            encoder_len=config.max_source_positions,
            kv_heads=config.decoder_attention_heads // mapping.attn.tp_size,
            head_dim=head_dim,
            page_size=_cross_kv_page_size(cross_dtype, head_dim),
            dtype=cross_dtype,
            device=torch.device("cuda"),
        )
        self.encoder = WhisperEncoder(
            config,
            mapping,
            quant_config=None,  # the tower stays in bf16/fp16
            prefix=add_prefix("model.encoder", prefix),
            mm_attention_backend=mm_attention_backend,
        )
        self.decoder = WhisperDecoder(
            config,
            mapping,
            cross_cache=self.cross_cache,
            quant_config=quant_config,
            prefix=add_prefix("model.decoder", prefix),
        )
        self.proj_out = ParallelLMHead(
            config.vocab_size,
            config.d_model,
            bias=False,
            prefix=add_prefix("proj_out", prefix),
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
        )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=mapping.attn.has_dp,
            tp_rank=mapping.lm_head.tp_rank,
            tp_size=mapping.lm_head.tp_size,
            tp_group=mapping.lm_head.tp_group,
            dp_lm_head_tp=mapping.attn.has_dp and mapping.lm_head.has_tp,
        )
        # The prefill seam's encoder + cross-KV write. Eager until the
        # multimodal encoder-graph install replaces it with the graphed runner
        # (make_encoder_cudagraph_wrappers); both stage and pass the same way.
        self.audio_encode_runner = AudioEncodeRunner(
            self,
            max_requests=min(self.cross_cache.num_slots, _ENCODE_PASS_MAX_REQUESTS),
            cuda_graph=False,
        )

    # ------------------------------------------------------------------
    # The causal-LM contract the prefill graph drives
    # ------------------------------------------------------------------
    #
    # ``execution/prefill_graph.py`` captures a model's inner stack
    # (``.model``), lands ``.model.embed_tokens`` into its static embeds
    # buffer outside the graph, asks ``prepare_model_kwargs`` which forward
    # kwargs the stack takes, and finishes with ``.logits_processor`` over
    # ``.lm_head``. Whisper keeps its checkpoint-shaped names (``decoder``,
    # ``proj_out``: the load errors name them) and exposes the contract's
    # names as read-only aliases, so the parameters are registered once.

    @property
    def model(self) -> WhisperDecoder:
        """The inner stack: the decoder (the encoder runs in the seam)."""
        return self.decoder

    @property
    def lm_head(self) -> ParallelLMHead:
        return self.proj_out

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        """The forward kwargs the decoder stack takes.

        ``req_pool_indices`` addresses each request's cross-attention slot;
        every forward call site passes it (the executor its ``[:bs]`` view, a
        prefill graph its whole persistent buffer), so a missing one is a
        call-site bug, not an omission with a meaning.
        """
        req_pool_indices = kwargs.get("req_pool_indices")
        if req_pool_indices is None:
            raise RuntimeError(
                "Whisper needs req_pool_indices to find each request's "
                "cross-attention KV slot"
            )
        return {
            "req_pool_indices": req_pool_indices,
            "input_embeds": kwargs.get("input_embeds"),
        }

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context: MultimodalForwardContext | None,
    ) -> None:
        """The prefill graph's multimodal seam: encode, and merge nothing.

        Runs before the decoder, outside the prefill graph (on the eager
        forward and a prefill-graph replay alike): the encoder and the
        cross-KV write for the step's requests whose slot does not hold
        their own encoder output, through :attr:`audio_encode_runner` (eager,
        or its own per-request-count graphs). Whisper's audio goes to
        cross-attention, not into the decoder's input embeddings, so there is
        nothing to merge and the decoder starts from its ordinary token embeddings
        (``None`` tells the graph to look them up as for text).
        """
        if (
            multimodal_context is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        features, slots, request_ids = _audio_features_to_encode(
            multimodal_context, self.cross_cache
        )
        if features is not None:
            self.encode_audio(features, slots, request_ids)
        return None

    def encode_audio(
        self,
        input_features: Sequence[torch.Tensor],
        slots: list[int],
        request_ids: list[str],
    ) -> None:
        """Encode each request's audio window into its slot's cross-attention
        K/V, and record each slot as holding its request's encoder output.

        ``input_features[i]`` is request ``i``'s ``[mel, frames]`` window (a
        stacked ``[N, mel, frames]`` tensor works too). The work runs through
        :attr:`audio_encode_runner` -- eager, or the per-request-count CUDA
        graphs -- in passes batched over the requests (see
        :meth:`encode_into_cross_cache`); nothing reads the device.
        """
        if len(slots) != len(request_ids):
            raise ValueError(
                f"encode_audio got {len(slots)} slot(s) for "
                f"{len(request_ids)} request id(s)"
            )
        self.audio_encode_runner(input_features, slots)
        # Host-side bookkeeping only. The writes above are queued on the
        # stream ahead of every later read, so a claim is never ahead of the
        # rows it vouches for from any reader's point of view.
        for slot, request_id in zip(slots, request_ids):
            self.cross_cache.claim_slot(slot, request_id)

    def encode_into_cross_cache(
        self, features: torch.Tensor, slot_index: torch.Tensor
    ) -> None:
        """One encoder pass: ``features`` (``[n, mel, frames]``, on device)
        into the cross KV of the slots in ``slot_index`` (``[n]`` int64).

        Device work only, with one fixed shape per ``n``, so
        :class:`AudioEncodeRunner` can capture it. Batched over the requests:
        the encoder runs once over all ``n`` windows, then per decoder layer
        one fused ``kv_proj`` GEMM over all ``n x encoder_len`` encoder rows
        and one ``index_copy_`` each for K and V into the slots, so the launch
        count is O(decoder_layers) whatever ``n`` is.
        """
        num_requests = int(features.shape[0])
        encoder_states = self.encoder(features)
        encoder_len = self.config.max_source_positions
        if encoder_states.shape[0] != num_requests * encoder_len:
            raise ValueError(
                f"encoder produced {encoder_states.shape[0]} rows for "
                f"{num_requests} request(s); expected {num_requests * encoder_len}"
            )
        for layer_id, layer in enumerate(self.decoder.layers):
            key, value = layer.encoder_attn.project_encoder_kv(
                encoder_states, num_requests
            )
            self.cross_cache.write_layer(layer_id, slot_index, key, value)

    def make_encoder_cudagraph_wrappers(
        self, mapping: Mapping
    ) -> dict[str, AudioEncodeRunner]:
        """The graphed encoder runner, captured now (``install_encoder_graphs``).

        Called once after the weights load and before the KV cache is sized
        (``ModelRunner.prepare_multimodal_runtime``), so the graphs' memory
        pool counts against the cache budget. Gated like every multimodal
        encoder graph (``TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH``); without
        it the model keeps its eager runner. ``mapping`` is the model's own;
        a captured pass holds no collective, so the encoder must not be
        tensor-parallel.
        """
        if mapping.vision.tp_size != 1:
            raise ValueError(
                "Whisper encoder CUDA graphs capture no collective; the encoder "
                f"runs at TP {mapping.vision.tp_size}"
            )
        runner = AudioEncodeRunner(
            self,
            max_requests=self.audio_encode_runner.max_requests,
            cuda_graph=True,
        )
        torch.cuda.synchronize()
        free_before = torch.cuda.mem_get_info()[0]
        start = time.perf_counter()
        runner.capture()
        logger.info(
            f"Whisper encoder CUDA graphs captured for 1..{runner.max_requests} "
            f"request(s) in {time.perf_counter() - start:.1f} s, "
            f"{(free_before - torch.cuda.mem_get_info()[0]) / 2**20:.0f} MiB"
        )
        return {"audio_encode_runner": runner}

    def slot_state_exporters(self) -> tuple[CrossAttentionCache, ...]:
        """The model's own slot-state owners (``SlotStateModel``).

        The cross-attention KV is per-slot device state outside every cache
        group; the model executor appends it to the retraction blob so a
        restored request reads its own encoder output.
        """
        return (self.cross_cache,)

    # ------------------------------------------------------------------
    # Multimodal hooks
    # ------------------------------------------------------------------

    def get_multimodal_encoder_specs(self) -> dict:
        """No registered encoder: Whisper's encoder output is not scattered.

        The ``MultimodalEmbedder`` pipeline exists to run a tower and splice
        its output into the LM's input embeddings at placeholder offsets --
        that is what ``qwen3_asr.py`` needs. Whisper's encoder output goes to
        every decoder layer's cross-attention instead, so the model runs the
        encoder itself in :meth:`encode_audio` and registers nothing here.
        """
        return {}

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any) -> list[int]:
        """Audio reserves no positions in the decoder's token sequence.

        A prefix-style audio model pads placeholder tokens so the tower's
        output has somewhere to land. Whisper's decoder sequence is only the
        transcript prompt and what it generates; the audio lives in the
        cross-attention cache, outside this sequence entirely.
        """
        return input_ids

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    #
    # Three things the official layout forces:
    #
    #  * ``k_proj`` carries no bias. The fused self-attn QKV and cross-attn KV
    #    biases are allocated for every shard, so the k slice must be an
    #    explicit zero rather than uninitialised memory -- a zero bias *is* no
    #    bias.
    #  * ``proj_out`` is absent: Whisper ties it to the decoder token
    #    embedding. Tied here from the checkpoint's actual contents rather
    #    than from a config flag, because the checkpoint is the thing that
    #    lacks the tensor.
    #  * the encoder's position table is a fixed sinusoid buffer, not a
    #    learned ``nn.Embedding``.

    # ``model.encoder.`` / ``model.decoder.`` in the checkpoint against
    # ``encoder.`` / ``decoder.`` here: this class is the whole model and
    # does not repeat the HF wrapper level.
    _PREFIX_REWRITES = (
        ("model.encoder.", "encoder."),
        ("model.decoder.", "decoder."),
    )
    # (fused param, checkpoint projection, shard id) per attention kind:
    # self-attention fuses q/k/v; cross-attention fuses k/v (its q_proj is
    # applied every step to the decoder rows, so it stays separate).
    _STACKED_QKV = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )
    _STACKED_CROSS_KV = (
        ("kv_proj", "k_proj", 0),
        ("kv_proj", "v_proj", 1),
    )

    @classmethod
    def _rewrite(cls, name: str) -> str:
        for source, target in cls._PREFIX_REWRITES:
            if name.startswith(source):
                name = target + name[len(source) :]
                break
        # The encoder tower calls its output projection ``proj``; both stacks
        # are ``out_proj`` in the checkpoint.
        if name.startswith("encoder.layers.") and ".self_attn.out_proj." in name:
            name = name.replace(".self_attn.out_proj.", ".self_attn.proj.")
        return name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        for param_name, param in params_dict.items():
            # Self-attn ``qkv_proj`` and cross-attn ``kv_proj``: the checkpoint
            # writes the q and v slices, the k slice stays zero.
            if param_name.endswith(("qkv_proj.bias", "encoder_attn.kv_proj.bias")):
                param.data.zero_()

        loaded: set[str] = set()
        for raw_name, loaded_weight in weights:
            name = self._rewrite(raw_name)

            if name == "encoder.embed_positions.weight":
                self.encoder.embed_positions.copy_(
                    loaded_weight.to(self.encoder.embed_positions.dtype)
                )
                loaded.add(name)
                continue

            fused_target = None
            stacked = ()
            if ".self_attn." in name:
                stacked = self._STACKED_QKV
            elif ".encoder_attn." in name:
                stacked = self._STACKED_CROSS_KV
            for target, source, shard_id in stacked:
                if source in name and name.replace(source, target) in params_dict:
                    fused_target = (name.replace(source, target), shard_id)
                    break
            if fused_target is not None:
                fused_name, shard_id = fused_target
                param = params_dict[fused_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                continue

            if name not in params_dict:
                raise KeyError(
                    f"Whisper checkpoint tensor {raw_name!r} has no destination "
                    f"parameter (looked for {name!r}); the checkpoint layout "
                    "does not match this implementation"
                )
            param = params_dict[name]
            loader = getattr(param, "weight_loader", None)
            if loader is None:
                param.data.copy_(loaded_weight)
            else:
                loader(param, loaded_weight)
            loaded.add(name)

        # Tied output projection: the checkpoint ships no proj_out.
        self.proj_out.weight = self.decoder.embed_tokens.weight
        return loaded

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ):
        """The eager forward: the same seam, stack and logits tail a prefill
        graph replays (``PrefillGraph.replay``), so eager and graphed prefill
        are one path."""
        multimodal_context = kwargs.pop("multimodal_context", None)
        self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            **self.prepare_model_kwargs(ctx, input_ids, kwargs),
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            LogitsMetadata.from_forward_context(ctx),
            aux_hidden_states,
        )


# Candidate cross-KV page sizes, in order of preference (see _cross_kv_page_size).
_CROSS_KV_PAGE_SIZES = (64, 128)


def _cross_kv_page_size(dtype: torch.dtype, head_dim: int) -> int:
    """The page size the cross-KV paged view uses, checked against the registry.

    The first candidate both paged MHA ops serve at Whisper's shape (dense
    ``dtype``, ``head_dim``, non-causal extend, single-row decode, no logit
    cap / sinks / window / LSE). 64 comes first: it pads 1500 encoder rows to
    1536 (+2.4%), and it is a page size the vendor paged kernels commonly
    accept. Resolved once at model construction, so a missing kernel fails
    at load time rather than on the first request.
    """
    signature = format_signature(
        q=dense_tensor_format(dtype),
        k_cache=dense_tensor_format(dtype),
        v_cache=dense_tensor_format(dtype),
    )
    common = {
        "head_dim": head_dim,
        "logit_cap": False,
        "return_lse": False,
        "sinks": False,
        "sliding_window": False,
    }
    for page_size in _CROSS_KV_PAGE_SIZES:
        try:
            select_kernel(
                "attention",
                "mha_decode_with_kvcache",
                signature,
                traits={**common, "q_len": 1, "page_size": page_size},
            )
            select_kernel(
                "attention",
                "mha_extend_with_kvcache",
                signature,
                traits={**common, "is_causal": False, "page_size": page_size},
            )
        except NoKernelFoundError:
            continue
        return page_size
    raise NoKernelFoundError(
        f"no paged MHA decode+extend kernel serves Whisper cross-attention "
        f"(dtype {dtype}, head_dim {head_dim}) at any page size in "
        f"{_CROSS_KV_PAGE_SIZES}"
    )


def _materialize_feature(item: Any) -> torch.Tensor:
    """Get an item's log-mel tensor, pulling it out of shared memory if needed.

    Features cross the process boundary as SHM handles (``publish_shm_features``
    nulls ``feature`` on the way out). Models that go through
    ``MultimodalEmbedder`` get them materialized by its feature transport;
    Whisper does not use that path -- its encoder output feeds
    cross-attention, not input embeddings -- so it consumes the handle here.

    Single-rank only for now: the transport also decides which rank owns which
    item under TP, and doing that wrong would give one rank a silently empty
    tensor. Refused rather than guessed.
    """
    if isinstance(item.feature, torch.Tensor):
        return item.feature
    handle = item.feature_shm
    if handle is None:
        raise ValueError(
            "Whisper audio item carries neither a feature tensor nor a shared "
            "memory handle"
        )
    handle.attach()
    tensor = handle.consume()
    item.feature = tensor
    item.feature_shm = None
    return tensor


def _audio_features_to_encode(
    multimodal_context: MultimodalForwardContext, cross_cache: CrossAttentionCache
) -> tuple[list[torch.Tensor] | None, list[int], list[str]]:
    """Collect the log-mel features of the extend requests whose slot does not
    hold their own encoder output.

    ``mm_inputs``, ``request_ids`` and ``request_pool_indices`` are aligned
    with the batch, and the extend requests are its first
    ``len(extend_seq_lens)`` rows. Request ``r`` in slot ``s`` is encoded iff
    ``s`` is not owned by ``r``:

      * a fresh prefill lands in a slot some earlier request encoded (or
        nobody did), so it encodes;
      * a chunked continuation, or a recompute-retracted request re-admitted
        into the slot it last encoded, still owns it and skips the encoder;
      * a request re-admitted into a slot another request encoded since --
        possibly with a prefix hit, so ``extend_prefix_lens > 0`` -- does not
        own it and encodes its own audio again (its request state keeps the
        audio item, so every extend of the request carries it).

    Ownership, not ``extend_prefix_lens``, is what says whose rows a slot
    holds; deciding by the prefix length read the previous occupant's cross KV
    after a resume. Slots come from the host-side ``request_pool_indices``, so
    the decision reads no device tensor.

    Returns ``(features, slots, request_ids)`` with ``features`` the
    requests' ``[num_mel_bins, frames]`` windows as they arrived (host,
    pinned when they came through shared memory; :class:`AudioEncodeRunner`
    stages them), or ``(None, [], [])`` when no extend request needs
    encoding.
    """
    features: list[torch.Tensor] = []
    slots: list[int] = []
    request_ids: list[str] = []
    num_extends = len(multimodal_context.extend_seq_lens)
    for position, entry in enumerate(multimodal_context.mm_inputs[:num_extends]):
        if entry is None:
            continue
        slot = multimodal_context.request_pool_indices[position]
        request_id = multimodal_context.request_ids[position]
        if cross_cache.owner_of(slot) == request_id:
            continue
        for item in entry.mm_items:
            if item is None or item.modality != Modality.AUDIO:
                continue
            features.append(_materialize_feature(item))
            slots.append(slot)
            request_ids.append(request_id)
            # One audio window per request: Whisper's encoder consumes a
            # fixed 30s window, and a second item would overwrite the first
            # request's cross KV rather than extend it.
            break
    if not features:
        return None, [], []
    return features, slots, request_ids


EntryClass = WhisperForConditionalGeneration

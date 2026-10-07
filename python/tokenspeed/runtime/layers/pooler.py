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

"""Pooling for embedding models.

An embedding request is a single prefill whose output is one vector per
request, not a token stream. This module owns the reduction from the
per-token hidden states of that prefill to that vector.

Where the four reference engines put this differs (see
``unieinfra/xref/15-encoder-embedding.md`` §3.4): vLLM makes it a layer the
runner calls, SGLang a layer the model calls, TRT-LLM hardcodes CLS inside
the model. We follow vLLM: the model owns ``pooler`` but the reduction is
here, so changing the pooling strategy does not touch a model file.

LAST pooling deliberately reuses ``ForwardContext.gather_ids`` -- the same
last-token-per-request index the logits processor gathers with -- rather than
recomputing boundaries. One source of truth for "where does this request's
last token live", and it is already on the context.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import torch
from torch import nn


class PoolingType(str, Enum):
    """How per-token hidden states reduce to one vector."""

    LAST = "last"
    CLS = "cls"
    MEAN = "mean"


@dataclasses.dataclass
class PoolerOutput:
    """One vector per request, in batch order."""

    embeddings: torch.Tensor


def _sequence_bounds(
    extend_seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (start, end) row offsets of each request in the flat token buffer.

    ``extend_seq_lens`` counts only the tokens this forward computed. For a
    pooling request that is the whole prompt: prefix caching is off for
    pooling models, so nothing was skipped.
    """
    ends = torch.cumsum(extend_seq_lens, dim=0)
    starts = ends - extend_seq_lens
    return starts, ends


def seq_lens_from_gather_ids(gather_ids: torch.Tensor) -> torch.Tensor:
    """Recover per-request token counts from the last-token indices.

    ``gather_ids`` for an extend forward is ``cumsum(extend_lens) - 1``, so
    differencing it recovers the lengths. This holds only because a pooling
    forward is a pure extend with prefix caching off -- an embedding server
    never mixes decode rows into the batch, and never skips a cached prefix,
    so every request's rows are contiguous and complete.
    """
    ends = gather_ids + 1
    starts = torch.cat([ends.new_zeros(1), ends[:-1]])
    return ends - starts


def pool_hidden_states(
    pooling_type: PoolingType,
    hidden_states: torch.Tensor,
    *,
    gather_ids: torch.Tensor | None = None,
    extend_seq_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce ``[num_tokens, hidden]`` to ``[num_requests, hidden]``.

    ``gather_ids`` is required for LAST, ``extend_seq_lens`` for CLS and MEAN.
    Both are raised on rather than silently defaulted: a pooling kernel that
    quietly reduces over the wrong rows returns a well-formed vector that is
    simply wrong, which no shape check downstream would catch.
    """
    if pooling_type == PoolingType.LAST:
        if gather_ids is None:
            raise ValueError("LAST pooling needs gather_ids")
        return hidden_states.index_select(0, gather_ids)

    if extend_seq_lens is None:
        raise ValueError(f"{pooling_type.value} pooling needs extend_seq_lens")
    starts, ends = _sequence_bounds(extend_seq_lens)

    if pooling_type == PoolingType.CLS:
        return hidden_states.index_select(0, starts)

    if pooling_type == PoolingType.MEAN:
        # Cumulative sums over the flat buffer, differenced per request, keeps
        # this one kernel instead of a python loop over requests.
        cumulative = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)
        zero = cumulative.new_zeros((1, cumulative.shape[-1]))
        padded = torch.cat([zero, cumulative], dim=0)
        totals = padded.index_select(0, ends) - padded.index_select(0, starts)
        counts = extend_seq_lens.to(totals.dtype).unsqueeze(-1).clamp(min=1)
        return (totals / counts).to(hidden_states.dtype)

    raise ValueError(f"unsupported pooling type: {pooling_type}")


class Pooler(nn.Module):
    """Reduce prefill hidden states to one embedding per request."""

    def __init__(self, pooling_type: PoolingType, normalize: bool) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        gather_ids: torch.Tensor | None = None,
        extend_seq_lens: torch.Tensor | None = None,
    ) -> PoolerOutput:
        pooled = pool_hidden_states(
            self.pooling_type,
            hidden_states,
            gather_ids=gather_ids,
            extend_seq_lens=extend_seq_lens,
        )
        if self.normalize:
            pooled = nn.functional.normalize(pooled.float(), p=2, dim=-1).to(
                hidden_states.dtype
            )
        return PoolerOutput(embeddings=pooled)

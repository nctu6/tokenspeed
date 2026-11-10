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


"""Encoder row fit (Gemma4 pan-and-scan) on single-rank and encoder-DP paths."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.multimodal import embedder as emb
from tokenspeed.runtime.multimodal.embedder import MultimodalEmbedder, _fit_encoder_rows

WIDTH = 4


def test_fit_rows_truncate_pad_identity():
    x = torch.arange(5 * WIDTH, dtype=torch.float32).reshape(5, WIDTH)
    assert torch.equal(_fit_encoder_rows(x, 5), x)
    assert torch.equal(_fit_encoder_rows(x, 3), x[:3])
    padded = _fit_encoder_rows(x, 7)
    assert padded.shape == (7, WIDTH)
    assert torch.equal(padded[:5], x)
    assert not padded[5:].any()


def _item(rows: int):
    return SimpleNamespace(offsets=[(0, rows - 1)])


def _dp_embedder(rank: int, world: int, emitted_extra: int):
    """Embedder at ``rank`` of an encoder-DP group with a fake encoder that
    emits ``emitted_extra`` rows more (or fewer) than the placeholders."""
    e = object.__new__(MultimodalEmbedder)
    e._encoder_dp_group = list(range(world))
    e._encoder_dp_rank = rank
    e._feature_transport = SimpleNamespace(
        route_to_item_owners=lambda *a, **k: None, move_to_device=lambda *a, **k: None
    )

    def run(items, spec, device):
        rows = sum(emb._item_token_count(i) for i in items) + emitted_extra
        return torch.ones(max(rows, 0), WIDTH)

    e._run_encoder = run
    seen = {}

    def gather(local, counts):
        seen["local_rows"] = local.shape[0]
        return torch.ones(sum(counts), WIDTH)

    e._gather_encoder_outputs = gather
    return e, seen


@pytest.mark.parametrize("extra", [-2, 0, 3])
@pytest.mark.parametrize("max_tokens", [1000, 6])  # one group / several groups
def test_dp_path_fits_rows_to_placeholders(extra, max_tokens):
    items = [_item(5), _item(4), _item(6)]
    lens = [5, 4, 6]
    assignment = emb._assign_encoder_items(tuple(lens), 2)
    for rank in range(2):
        e, seen = _dp_embedder(rank, 2, extra)
        out = e._encode_data_parallel(
            items,
            None,
            torch.device("cpu"),
            WIDTH,
            torch.float32,
            max_tokens=max_tokens,
        )
        assert seen["local_rows"] == assignment.token_counts_by_rank[rank]
        assert [t.shape[0] for t in out] == lens

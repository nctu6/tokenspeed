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

"""What the prefill CUDA graph decides for the Gemma 3 / Gemma 4 wrappers.

Preservation property:
the Whisper fix may generalize the prefill graph's inner-stack contract, and
must not change it for other models. For the tiny Gemma 3 and Gemma 4
conditional-generation models, as on the unfixed code:

* the prefill graph is enabled and wraps the model's inner text stack;
* the embeddings it feeds that stack are the model's scaled
  ``get_input_embeddings`` (``sqrt(hidden_size)``), not a raw lookup -- the
  Gemma prefill-graph fix;
* a forward that carries multimodal inputs gets the same eager/replay answer
  as before (Gemma has no embeds-only multimodal seam, so it is refused).
"""

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
from test.runtime.models.test_whisper_prefill import (  # noqa: E402
    _executor_config,
    _ModelConfig,
    _ModelRunner,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

VOCAB = 64


def _gemma3():
    from transformers import Gemma3TextConfig

    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.gemma3 import Gemma3ForConditionalGeneration

    config = Gemma3TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
    )
    with torch.device("cpu"):
        return Gemma3ForConditionalGeneration(
            config=config, mapping=Mapping(rank=0, world_size=1), quant_config=None
        )


def _gemma4():
    from test.runtime.models.test_gemma4 import _build_tiny_cond_gen

    return _build_tiny_cond_gen()


BUILDERS = {"gemma3": _gemma3, "gemma4": _gemma4}


def _graph(model):
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    return PrefillGraph(
        model_runner=_ModelRunner(
            model=model,
            model_config=_ModelConfig(requires_request_token_history=False),
            is_generation=True,
            is_multimodal=True,
        ),
        attn_backend=object(),
        token_to_kv_pool=None,
        input_buffers=None,
        config=_executor_config(),
    )


def _extend_ctx(num_tokens: int):
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout

    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=1,
        input_num_tokens=num_tokens,
        forward_mode=ForwardMode.EXTEND,
        output_layout=ForwardOutputLayout(1, 1, 0, 1),
    )


def _image_context():
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalForwardContext,
        MultimodalInputs,
    )

    return MultimodalForwardContext(
        mm_inputs=[
            MultimodalInputs(
                mm_items=[
                    MultimodalDataItem(
                        modality=Modality.IMAGE, feature=torch.zeros(1, 3, 8, 8)
                    )
                ]
            )
        ],
        request_ids=["r0"],
        request_pool_indices=[1],
        extend_prefix_lens=[0],
        extend_seq_lens=[8],
        max_encoder_tokens=4096,
    )


@pytest.mark.parametrize("name", sorted(BUILDERS))
@torch.no_grad()
def test_prefill_graph_decisions_for_gemma_are_unchanged(name):
    """**Validates: Requirements 3.4, 3.5** (Property 2, prefill graph)."""
    model = BUILDERS[name]()
    model.model.embed_tokens.weight.normal_(0.0, 0.02)
    graph = _graph(model)

    assert not graph.disable, f"{name}: the prefill graph is disabled"
    assert graph.inner_model is model.model

    ids = torch.tensor([0, 1, 5, 7, VOCAB - 1], dtype=torch.int64)
    fed = graph._embed_tokens(ids)
    torch.testing.assert_close(fed, model.get_input_embeddings(ids), atol=0.0, rtol=0.0)
    hidden = model.model.embed_tokens.weight.shape[1]
    torch.testing.assert_close(
        fed, model.model.embed_tokens(ids) * hidden**0.5, atol=1e-6, rtol=1e-2
    )

    # Nothing is captured in a unit test, so the text forward runs eager; a
    # multimodal forward is refused by the missing seam before any bucket lookup.
    ctx = _extend_ctx(8)
    assert graph.can_run(ctx, None) is False
    assert graph.can_run(ctx, _image_context()) is False

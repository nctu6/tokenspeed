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

"""Building and loading Whisper: the explicit refusals, and the HF layout.

Preservation properties:

* ``--language-model-only`` and a checkpoint tensor with no destination raise
  the same errors as the pre-fix model (messages observed on the unfixed code).
* An HF-layout checkpoint loads every parameter the forward uses: two models
  with different random weights, loaded from the same checkpoint, compute the
  same logits.
"""

import os
import re
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
    whisper_config,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Whisper runtime needs CUDA"
)


def test_language_model_only_is_refused():
    """**Validates: Requirements 3.8** (Property 3, load errors)."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.whisper import WhisperForConditionalGeneration

    message = (
        "--language-model-only is not meaningful for Whisper: the decoder "
        "cross-attends to the encoder on every step, so dropping the encoder "
        "leaves no model to run."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        WhisperForConditionalGeneration(
            whisper_config(4),
            Mapping(rank=0, world_size=1),
            is_multimodal_active=False,
            mm_attention_backend="triton_attn",
        )


@pytest.mark.parametrize(
    "raw_name, looked_for",
    [
        (
            "model.decoder.layers.0.encoder_attn.bogus.weight",
            "decoder.layers.0.encoder_attn.bogus.weight",
        ),
        ("model.encoder.layers.0.unknown.bias", "encoder.layers.0.unknown.bias"),
        ("lm_head.weight", "lm_head.weight"),
    ],
)
def test_a_checkpoint_tensor_without_destination_is_refused(raw_name, looked_for):
    """**Validates: Requirements 3.8** (Property 3, load errors)."""
    model = build_whisper(4, seed=0)
    message = (
        f"Whisper checkpoint tensor {raw_name!r} has no destination parameter "
        f"(looked for {looked_for!r}); the checkpoint layout does not match this "
        "implementation"
    )
    with pytest.raises(KeyError, match=re.escape(message)):
        model.load_weights([(raw_name, torch.zeros(4))])


@torch.inference_mode()
def _prefill_logits(model, num_slots: int) -> torch.Tensor:
    runtime = WhisperRuntime(model, max_bs=num_slots)
    out = runtime.extend(
        [
            Request(1, 0, [1, 2, 3, 4], audio_features(7), rid="r0"),
            Request(3, 1, [1, 5, 6], audio_features(8), rid="r1"),
        ],
        [0, 0],
    )()
    return out.next_token_logits.float().clone()


def test_an_hf_checkpoint_sets_every_parameter_the_forward_uses():
    """**Validates: Requirements 3.1, 3.8** (Property 3, checkpoint layout)."""
    from transformers import WhisperForConditionalGeneration as HFWhisper

    config = whisper_config(4)
    # HF's special-token defaults index the full 51,866-token vocabulary.
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.decoder_start_token_id = 1
    torch.manual_seed(0)
    reference = HFWhisper(config).eval()
    with torch.no_grad():
        # HF zero-initialises biases; random ones catch a bias that is not loaded.
        for name, param in reference.named_parameters():
            if name.endswith("bias") and "layer_norm" not in name:
                param.normal_(0.0, 0.05)
    checkpoint = dict(reference.state_dict())

    logits = []
    for seed in (1, 2):
        model = build_whisper(4, seed=seed)
        loaded = model.load_weights(
            (name, tensor.to(torch.bfloat16)) for name, tensor in checkpoint.items()
        )
        assert loaded
        logits.append(_prefill_logits(model, 4))
    assert torch.equal(logits[0], logits[1]), (
        "two models loaded from one checkpoint disagree: a parameter the "
        "forward reads kept its random initial value"
    )

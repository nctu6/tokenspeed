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

"""Server-arg resolution for encoder-decoder checkpoints.

An encoder-decoder model's decoder KV depends on the encoder input through
cross-attention, but the prefix key is the decoder prompt alone, so
``ModelConfig`` forces prefix caching (and the KVStore that indexes the same
keys) off for it -- and only for it.
"""

import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-only; scheduled in runtime-1gpu because it imports the full runtime.
register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.model_config import ModelConfig  # noqa: E402
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _whisper_checkpoint(path):
    from transformers import WhisperConfig

    WhisperConfig(
        vocab_size=128,
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        decoder_layers=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=256,
        decoder_ffn_dim=256,
        d_model=128,
        max_source_positions=1500,
        max_target_positions=64,
        architectures=["WhisperForConditionalGeneration"],
    ).save_pretrained(path)
    return path


def _llama_checkpoint(path):
    from transformers import LlamaConfig

    LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        architectures=["LlamaForCausalLM"],
    ).save_pretrained(path)
    return path


def _resolve(path, **overrides) -> ServerArgs:
    fields = dict(
        model=str(path),
        device="cpu",
        dtype="bfloat16",
        trust_remote_code=False,
        max_model_len=64,
        max_num_seqs=4,
        enforce_eager=True,
        disable_prefill_graph=True,
    )
    fields.update(overrides)
    args = ServerArgs(**fields)
    ModelConfig(
        model_path=args.model,
        trust_remote_code=False,
        model_override_args="{}",
        server_args=args,
        dtype=args.dtype,
        context_length=args.max_model_len,
    )
    return args


@pytest.mark.parametrize("disable_kvstore", [False, True])
def test_encoder_decoder_model_never_prefix_caches(tmp_path, disable_kvstore):
    """**Validates: Requirement 3.7** (H14): no prefix hit can hand a request
    another request's decoder KV, whatever the operator asked for."""
    args = _resolve(_whisper_checkpoint(tmp_path), disable_kvstore=disable_kvstore)
    assert args.enable_prefix_caching is False
    assert args.enable_kvstore is False


@pytest.mark.parametrize("disable_kvstore", [False, True])
def test_decoder_only_model_keeps_prefix_caching(tmp_path, disable_kvstore):
    """**Validates: Requirement 3.4**: the guard keys on the checkpoint being
    encoder-decoder; a decoder-only model resolves exactly as before."""
    args = _resolve(_llama_checkpoint(tmp_path), disable_kvstore=disable_kvstore)
    assert args.enable_prefix_caching is True
    assert args.enable_kvstore is (not disable_kvstore)

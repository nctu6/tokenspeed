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

"""The batched cross-attention KV write of a Whisper prefill step (R2).

Properties:

* One prefill step writes every fresh request's cross KV in one batch; each
  written slot holds its own request's ``kv_proj`` output and every other
  slot (and the page padding past ``encoder_len``) is left as it was.
* The write's kernel launches do not grow with the request count, and the
  prefill forward performs no device-to-host sync, for any request count the
  pool admits.
* The fused ``kv_proj`` keeps a zero k bias after loading an HF checkpoint
  (Whisper's ``k_proj`` has no bias).

Graph replay == eager for the prefill bucket is
``test_whisper_prefill.py::test_whisper_audio_prefill_graph_replay_equals_eager``.
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
from test.runtime.models.whisper_fixtures import (  # noqa: E402
    ENCODER_LEN,
    Request,
    WhisperRuntime,
    audio_features,
    build_whisper,
    cuda_kernels,
    whisper_config,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Whisper runtime needs CUDA"
)

PROMPT = [1, 2, 3, 4]
NUM_SLOTS = 17


def _requests(slots: list[int], seed: int, tag: str) -> list[Request]:
    """Fresh requests (new ids, so the model encodes them) in ``slots``."""
    return [
        Request(slot, row, PROMPT, audio_features(seed * 100 + row), rid=f"{tag}{row}")
        for row, slot in enumerate(slots)
    ]


def _random_slots(count: int, generator: torch.Generator) -> list[int]:
    return torch.randperm(NUM_SLOTS, generator=generator)[:count].tolist()


@pytest.mark.parametrize("count", [1, 3, 8])
@torch.inference_mode()
def test_each_slot_holds_its_own_kv_proj_and_other_slots_are_untouched(count):
    """**Validates: Requirements 2.5, 3.7** (Property 4, own cross KV).

    One prefill forward with ``count`` fresh requests. Afterwards each
    request's slot holds ``kv_proj`` of that request's encoder rows alone;
    slots outside the step, and every slot's page padding past
    ``encoder_len``, still hold what was there before.
    """
    model = build_whisper(NUM_SLOTS, seed=count)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    cache = model.cross_cache
    generator = torch.Generator(device="cuda").manual_seed(count)
    # A recognizable prior content: a write that spills over it shows.
    cache.k.normal_(generator=generator)
    cache.v.normal_(generator=generator)
    before_k, before_v = cache.k.clone(), cache.v.clone()

    slots = _random_slots(count, torch.Generator().manual_seed(count))
    requests = _requests(slots, seed=count, tag="w")
    runtime.extend(requests, [0] * count)()
    torch.cuda.synchronize()

    # The reference encoder pass over the same batch (the encoder is not the
    # code under test), then each request's rows projected on their own.
    features = torch.stack([r.audio for r in requests]).cuda()
    states = model.encoder(features).view(count, ENCODER_LEN, -1)
    for layer_id, layer in enumerate(model.decoder.layers):
        attn = layer.encoder_attn
        for row, slot in enumerate(slots):
            kv, _ = attn.kv_proj(states[row])
            k, v = kv.split([attn.kv_size, attn.kv_size], dim=-1)
            shape = (ENCODER_LEN, attn.num_heads, attn.head_dim)
            torch.testing.assert_close(
                cache.k[layer_id, slot, :ENCODER_LEN], k.reshape(shape)
            )
            torch.testing.assert_close(
                cache.v[layer_id, slot, :ENCODER_LEN], v.reshape(shape)
            )

    untouched = [s for s in range(NUM_SLOTS) if s not in slots]
    assert torch.equal(cache.k[:, untouched], before_k[:, untouched])
    assert torch.equal(cache.v[:, untouched], before_v[:, untouched])
    padding = slice(ENCODER_LEN, cache.padded_len)
    assert torch.equal(cache.k[:, :, padding], before_k[:, :, padding])
    assert torch.equal(cache.v[:, :, padding], before_v[:, :, padding])


@pytest.mark.parametrize("seed", range(4))
@torch.inference_mode()
def test_write_launches_are_flat_in_the_request_count_and_never_sync(seed):
    """**Validates: Requirements 1.5, 2.5** (Property 1, prefill).

    Seeded property over request counts in [1, 16] (always including both
    ends) and random slots:

    * the cross-KV write's kernel launches -- ``encode_audio``'s kernels
      minus those of the encoder pass it starts with, over the same
      features -- are the same for every count;
    * the whole prefill forward runs under ``set_sync_debug_mode("error")``.

    Subtracting the encoder pass cancels its library choices that depend on
    the batch (cuDNN conv algorithm at batch 1), which are not a
    per-request loop.
    """
    generator = torch.Generator().manual_seed(seed)
    counts = {1, NUM_SLOTS - 1}
    counts.update(torch.randint(2, NUM_SLOTS - 1, (2,), generator=generator).tolist())
    model = build_whisper(NUM_SLOTS, seed=seed)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)

    write_launches = {}
    for count in sorted(counts):
        slots = _random_slots(count, generator)
        requests = _requests(slots, seed=seed * 1000 + count, tag=f"s{count}-")
        features = torch.stack([r.audio for r in requests]).cuda()
        rids = [r.rid for r in requests]

        def encode():
            model.encode_audio(features, slots, rids)

        def encoder():
            model.encoder(features)

        encode()  # warm-up: first launch of a shape may autotune
        write_launches[count] = len(cuda_kernels(encode)) - len(cuda_kernels(encoder))

        # The whole forward, with the features as requests carry them (host).
        runtime.extend(_requests(slots, seed, f"warm{count}-"), [0] * count)()
        run = runtime.extend(_requests(slots, seed, f"run{count}-"), [0] * count)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()

    print(f"cross-KV write launches by request count: {write_launches}")
    assert (
        len(set(write_launches.values())) == 1
    ), f"cross-KV write launches grow with the request count: {write_launches}"


def test_fused_kv_proj_keeps_a_zero_k_bias_after_an_hf_load():
    """**Validates: Requirements 2.5, 3.1** (Property 3, checkpoint layout).

    HF Whisper's cross-attention ``k_proj`` has no bias, so the fused
    ``kv_proj`` bias must load as ``[0, v_proj.bias]`` -- exactly zero in the
    k slice even when the parameter held something else before loading --
    and its weight as ``[k_proj.weight; v_proj.weight]``.
    """
    from transformers import WhisperForConditionalGeneration as HFWhisper

    config = whisper_config(4)
    config.pad_token_id = 0
    config.bos_token_id = 1
    config.eos_token_id = 2
    config.decoder_start_token_id = 1
    torch.manual_seed(0)
    reference = HFWhisper(config).eval()
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if name.endswith("bias") and "layer_norm" not in name:
                param.normal_(0.0, 0.05)
    checkpoint = {
        name: tensor.to(torch.bfloat16)
        for name, tensor in reference.state_dict().items()
    }

    # build_whisper fills every bias with random values: a k slice left
    # unzeroed by the loader stays nonzero.
    model = build_whisper(4, seed=0)
    model.load_weights(checkpoint.items())
    for layer_id, layer in enumerate(model.decoder.layers):
        attn = layer.encoder_attn
        prefix = f"model.decoder.layers.{layer_id}.encoder_attn"
        bias = attn.kv_proj.bias
        k_bias, v_bias = bias.split([attn.kv_size, attn.kv_size])
        assert torch.count_nonzero(k_bias) == 0, f"layer {layer_id}: k bias not zero"
        assert torch.equal(v_bias.cpu(), checkpoint[f"{prefix}.v_proj.bias"])
        k_weight, v_weight = attn.kv_proj.weight.split([attn.kv_size, attn.kv_size])
        assert torch.equal(k_weight.cpu(), checkpoint[f"{prefix}.k_proj.weight"])
        assert torch.equal(v_weight.cpu(), checkpoint[f"{prefix}.v_proj.weight"])


def _graphed_runner(model):
    """The graphed encoder runner the multimodal encoder-graph install puts on
    the model (``MultimodalRuntime.install_encoder_graphs``)."""
    (runner,) = model.make_encoder_cudagraph_wrappers(model.mapping).values()
    return runner


@pytest.mark.parametrize("backend", ["triton_attn", "sdpa"])
@pytest.mark.parametrize("seed", range(3))
@torch.inference_mode()
def test_graphed_encoder_pass_writes_what_the_eager_pass_writes(seed, backend):
    """**Validates: Requirements 2.1, 2.5, 3.6** (Property 4, graph == eager).

    Seeded property over request counts in [1, NUM_SLOTS] (always including
    1, the largest captured pass and NUM_SLOTS, which takes two passes) and
    random slots: the graphed runner leaves the cross-KV buffer bit-identical
    to the eager runner's -- the written slots hold the same rows, every
    other slot and the page padding are untouched -- and claims each slot
    for its request. For both encoder attention backends Whisper serves
    with (``triton_attn``, ``sdpa``).
    """
    generator = torch.Generator().manual_seed(seed)
    model = build_whisper(NUM_SLOTS, seed=seed, mm_attention_backend=backend)
    eager = model.audio_encode_runner
    graphed = _graphed_runner(model)
    assert graphed.max_requests < NUM_SLOTS  # a full pool takes two passes
    cache = model.cross_cache
    counts = {1, graphed.max_requests, NUM_SLOTS}
    counts.update(torch.randint(2, NUM_SLOTS, (2,), generator=generator).tolist())

    for count in sorted(counts):
        slots = _random_slots(count, generator)
        requests = _requests(slots, seed=seed * 1000 + count, tag=f"g{count}-")
        features = [r.audio for r in requests]
        rids = [r.rid for r in requests]
        prior = torch.Generator(device="cuda").manual_seed(seed * 1000 + count)
        cache.k.normal_(generator=prior)
        cache.v.normal_(generator=prior)
        before_k, before_v = cache.k.clone(), cache.v.clone()

        model.audio_encode_runner = eager
        model.encode_audio(features, slots, rids)
        torch.cuda.synchronize()
        eager_k, eager_v = cache.k.clone(), cache.v.clone()

        cache.k.copy_(before_k)
        cache.v.copy_(before_v)
        for slot in slots:
            cache.release_slot(slot)
        model.audio_encode_runner = graphed
        model.encode_audio(features, slots, rids)
        torch.cuda.synchronize()

        assert torch.equal(cache.k, eager_k), f"{count} requests: K differs"
        assert torch.equal(cache.v, eager_v), f"{count} requests: V differs"
        untouched = [s for s in range(NUM_SLOTS) if s not in slots]
        assert torch.equal(cache.k[:, untouched], before_k[:, untouched])
        assert all(cache.owner_of(s) == r for s, r in zip(slots, rids))


@pytest.mark.parametrize("seed", range(2))
@torch.inference_mode()
def test_graphed_encoder_prefill_equals_eager_and_never_syncs(seed):
    """**Validates: Requirements 2.5, 3.6** (Property 1 prefill, Property 4).

    A whole prefill forward with the graphed encoder runner installed gives
    the eager forward's logits exactly, for one request and for a step of
    several, and runs under ``set_sync_debug_mode("error")``.
    """
    generator = torch.Generator().manual_seed(seed)
    model = build_whisper(NUM_SLOTS, seed=seed)
    runtime = WhisperRuntime(model, max_bs=NUM_SLOTS)
    eager = model.audio_encode_runner
    graphed = _graphed_runner(model)
    for count in (1, int(torch.randint(2, NUM_SLOTS, (1,), generator=generator))):
        slots = _random_slots(count, generator)

        model.audio_encode_runner = eager
        expected = runtime.extend(_requests(slots, seed, f"e{count}-"), [0] * count)()
        expected = expected.next_token_logits.clone()

        model.audio_encode_runner = graphed
        run = runtime.extend(_requests(slots, seed, f"g{count}-"), [0] * count)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            output = run()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.synchronize()
        assert torch.equal(output.next_token_logits, expected), f"{count} requests"


@torch.inference_mode()
def test_encoder_pass_rejects_a_window_of_the_wrong_shape():
    """**Validates: Requirement 3.2** (Property 3, clear errors).

    The feature extractor pads every clip to the model's window; a window of
    another length cannot fill ``encoder_len`` rows and is refused before any
    device work, for the eager and the graphed runner alike.
    """
    model = build_whisper(4, seed=0)
    short = audio_features(0)[:, :-2]
    for runner in (model.audio_encode_runner, _graphed_runner(model)):
        model.audio_encode_runner = runner
        with pytest.raises(ValueError, match="window"):
            model.encode_audio([short], [1], ["short"])
        assert model.cross_cache.owner_of(1) is None


@pytest.mark.parametrize("count", [1, 3])
@torch.inference_mode()
def test_sdpa_encoder_matches_the_triton_encoder(count):
    """**Validates: Requirements 2.1, 3.1** (encoder numerics).

    The same weights encode the same windows to the same states (within
    bf16 rounding) whichever encoder attention backend the model was built
    with, at one request and at several.
    """
    triton = build_whisper(4, seed=count)
    sdpa = build_whisper(4, seed=count, mm_attention_backend="sdpa")
    features = torch.stack([audio_features(count * 10 + i) for i in range(count)])
    got = sdpa.encoder(features.cuda()).float()
    want = triton.encoder(features.cuda()).float()
    # Two bf16 ulps of the LayerNorm-ed states (|x| up to ~4) at worst, and
    # well under one ulp on average: the attention kernels' accumulation
    # orders differ, nothing else does.
    torch.testing.assert_close(got, want, atol=3e-2, rtol=1.6e-2)
    assert (got - want).abs().mean() < 5e-3 * want.abs().mean()

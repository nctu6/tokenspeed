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

"""Whisper cross-attention: what a step reads, and whose encoder output it reads.

Properties of the fixed read path:

* A decode step reads exactly the batch's cross KV: the same kernels and no
  copy of the batch's KV, whatever the request-pool size.
* A request's cross-attention always reads its own encoder output, also when
  it resumes into a slot another request encoded (retraction with recompute,
  chunked continuation) -- the step after resume carries its audio again.

Preservation properties (Property 4; hold on the unfixed code and must keep
holding):

* Through any sequence of fresh admits, decode steps, finishes, aborts and
  retractions (snapshot or recompute, into another slot) over a small pool --
  slots reused by new requests, random batch compositions -- every logit row
  equals the same request run alone.
* An extend runs the encoder for a request iff its slot does not hold that
  request's encoder output.
* A captured decode graph replayed with new slots written in place equals the
  eager forward of the same batch.
* A decode forward performs no device-to-host sync.
"""

import dataclasses
import os
import random
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
    CONTEXT_LEN,
    D_MODEL,
    DECODER_LAYERS,
    ENCODER_LEN,
    VOCAB,
    Request,
    WhisperRuntime,
    audio_features,
    build_whisper,
    cuda_kernels,
    peak_bytes,
)

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Whisper runtime needs CUDA"
)

PROMPT = [1, 2, 3, 4]


@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_decode_cross_read_does_not_depend_on_the_request_pool_size(seed):
    """**Validates: Requirements 2.1, 2.2, 2.3** (Property 1, crossRead).

    Same weights, same bs=16 decode batch, two pool sizes (the slot count is
    ``max_num_seqs + 1``: MNS=16 and MNS=32). The cross read must launch the
    same kernels for both and must not materialise the batch's cross KV.
    """
    bs = 16
    generator = torch.Generator().manual_seed(seed)
    kernels, peaks = {}, {}
    for num_slots in (17, 33):
        model = build_whisper(num_slots, seed=seed)
        runtime = WhisperRuntime(model, max_bs=num_slots)
        slots = torch.randperm(num_slots, generator=generator)[:bs].tolist()
        run = runtime.decode(slots, page_rows=list(range(bs)), seq_lens=[5] * bs)
        run()  # warm-up: kernel compilation and autotune stay out of the measurement
        kernels[num_slots] = cuda_kernels(run)
        peaks[num_slots] = peak_bytes(run)
        del model, runtime
        torch.cuda.empty_cache()

    # One layer's K (or V) for the batch: any copy of the batch's cross KV is at least this.
    batch_kv_bytes = bs * ENCODER_LEN * D_MODEL * 2
    print(
        f"cross read: kernels 17 slots={len(kernels[17])} 33 slots={len(kernels[33])}; "
        f"peak 17 slots={peaks[17]} B, 33 slots={peaks[33]} B, "
        f"batch K of one layer={batch_kv_bytes} B"
    )
    assert kernels[17] == kernels[33], (
        "decode launches different kernels for the same batch at 17 and 33 "
        f"slots: only in 17={sorted(set(kernels[17]) - set(kernels[33]))}, "
        f"only in 33={sorted(set(kernels[33]) - set(kernels[17]))}"
    )
    for num_slots, peak in peaks.items():
        assert peak < batch_kv_bytes, (
            f"{num_slots} slots: the decode step allocated {peak} B, at least one "
            f"layer's batch cross KV ({batch_kv_bytes} B of {DECODER_LAYERS} "
            "layers): the read copies KV instead of reading it in place"
        )


@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_resumed_request_reads_its_own_cross_kv_in_a_reused_slot(seed):
    """**Validates: Requirements 3.7** (Property 1/4, stale cross KV).

    Request B prefills its prompt in slot 2, then is resumed in slot 5 after
    request A encoded slot 5 (a recompute retraction that re-admits B into a
    different slot keeps B's prefix pages and carries B's audio again). B's
    next step must equal the same step run in B's original slot.
    """
    model = build_whisper(8, seed=seed)
    runtime = WhisperRuntime(model, max_bs=8)
    audio_a, audio_b = audio_features(100 + seed), audio_features(200 + seed)
    tokens_b = PROMPT + [7]

    # B prefills its prompt in slot 2, page row 0; then its next token in place.
    runtime.extend([Request(2, 0, PROMPT, audio_b, rid="b")], [0])()
    own_slot = runtime.extend(
        [Request(2, 0, tokens_b, audio_b, rid="b")], [len(PROMPT)]
    )()
    # A takes slot 5 (page row 1) and encodes its own audio there.
    runtime.extend([Request(5, 1, PROMPT, audio_a, rid="a")], [0])()
    # B resumes in slot 5 with its prefix pages (row 0) and its audio.
    reused_slot = runtime.extend(
        [Request(5, 0, tokens_b, audio_b, rid="b")], [len(PROMPT)]
    )()

    expected = own_slot.next_token_logits.float()
    actual = reused_slot.next_token_logits.float()
    print(
        "stale cross KV: max |resumed - own slot| = "
        f"{(actual - expected).abs().max().item():.4f}"
    )
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_extend_encodes_iff_the_slot_is_not_the_requests_own():
    """**Validates: Requirement 3.7** (R5, encode by ownership).

    Through the model's real extend path: a slot's rows are rewritten by an
    extend exactly when the slot does not hold the request's own encoder
    output. Marker rows written over a slot reveal whether the encoder ran.
    A continuation of the owner (any prefix) leaves them; a new request id in
    the slot -- the previous occupant finished, no release event -- re-encodes
    with a prefix hit; a restored snapshot claims its slot.
    """
    model = build_whisper(8, seed=0)
    runtime = WhisperRuntime(model, max_bs=8)
    cross = model.cross_cache
    audio_a, audio_b = audio_features(10), audio_features(20)
    tokens = PROMPT + [7]

    def scribble(slot: int) -> None:
        for rows in cross.slot_state_rows(slot):
            rows.fill_(3.0)

    def scribbled(slot: int) -> bool:
        return all(bool(torch.all(rows == 3.0)) for rows in cross.slot_state_rows(slot))

    runtime.extend([Request(2, 0, PROMPT, audio_a, rid="a")], [0])()
    assert cross.owner_of(2) == "a"
    own_rows = [rows.clone() for rows in cross.slot_state_rows(2)]

    # The owner's continuation does not run the encoder over its slot.
    scribble(2)
    runtime.extend([Request(2, 0, tokens, audio_a, rid="a")], [len(PROMPT)])()
    assert scribbled(2)

    # "a" finished; "b" lands in slot 2 with a prefix hit and encodes its own.
    runtime.extend([Request(2, 1, tokens, audio_b, rid="b")], [len(PROMPT)])()
    assert cross.owner_of(2) == "b" and not scribbled(2)
    # "a" resumed by recompute into slot 2 encodes again: its own rows return.
    runtime.extend([Request(2, 0, tokens, audio_a, rid="a")], [len(PROMPT)])()
    assert cross.owner_of(2) == "a"
    for rows, expected in zip(cross.slot_state_rows(2), own_rows):
        torch.testing.assert_close(rows, expected, atol=1e-2, rtol=1e-2)

    # A snapshot restored into slot 6 claims it: the next extend skips the encoder.
    image = torch.empty((cross.slot_state_bytes(),), dtype=torch.uint8, pin_memory=True)
    cross.export_slot_state(2, image, None, request_id="a")
    torch.cuda.synchronize()
    cross.import_slot_state(6, image, None, request_id="a")
    torch.cuda.synchronize()
    assert cross.owner_of(6) == "a"
    scribble(6)
    runtime.extend([Request(6, 0, tokens, audio_a, rid="a")], [len(PROMPT)])()
    assert scribbled(6)


@dataclasses.dataclass
class _Life:
    """One request through the lifecycle: where it lives and what it produced."""

    slot: int
    # The request's decoder KV pages. A retraction moves the request to
    # another slot but its pages keep their contents (a snapshot restores
    # them, a recompute re-matches them as a prefix), so the row stays.
    page_row: int
    audio_seed: int
    prompt: list[int]
    tokens: list[int]
    # Next-token logits after the prompt, then after every decoded token.
    logits: list[torch.Tensor]
    # What produced each logit row (and the moves before it), for failure messages.
    history: list[str] = dataclasses.field(default_factory=list)
    moves: list[str] = dataclasses.field(default_factory=list)
    # Logit row -> prefix length, for rows a recompute extend produced: the
    # solo reference replays those rows as the same extend (extend and decode
    # kernels round bf16 differently, which is not what this property checks).
    recomputed: dict[int, int] = dataclasses.field(default_factory=dict)

    def produced(self, label: str) -> None:
        self.history.append(" ".join(self.moves + [label]))
        self.moves.clear()

    @property
    def rid(self) -> str:
        return f"life{self.audio_seed}"


def _solo_logits(solo: WhisperRuntime, life: _Life) -> list[torch.Tensor]:
    """The same request run alone in slot 0 of a fresh pool: the reference."""
    audio = audio_features(life.audio_seed)
    out = solo.extend([Request(0, 0, life.prompt, audio, rid=life.rid)], [0])()
    rows = [out.next_token_logits[0].float().clone()]
    for length in range(len(life.prompt) + 1, len(life.tokens) + 1):
        prefix = life.recomputed.get(len(rows))
        if prefix is None:
            out = solo.decode([0], [0], [length], tokens=[life.tokens[length - 1]])()
        else:
            out = solo.extend(
                [Request(0, 0, life.tokens[:length], audio, rid=life.rid)], [prefix]
            )()
        rows.append(out.next_token_logits[0].float().clone())
    return rows


def _check_against_solo(solo: WhisperRuntime, life: _Life) -> float:
    expected = _solo_logits(solo, life)
    assert len(expected) == len(life.logits)
    worst = 0.0
    for step, (actual, reference) in enumerate(zip(life.logits, expected)):
        worst = max(worst, (actual - reference).abs().max().item())
        torch.testing.assert_close(
            actual,
            reference,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m, step=step: (
                f"slot {life.slot}, step {step} ({life.history[step]}); "
                f"history {life.history}: {m}"
            ),
        )
    return worst


@pytest.mark.parametrize("seed", range(6))
@torch.inference_mode()
def test_every_request_reads_its_own_cross_kv_through_admit_decode_finish(seed):
    """**Validates: Requirements 3.6, 3.7** (Property 4, cross KV ownership).

    A seeded lifecycle over a pool of at most 8 slots: batches of fresh admits
    (prefix 0), decode steps over random subsets in random order, finishes and
    aborts that free a slot, new requests admitted into freed slots, and
    retractions that move a running request into another slot:

    * retract-with-snapshot: the slot's cross-KV image is exported, the
      vacated rows are overwritten, the image is imported into a different
      free slot, and the request continues decoding there;
    * retract-with-recompute: the request is re-admitted into a free slot
      (possibly its own, possibly one a finished request encoded) by an
      extend with a prefix hit and its audio attached.

    The decoder KV pages of a retracted request keep their row (a snapshot
    restores them, a recompute re-matches them). Every logit row a request
    produced equals the same request run alone.
    """
    from tokenspeed.runtime.execution.slot_state import read_prepared_marker

    rng = random.Random(seed)
    num_slots = rng.randint(4, 8)
    runtime = WhisperRuntime(build_whisper(num_slots, seed=seed), max_bs=num_slots)
    solo = WhisperRuntime(build_whisper(1, seed=seed), max_bs=1)
    cross = runtime.model.cross_cache

    free = list(range(num_slots))
    free_rows = list(range(num_slots))
    active: dict[int, _Life] = {}
    next_audio = 1000 * seed
    checked, worst, events = 0, 0.0, []

    def growable() -> list[_Life]:
        return [life for life in active.values() if len(life.tokens) < CONTEXT_LEN - 1]

    for _ in range(18):
        draw = rng.random()
        if free and (not active or draw < 0.3):
            # Admit 1..3 fresh requests in one extend.
            count = rng.randint(1, min(3, len(free)))
            slots = rng.sample(free, count)
            lives = []
            for slot in slots:
                free.remove(slot)
                row = free_rows.pop(rng.randrange(len(free_rows)))
                prompt = [1] + [rng.randrange(VOCAB) for _ in range(rng.randint(1, 5))]
                lives.append(_Life(slot, row, next_audio, prompt, list(prompt), []))
                next_audio += 1
            out = runtime.extend(
                [
                    Request(
                        life.slot,
                        life.page_row,
                        life.prompt,
                        audio_features(life.audio_seed),
                        rid=life.rid,
                    )
                    for life in lives
                ],
                [0] * count,
            )()
            for index, life in enumerate(lives):
                life.logits.append(out.next_token_logits[index].float().clone())
                life.produced(f"admit(bs={count})")
                active[life.slot] = life
            events.append(f"admit{sorted(slots)}")
        elif active and draw < 0.65:
            # Decode a random subset in random order (one new token each).
            candidates = growable()
            if not candidates:
                continue
            batch = rng.sample(candidates, rng.randint(1, len(candidates)))
            new_tokens = [rng.randrange(VOCAB) for _ in batch]
            for life, token in zip(batch, new_tokens):
                life.tokens.append(token)
            slots = [life.slot for life in batch]
            out = runtime.decode(
                slots,
                [life.page_row for life in batch],
                [len(life.tokens) for life in batch],
                tokens=new_tokens,
            )()
            for index, life in enumerate(batch):
                life.logits.append(out.next_token_logits[index].float().clone())
                life.produced(f"decode(bs={len(batch)})")
            events.append(f"decode{slots}")
        elif active and free and draw < 0.75:
            # Retract with snapshot: image the slot, scribble over the vacated
            # rows, restore into a different free slot; decoding continues there.
            src = rng.choice(sorted(active))
            dst = rng.choice(free)
            life = active.pop(src)
            free.remove(dst)
            free.append(src)
            image = torch.empty(
                (cross.slot_state_bytes(),), dtype=torch.uint8, pin_memory=True
            )
            cross.export_slot_state(src, image, None, request_id=life.rid)
            torch.cuda.synchronize()
            assert read_prepared_marker(image), "a running request owns its slot"
            for rows in cross.slot_state_rows(src):
                rows.normal_()
            cross.import_slot_state(dst, image, None, request_id=life.rid)
            assert cross.owner_of(dst) == life.rid
            life.slot = dst
            life.moves.append(f"snapshot({src}->{dst})")
            active[dst] = life
            events.append(f"snapshot[{src}->{dst}]")
        elif active and draw < 0.85:
            # Retract with recompute: re-admit into a free slot or its own,
            # with a prefix hit on its pages and its audio attached; the
            # extend feeds the new token (and re-feeds any recomputed suffix).
            candidates = growable()
            if not candidates:
                continue
            life = rng.choice(candidates)
            src = life.slot
            del active[src]
            free.append(src)
            dst = rng.choice(free)
            free.remove(dst)
            life.tokens.append(rng.randrange(VOCAB))
            prefix = rng.randint(1, len(life.tokens) - 1)
            out = runtime.extend(
                [
                    Request(
                        dst,
                        life.page_row,
                        life.tokens,
                        audio_features(life.audio_seed),
                        rid=life.rid,
                    )
                ],
                [prefix],
            )()
            life.recomputed[len(life.logits)] = prefix
            life.logits.append(out.next_token_logits[0].float().clone())
            life.produced(
                f"recompute({src}->{dst}, prefix={prefix}/{len(life.tokens)})"
            )
            life.slot = dst
            active[dst] = life
            events.append(f"recompute[{src}->{dst},prefix={prefix}]")
        elif active:
            # Finish or abort: the slot is freed and may be reused by the next admit.
            slot = rng.choice(sorted(active))
            life = active.pop(slot)
            free.append(slot)
            free_rows.append(life.page_row)
            worst = max(worst, _check_against_solo(solo, life))
            checked += 1
            events.append(f"{rng.choice(('finish', 'abort'))}[{slot}]")
    for life in active.values():
        worst = max(worst, _check_against_solo(solo, life))
        checked += 1
    print(
        f"seed {seed}: {num_slots} slots, {checked} requests checked, "
        f"max |batched - solo| = {worst:.4f}; events: {' '.join(events)}"
    )


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("bs", [1, 16])
@torch.inference_mode()
def test_decode_forward_never_syncs_the_host(bs, seed):
    """**Validates: Requirements 2.1, 3.6** (Property 4, no D2H sync).

    The slot lookup of the cross read (page table from ``req_pool_indices``)
    stays on device, so a whole decode forward runs under
    ``set_sync_debug_mode("error")``. Inputs are refreshed before the guarded
    region: the host-to-device copies that stage them are the caller's, not
    the forward's.
    """
    num_slots = 17
    rng = random.Random(seed)
    runtime = WhisperRuntime(build_whisper(num_slots, seed=seed), max_bs=num_slots)
    slots = rng.sample(range(num_slots), bs)
    runtime.extend(
        [
            Request(slot, slot, PROMPT, audio_features(300 + slot), rid=f"r{slot}")
            for slot in slots
        ],
        [0] * bs,
    )()
    seq_len = len(PROMPT) + 1
    step = runtime.decode(slots, slots, [seq_len] * bs)
    step()  # warm-up: compilation and autotune may sync
    step.refresh(slots, slots, [seq_len] * bs, None)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


@pytest.mark.parametrize("bs,num_slots", [(2, 8), (4, 8), (16, 17), (16, 33)])
@torch.inference_mode()
def test_decode_graph_replay_equals_eager_with_slots_refreshed_in_place(bs, num_slots):
    """**Validates: Requirements 3.6** (Property 4, graph equals eager).

    Capture one decode forward, then replay it for new batches whose slots,
    tokens and decode metadata are written into the captured inputs in place.
    Each replay equals the eager forward of the same batch, bit for bit. On the
    unfixed code bs=2 of 8 slots takes the gather read and bs=4 the padded one;
    bs=16 runs at both production pool sizes (max_num_seqs 16 and 32).
    """
    rng = random.Random(bs * 100 + num_slots)
    runtime = WhisperRuntime(build_whisper(num_slots, seed=bs), max_bs=num_slots)
    prompt = [1, 2, 3, 4]
    runtime.extend(
        [
            Request(slot, slot, prompt, audio_features(500 + slot), rid=f"r{slot}")
            for slot in range(num_slots)
        ],
        [0] * num_slots,
    )()
    seq_len = len(prompt) + 1

    step = runtime.decode(list(range(bs)), list(range(bs)), [seq_len] * bs)
    step()  # eager warm-up: compilation, autotune and lazily built buffers
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()

    for trial in range(4):
        slots = rng.sample(range(num_slots), bs)
        tokens = [rng.randrange(VOCAB) for _ in range(bs)]
        step.refresh(slots, slots, [seq_len] * bs, tokens)
        expected = step().next_token_logits.clone()
        step.refresh(slots, slots, [seq_len] * bs, tokens)
        graph.replay()
        torch.cuda.synchronize()
        actual = captured.next_token_logits
        print(
            f"bs={bs} trial {trial} slots {slots}: max |replay - eager| = "
            f"{(actual.float() - expected.float()).abs().max().item():.6f}"
        )
        assert torch.equal(actual, expected), f"slots {slots}: replay != eager"


def test_the_model_lists_its_cross_kv_as_its_only_slot_state_owner():
    """**Validates: Requirement 3.7.** The executor learns the model's own
    slot-state owners through the generic ``SlotStateModel`` hook: Whisper
    lists its cross-attention cache, a model without per-slot state nothing."""
    from tokenspeed.runtime.execution.slot_state import (
        SlotStateExporter,
        model_slot_state_exporters,
    )

    model = build_whisper(4, seed=0)
    exporters = model_slot_state_exporters(model)
    assert exporters == (model.cross_cache,)
    assert isinstance(model.cross_cache, SlotStateExporter)
    # Encoder rows of every decoder layer, K then V, behind the marker.
    image_bytes = model.cross_cache.slot_state_bytes()
    assert image_bytes >= 2 * DECODER_LAYERS * ENCODER_LEN * D_MODEL * 2
    # The decoder alone is an ordinary module: it owns no slot state.
    assert model_slot_state_exporters(model.decoder) == ()

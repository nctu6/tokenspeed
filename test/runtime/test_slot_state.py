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

"""The slot-state image: every per-slot owner exports what it must, exactly.

CPU-only. Two guards per owner (``RuntimeStates``, the drafters with
cross-round state, the Inkling ring, the DSA KPool tail, the executor's
aggregate):

* a round trip -- export one slot's rows into a uint8 image, wipe them, import
  into another slot, compare byte for byte;
* the completeness test -- every tensor an owner (or a runtime component it
  holds) allocates with a slot-sized dimension is either reachable from
  ``slot_state_rows`` or named in the exporter's ``token_derived_slot_state``
  / ``constant_slot_state``, which every exporter class must declare. A
  per-slot buffer added to a constructed owner without an exporter fails
  here instead of silently breaking a restored request; an owner class this
  file does not construct is not guarded.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-only; scheduled in runtime-1gpu because it imports the full runtime.
register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.drafter.mtp import Mtp  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.slot_state import (  # noqa: E402
    PREPARED_MARKER_BYTES,
    SLOT_STATE_ALIGNMENT,
    SlotStateExporter,
    SlotStateLayout,
    pack_slot_rows,
    read_prepared_marker,
    slot_state_image_bytes,
    unpack_slot_rows,
)
from tokenspeed.runtime.layers.attention.backends.base import (  # noqa: E402
    AttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import (  # noqa: E402
    DSABackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.inkling import (  # noqa: E402
    InklingAttnBackend,
    InklingConvStatePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (  # noqa: E402
    HybridGlm53FlashTokenToKVPool,
    _KPoolTailWorkspace,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.flashinfer import (  # noqa: E402
    FlashInferSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.flashinfer_full import (  # noqa: E402
    FlashInferFullSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.greedy import (  # noqa: E402
    GreedySamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import (  # noqa: E402
    TritonFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.sampling.utils import coin_eps  # noqa: E402

SAMPLING_BACKENDS = (
    GreedySamplingBackend,
    FlashInferSamplingBackend,
    FlashInferFullSamplingBackend,
    TritonSamplingBackend,
    TritonFullSamplingBackend,
)

# One request-pool geometry for every owner: max_bs decode rows, one sink row
# (max_req_pool_size = max_bs + 1) and the graph-padding row past it.
MAX_BS = 9
MAX_REQ_POOL_SIZE = MAX_BS + 1
POOL_ROWS = MAX_REQ_POOL_SIZE + 1  # 11: RuntimeStates, drafters, Inkling (max_bs + 2)
SPEC_TOKENS = 4
VOCAB = 32
HIDDEN = 8


# ----------------------------------------------------------------------
# Builders: real owner classes over fakes for the runners they never touch here.
# ----------------------------------------------------------------------


def _runtime_states(*, draft_probs: bool, trees: bool, history: bool) -> RuntimeStates:
    states = RuntimeStates(
        req_pool_size=MAX_REQ_POOL_SIZE,
        vocab_size=VOCAB,
        output_length=SPEC_TOKENS,
        device="cpu",
    )
    if draft_probs:
        states.init_draft_probs(spec_num_tokens=SPEC_TOKENS, reject_threshold=2.0)
    if trees:
        states.init_draft_trees(SPEC_TOKENS)
    if history:
        states.init_ngram_state(3)
        states.init_request_token_history(16)
        states.init_draft_request_token_history(16)
    return states


def _input_buffers() -> SimpleNamespace:
    return SimpleNamespace(
        max_bs=MAX_BS,
        state_write_padding_pool_index=MAX_REQ_POOL_SIZE,
        seq_lens_buf=torch.zeros(MAX_BS, dtype=torch.int32),
        req_pool_indices_buf=torch.zeros(MAX_BS, dtype=torch.int64),
        extend_prefix_lens_cpu=torch.zeros(MAX_BS, dtype=torch.int64),
    )


def _mtp(states: RuntimeStates) -> Mtp:
    class _MtpModel:
        num_mtp_layers = 3

    runner = SimpleNamespace(
        device="cpu",
        forward_accepts_spec_step_idx=True,
        model=_MtpModel(),
        model_config=SimpleNamespace(hidden_size=HIDDEN, dtype=torch.bfloat16),
    )
    return Mtp(
        spec_num_tokens=SPEC_TOKENS,
        spec_num_steps=3,
        draft_model_runner=runner,
        runtime_states=states,
        input_buffers=_input_buffers(),
        vocab_size=VOCAB,
    )


def _eagle(states: RuntimeStates) -> Eagle:
    runner = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(get_hot_token_id=lambda: None),
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1), world_size=1),
        model_config=SimpleNamespace(requires_request_token_history=True),
    )
    return Eagle(
        spec_num_tokens=SPEC_TOKENS,
        spec_num_steps=3,
        draft_model_runner=runner,
        runtime_states=states,
        input_buffers=_input_buffers(),
        vocab_size=VOCAB,
    )


class _Leaf(AttentionBackend):
    """A stateless leaf: the composite default images nothing for it."""

    def __init__(self) -> None:
        self._init_pool_binding()


def _inkling() -> InklingAttnBackend:
    conv_pool = InklingConvStatePool(
        num_layers=2,
        num_slots=MAX_BS + 2,
        conv_dim=HIDDEN,
        ring_size=5,
        dtype=torch.bfloat16,
        device="cpu",
    )
    return InklingAttnBackend(_Leaf(), conv_pool)


def _kpool_pool() -> HybridGlm53FlashTokenToKVPool:
    pool = HybridGlm53FlashTokenToKVPool.__new__(HybridGlm53FlashTokenToKVPool)
    pool._kpool_tail_workspace = _KPoolTailWorkspace(
        options=None,
        storage=torch.zeros((2, 2, MAX_BS + 2, 3, HIDDEN), dtype=torch.bfloat16),
        row_by_layer={0: 0, 1: 1},
    )
    return pool


def _dsa(pool, *, is_draft: bool) -> DSABackend:
    backend = DSABackend.__new__(DSABackend)
    backend._init_pool_binding()
    backend.cache_pool = pool
    backend.is_draft = is_draft
    backend.kpool_runtime = object()
    backend._dense_backend = _Leaf()
    return backend


def _sampling_backend(cls) -> SamplingBackend:
    """A sampling backend over the test pool geometry, on the CPU."""
    config = SamplingBackendConfig(
        synthetic_acceptance_length=None,
        enable_speculative_sampling=cls is not GreedySamplingBackend,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=SPEC_TOKENS,
        max_req_pool_size=MAX_REQ_POOL_SIZE,
        vocab_size=VOCAB,
        device="cpu",
    )
    # The Triton backends size a scratch buffer by the SM count.
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        return cls(config)


def _sampling_params(rid: str, **overrides) -> SamplingParams:
    fields = dict(
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        min_p=0.1,
        frequency_penalty=0.5,
        presence_penalty=0.25,
        repetition_penalty=1.5,
    )
    fields.update(overrides)
    params = SamplingParams(**fields)
    params.resolve_seed(rid)
    params.normalize(None)
    return params


def _randomize(rows: list[torch.Tensor]) -> None:
    for row in rows:
        if row.dtype == torch.bool:
            row.copy_(torch.rand(row.shape) > 0.5)
        elif row.dtype.is_floating_point:
            row.copy_(torch.randn(row.shape).to(row.dtype))
        else:
            row.copy_(torch.randint(1, 100, row.shape).to(row.dtype))


RESTORED = "req-restored"


def _round_trip(
    exporter: SlotStateExporter,
    *,
    src_slot: int,
    dst_slot: int,
    rows_rewritten_on_import: frozenset[int] = frozenset(),
) -> None:
    """Export ``src_slot`` as ``RESTORED``'s, wipe it, import into ``dst_slot``;
    both must match except the rows an owner deliberately rewrites on import."""
    torch.manual_seed(0)
    source_rows = exporter.slot_state_rows(src_slot)
    _randomize(source_rows)
    expected = [row.clone() for row in source_rows]
    assert exporter.slot_state_bytes() == slot_state_image_bytes(source_rows)
    assert exporter.slot_state_bytes() % SLOT_STATE_ALIGNMENT == 0
    image = torch.full((exporter.slot_state_bytes(),), 0xEE, dtype=torch.uint8)

    exporter.export_slot_state(src_slot, image, None, request_id=RESTORED)
    for row in source_rows:
        row.zero_()
    exporter.import_slot_state(dst_slot, image, None, request_id=RESTORED)

    for index, (restored, original) in enumerate(
        zip(exporter.slot_state_rows(dst_slot), expected)
    ):
        assert restored.shape == original.shape
        if index not in rows_rewritten_on_import:
            assert torch.equal(restored, original)


# ----------------------------------------------------------------------
# The packing helpers
# ----------------------------------------------------------------------


def test_pack_aligns_every_row_and_refuses_a_short_image():
    rows = [
        torch.tensor(7, dtype=torch.int32),
        torch.ones(3, dtype=torch.bool),
        torch.arange(6, dtype=torch.float32).view(2, 3),
    ]
    nbytes = slot_state_image_bytes(rows)
    # 4 B, 3 B and 24 B, each padded to the 16 B alignment.
    assert (
        nbytes == SLOT_STATE_ALIGNMENT + SLOT_STATE_ALIGNMENT + 2 * SLOT_STATE_ALIGNMENT
    )
    image = torch.zeros(nbytes, dtype=torch.uint8)
    pack_slot_rows(rows, image, None)
    out = [torch.zeros_like(row) for row in rows]
    unpack_slot_rows(out, image, None)
    assert all(torch.equal(a, b) for a, b in zip(out, rows))
    with pytest.raises(ValueError, match="cannot hold"):
        pack_slot_rows(rows, image[:-1], None)
    with pytest.raises(ValueError, match="1-D uint8"):
        pack_slot_rows(rows, image.view(torch.int8), None)


def test_pack_copies_a_non_contiguous_slot_slice():
    ring = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).view(2, 4, 3)
    row = ring[:, 2]  # the slot dimension is not the leading one
    assert not row.is_contiguous()
    image = torch.zeros(slot_state_image_bytes([row]), dtype=torch.uint8)
    pack_slot_rows([row], image, None)
    target = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
    unpack_slot_rows([target[:, 1]], image, None)
    assert torch.equal(target[:, 1], ring[:, 2])
    assert target[:, 0].abs().sum() == 0


# ----------------------------------------------------------------------
# Round trips, one owner at a time
# ----------------------------------------------------------------------


@pytest.mark.parametrize("draft_probs", [False, True])
@pytest.mark.parametrize("trees", [False, True])
def test_runtime_states_round_trip(draft_probs, trees):
    states = _runtime_states(draft_probs=draft_probs, trees=trees, history=True)
    _round_trip(states, src_slot=3, dst_slot=7)
    # Token-derived rows stay out of the image and untouched by it.
    assert states.request_token_history_ids.abs().sum() == 0


@pytest.mark.parametrize("candidates_real", [True, False])
def test_runtime_states_image_whether_the_candidates_are_real(candidates_real):
    """A restored request's first decode op carries its token explicitly (no
    forward of its own is in flight), and the prologue verifies an explicit id
    single-token unless the row's candidates are marked real. The bit travels
    with the row as it stood: True after the victim's last drafter forward or
    a landing that shipped candidates, False for a PD decode role's victim
    landed without them -- whose stale candidate columns must not be verified."""
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    width = states.future_input_map.shape[1]
    states.future_input_map[3] = torch.arange(1, width + 1)
    if candidates_real:
        states.mark_spec_candidates_drafted(torch.tensor([3]))
    else:
        states.reset_states(torch.tensor([3]), torch.tensor([40], dtype=torch.int32))
    assert bool(states.spec_candidates_ready[3]) is candidates_real
    states.spec_candidates_ready[7] = not candidates_real  # the new slot's past
    image = torch.empty((states.slot_state_bytes(),), dtype=torch.uint8)
    states.export_slot_state(3, image, None, request_id="restored")
    states.import_slot_state(7, image, None, request_id="restored")
    assert bool(states.spec_candidates_ready[7]) is candidates_real
    assert torch.equal(states.future_input_map[7], states.future_input_map[3])


def test_mtp_stash_round_trip():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    _round_trip(_mtp(states), src_slot=1, dst_slot=5)


def test_eagle_history_frontier_round_trip():
    states = _runtime_states(draft_probs=False, trees=False, history=True)
    eagle = _eagle(states)
    assert eagle.draft_history_lengths_buf is not None
    _round_trip(eagle, src_slot=2, dst_slot=8)


def test_inkling_ring_round_trip_through_the_wrapper():
    backend = _inkling()
    _round_trip(backend, src_slot=3, dst_slot=9)
    # The wrapper images its own ring; its leaf is a separate (empty) node.
    assert backend.slot_state_exporters() == (backend, backend.child_backends()[0])


def test_dsa_kpool_tail_round_trip_once_per_arena():
    pool = _kpool_pool()
    target = _dsa(pool, is_draft=False)
    draft = _dsa(pool, is_draft=True)
    assert draft.slot_state_bytes() == 0, "the shared tail is imaged by the target"
    _round_trip(target, src_slot=2, dst_slot=5)


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS, ids=lambda cls: cls.__name__)
def test_sampling_backend_round_trip_continues_where_the_victim_stopped(cls):
    """A restored request samples as the unretracted run would: its scalars,
    penalty history, bias and coin generator land in the new slot, and the
    next prepare_step sees no rid flip there -- so nothing is re-initialised."""
    backend = _sampling_backend(cls)
    victim, params = "req-victim", _sampling_params("req-victim")
    params.logit_bias = {"3": 2.0, "7": -1.5}
    src_slot, dst_slot = 4, 8
    # The victim's steps so far: scattered scalars (and a generator) on its
    # first step, then some accumulated history.
    backend.prepare_step([victim], [src_slot], [params], num_tokens_per_req=2)
    rows = backend.slot_state_rows(src_slot)
    for row in rows:
        if row.ndim == 1 and row.dtype == torch.int32:  # the penalty counts
            row[::3] = 2
    expected = [row.clone() for row in rows]
    # The victim's own generator stays where its last step left it: it is
    # the oracle for what an unretracted next step would have drawn.
    generator = (
        backend._cpu_generator_per_slot[src_slot]
        if isinstance(backend, FlashInferSamplingBackend)
        else None
    )

    assert backend.slot_state_bytes() % SLOT_STATE_ALIGNMENT == 0
    image = torch.full((backend.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    backend.export_slot_state(src_slot, image, None, request_id=victim)
    for row in rows:
        row.zero_()
    backend.import_slot_state(dst_slot, image, None, request_id=victim)
    assert all(
        torch.equal(a, b) for a, b in zip(backend.slot_state_rows(dst_slot), expected)
    )

    # The restored slot is the victim's: the next step must not reset it.
    with patch.object(
        type(backend), "_reset_slot", side_effect=AssertionError("slot was reset")
    ):
        backend.prepare_step([victim], [dst_slot], [params], num_tokens_per_req=2)
    if cls is GreedySamplingBackend:
        assert backend.slot_state_bytes() == 0
        return
    assert backend._last_rid_per_slot[dst_slot] == victim
    assert all(
        torch.equal(a, b) for a, b in zip(backend.slot_state_rows(dst_slot), expected)
    )
    if generator is not None:
        # The coin stream continues from the victim's last step: the step
        # above refilled the restored slot's coins (n = 2, plus the final
        # coin) from the restored generator, and they are exactly what the
        # unretracted generator draws next.
        assert backend._cpu_generator_per_slot[dst_slot] is not generator
        lo = coin_eps(torch.float32)
        unretracted_coins = torch.empty((1, 2), dtype=torch.float32)
        unretracted_coins[0, :2].uniform_(lo, 1.0, generator=generator)
        unretracted_final = torch.empty((1,), dtype=torch.float32)
        unretracted_final[0].uniform_(lo, 1.0, generator=generator)
        assert torch.equal(backend._coins_buf[0, :2], unretracted_coins[0])
        assert torch.equal(backend._final_coins_buf[0], unretracted_final[0])


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS[1:], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("previous_occupant", [None, "req-previous"])
def test_sampling_backend_images_an_unprepared_slot_as_nobodys(cls, previous_occupant):
    """A PD decode role's request retracted between its landing and its first
    decode never went through prepare_step: the slot is either untouched or
    the previous occupant's (scalars, seed, coin generator). The image says
    so and carries no payload; the restore leaves the new slot unclaimed, and
    the request's first prepare_step resets it from its own SamplingParams,
    exactly as the unretracted first decode would have."""
    backend = _sampling_backend(cls)
    victim, params = "req-victim", _sampling_params("req-victim", temperature=0.3)
    src_slot, dst_slot = 4, 8
    if previous_occupant is not None:
        backend.prepare_step(
            [previous_occupant], [src_slot], [_sampling_params(previous_occupant)]
        )
    image = torch.full((backend.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    backend.export_slot_state(src_slot, image, None, request_id=victim)
    assert not read_prepared_marker(image)
    assert torch.all(image[PREPARED_MARKER_BYTES:] == 0xEE), "no payload imaged"

    # The new slot held someone else; the restore makes it nobody's.
    backend.prepare_step(["req-stale"], [dst_slot], [_sampling_params("req-stale")])
    backend.import_slot_state(dst_slot, image, None, request_id=victim)
    assert backend._last_rid_per_slot[dst_slot] is None
    with patch.object(type(backend), "_reset_slot", wraps=backend._reset_slot) as reset:
        backend.prepare_step([victim], [dst_slot], [params], num_tokens_per_req=2)
    reset.assert_called_once()
    assert reset.call_args.args[:2] == (dst_slot, params)
    assert backend._last_rid_per_slot[dst_slot] == victim
    assert float(backend._temperature_pool[dst_slot]) == pytest.approx(0.3)
    assert int(backend._seed_pool[dst_slot]) == int(params.seed)
    if isinstance(backend, FlashInferSamplingBackend):
        # The coin stream starts from the request's own seed: the step above
        # drew its coins from a generator seeded by the reset.
        expected = torch.Generator(device="cpu")
        expected.manual_seed(int(params.seed))
        lo = coin_eps(torch.float32)
        coins = torch.empty((1, 2), dtype=torch.float32)
        coins[0, :2].uniform_(lo, 1.0, generator=expected)
        assert torch.equal(backend._coins_buf[0, :2], coins[0])


def test_a_leaf_without_slot_state_images_nothing():
    leaf = _Leaf()
    assert leaf.slot_state_bytes() == 0
    leaf.export_slot_state(0, torch.zeros(0, dtype=torch.uint8), None, request_id="r")
    leaf.import_slot_state(0, torch.zeros(0, dtype=torch.uint8), None, request_id="r")


def test_model_executor_lists_every_owner_once_and_the_layout_is_fixed():
    states = _runtime_states(draft_probs=True, trees=False, history=False)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.model_slot_state_exporters = ()
    # A composite tree is flattened: the wrapper and its leaf are separate
    # segments, each imaging its own rows.
    executor.attn_backend = _inkling()
    leaf = executor.attn_backend.child_backends()[0]
    executor.drafter = _mtp(states)
    executor.sampling_backend = _sampling_backend(FlashInferFullSamplingBackend)
    # A shared draft tree is listed once; a distinct one adds its own nodes.
    executor.draft_attn_backend = executor.attn_backend
    assert executor.slot_state_exporters() == (
        states,
        executor.attn_backend,
        leaf,
        executor.drafter,
        executor.sampling_backend,
    )
    executor.draft_attn_backend = _inkling()
    exporters = executor.slot_state_exporters()
    assert exporters == (
        states,
        executor.attn_backend,
        leaf,
        executor.draft_attn_backend,
        executor.draft_attn_backend.child_backends()[0],
        executor.drafter,
        executor.sampling_backend,
    )
    # Slot 2 of the sampling backend must have been prepared to be imaged.
    executor.sampling_backend.prepare_step(["r2"], [2], [_sampling_params("r2")])

    # The layout measures every owner once at construction and only slices
    # afterwards: no owner is asked its size again on export or import.
    layout = SlotStateLayout(exporters)
    assert layout.nbytes == sum(owner.slot_state_bytes() for owner in exporters)
    assert [owner for owner, _, _ in layout.segments] == list(exporters)
    assert leaf.slot_state_bytes() == 0
    rows = lambda slot: [  # noqa: E731
        row for owner in exporters for row in owner.slot_state_rows(slot)
    ]
    torch.manual_seed(1)
    _randomize(rows(2))
    expected = [row.clone() for row in rows(2)]
    image = torch.zeros(layout.nbytes, dtype=torch.uint8)
    with patch.object(
        RuntimeStates, "slot_state_bytes", side_effect=AssertionError("re-measured")
    ):
        layout.export(2, image, None, request_id="r2")
        for row in rows(2):
            row.zero_()
        layout.import_(6, image, None, request_id="r6")
    assert all(torch.equal(a, b) for a, b in zip(rows(6), expected))
    with pytest.raises(ValueError, match="need"):
        layout.export(2, image[:-SLOT_STATE_ALIGNMENT], None, request_id="r2")

    class _Unpadded:
        def slot_state_bytes(self):
            return 3

    with pytest.raises(ValueError, match="not padded"):
        SlotStateLayout([_Unpadded()])


# ----------------------------------------------------------------------
# Completeness: a per-slot buffer must be exported or declared token-derived.
# ----------------------------------------------------------------------


def _is_runtime_component(value) -> bool:
    """A helper object of the runtime an owner keeps its tensors in (a conv
    pool, a tail workspace) -- not a tensor, not another exporter, not a test
    fake (``SimpleNamespace``) and not a module."""
    return (
        not isinstance(value, (torch.Tensor, torch.nn.Module, SlotStateExporter))
        and hasattr(value, "__dict__")
        and type(value).__module__.startswith("tokenspeed.")
    )


def _tensor_attributes(obj) -> dict[str, torch.Tensor]:
    """``name -> tensor`` over the object's own attributes and, one level down,
    the runtime components it holds (``component.tensor``)."""
    found: dict[str, torch.Tensor] = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            found[name] = value
        elif _is_runtime_component(value):
            for sub_name, sub_value in vars(value).items():
                if isinstance(sub_value, torch.Tensor):
                    found[f"{name}.{sub_name}"] = sub_value
    return found


def _per_slot_tensors(obj, slot_domains: set[int]) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in _tensor_attributes(obj).items()
        if any(dim in slot_domains for dim in tensor.shape)
    }


def _storage_ids(rows) -> set[int]:
    return {row.untyped_storage().data_ptr() for row in rows}


def _assert_slot_state_complete(owner, *, exporter, slot_domains: set[int]) -> None:
    """Every slot-sized tensor of ``owner`` (and its components) is exported by
    ``exporter`` or declared by the exporter's class; the declarations are
    required attributes, empty or not."""
    exported = _storage_ids(exporter.slot_state_rows(0))
    declared = set(type(exporter).token_derived_slot_state) | set(
        type(exporter).constant_slot_state
    )
    uncovered = sorted(
        name
        for name, tensor in _per_slot_tensors(owner, slot_domains).items()
        if tensor.untyped_storage().data_ptr() not in exported and name not in declared
    )
    assert not uncovered, (
        f"{type(owner).__name__} keeps per-slot state with no exporter: {uncovered}. "
        "List it in slot_state_rows (restored byte for byte), in "
        "token_derived_slot_state (reseeded from the request's tokens) or in "
        "constant_slot_state (identical for every slot)."
    )


def test_every_owner_a_constructed_executor_lists_is_complete():
    """The owners as the model executor lists them for the blob, each checked
    against its own tensors: a per-slot buffer added to any of them without an
    exporter fails here."""
    states = _runtime_states(draft_probs=True, trees=True, history=True)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.model_slot_state_exporters = ()
    executor.attn_backend = _inkling()
    executor.draft_attn_backend = _inkling()
    executor.drafter = _eagle(states)
    executor.sampling_backend = _sampling_backend(FlashInferFullSamplingBackend)
    exporters = executor.slot_state_exporters()
    assert len(exporters) == 7
    for owner in exporters:
        assert isinstance(owner, SlotStateExporter)
        _assert_slot_state_complete(owner, exporter=owner, slot_domains={POOL_ROWS})


def test_runtime_states_slot_state_is_complete():
    states = _runtime_states(draft_probs=True, trees=True, history=True)
    _assert_slot_state_complete(states, exporter=states, slot_domains={POOL_ROWS})
    # The declared token-derived rows exist and really are per-slot.
    per_slot = _per_slot_tensors(states, {POOL_ROWS})
    assert set(RuntimeStates.token_derived_slot_state) <= set(per_slot)


def test_drafter_slot_state_is_complete():
    states = _runtime_states(draft_probs=False, trees=False, history=True)
    mtp = _mtp(states)
    _assert_slot_state_complete(mtp, exporter=mtp, slot_domains={POOL_ROWS})
    eagle = _eagle(states)
    _assert_slot_state_complete(eagle, exporter=eagle, slot_domains={POOL_ROWS})


def test_backend_slot_state_is_complete():
    inkling = _inkling()
    _assert_slot_state_complete(inkling, exporter=inkling, slot_domains={POOL_ROWS})
    pool = _kpool_pool()
    target = _dsa(pool, is_draft=False)
    _assert_slot_state_complete(pool, exporter=target, slot_domains={POOL_ROWS})


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS, ids=lambda cls: cls.__name__)
def test_sampling_backend_slot_state_is_complete(cls):
    backend = _sampling_backend(cls)
    _assert_slot_state_complete(backend, exporter=backend, slot_domains={POOL_ROWS})
    # The declared constant pools exist, are per-slot, and really are constant.
    per_slot = _per_slot_tensors(backend, {POOL_ROWS})
    for name in SamplingBackend.constant_slot_state:
        assert name in per_slot
        assert per_slot[name].abs().sum() == 0
    # Every per-slot thing the rid-flip reset writes is in the image.
    if cls is not GreedySamplingBackend:
        exported = _storage_ids(backend.slot_state_rows(0))
        written = []
        for name, tensor in per_slot.items():
            if name in SamplingBackend.constant_slot_state:
                continue
            if name in SamplingBackend.token_derived_slot_state:
                # Derived scratch is neither restored nor changed by admission.
                tensor[2].fill_(197)
                before = tensor.clone()
                backend._reset_slot(2, _sampling_params("r2"))
                assert torch.equal(tensor, before)
                continue
            tensor.fill_(0)
            backend._reset_slot(2, _sampling_params("r2"))
            if tensor[2].abs().sum() != 0:
                written.append(name)
            assert tensor.untyped_storage().data_ptr() in exported, name
        assert "_temperature_pool" in written and "_seed_pool" in written


def test_completeness_check_catches_an_unexported_buffer():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    states.rogue_per_slot = torch.zeros(POOL_ROWS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="rogue_per_slot"):
        _assert_slot_state_complete(states, exporter=states, slot_domains={POOL_ROWS})
    backend = _sampling_backend(TritonFullSamplingBackend)
    backend._rogue_pool = torch.zeros(POOL_ROWS, VOCAB, dtype=torch.int32)
    with pytest.raises(AssertionError, match="_rogue_pool"):
        _assert_slot_state_complete(backend, exporter=backend, slot_domains={POOL_ROWS})


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS[1:], ids=lambda cls: cls.__name__)
def test_sampling_offsets_recompute_after_restore_to_another_slot(cls):
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    backend = _sampling_backend(cls)
    rid = "offset-restore"
    backend.prepare_step([rid], [4], [_sampling_params(rid)], num_tokens_per_req=2)
    frontier = torch.zeros(POOL_ROWS, dtype=torch.int32)
    frontier[4] = 130
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([4]),
        valid_cache_lengths=frontier,
        extend_lengths=torch.tensor([895], dtype=torch.int32),
    )
    assert backend._prepare_sampling_offsets(info)[4].item() == 1025
    image = torch.empty(backend.slot_state_bytes(), dtype=torch.uint8)
    backend.export_slot_state(4, image, None, request_id=rid)
    backend._sampling_offsets_pool[8] = -99
    backend.import_slot_state(8, image, None, request_id=rid)
    # RuntimeStates restores the accepted frontier separately. A destination's
    # stale scratch must not affect the position regenerated for the next draw.
    frontier[8] = frontier[4]
    restored = SamplingBatchInfo(
        req_pool_indices=torch.tensor([8]),
        valid_cache_lengths=frontier,
        extend_lengths=torch.tensor([895], dtype=torch.int32),
    )
    assert backend._prepare_sampling_offsets(restored)[8].item() == 1025
    assert frontier[8].item() == 130
    frontier[8] = 1025
    restored.extend_lengths = None
    assert backend._prepare_sampling_offsets(restored)[8].item() == 1026


# ----------------------------------------------------------------------
# Preservation: a non-Whisper executor's blob is byte-identical to the one before
# Whisper's cross KV became slot state (expected layouts observed on that code).
# ----------------------------------------------------------------------


def _mtp_executor() -> ModelExecutor:
    """Target Inkling tree, MTP drafter sharing it, FlashInfer-full sampling."""
    states = _runtime_states(draft_probs=True, trees=False, history=False)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.model_slot_state_exporters = ()
    executor.attn_backend = _inkling()
    executor.draft_attn_backend = executor.attn_backend
    executor.drafter = _mtp(states)
    executor.sampling_backend = _sampling_backend(FlashInferFullSamplingBackend)
    return executor


def _eagle_executor() -> ModelExecutor:
    """Distinct target and draft Inkling trees, Eagle drafter, Triton-full sampling."""
    states = _runtime_states(draft_probs=True, trees=True, history=True)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.model_slot_state_exporters = ()
    executor.attn_backend = _inkling()
    executor.draft_attn_backend = _inkling()
    executor.drafter = _eagle(states)
    executor.sampling_backend = _sampling_backend(TritonFullSamplingBackend)
    return executor


def _plain_executor() -> ModelExecutor:
    """A dense LLM: stateless leaf, no drafter, greedy sampling."""
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.model_slot_state_exporters = ()
    executor.attn_backend = _Leaf()
    executor.draft_attn_backend = None
    executor.drafter = None
    executor.sampling_backend = _sampling_backend(GreedySamplingBackend)
    return executor


def _blob_record(executor: ModelExecutor) -> tuple[list, int, str]:
    """The executor's blob layout and the sha256 of one seeded slot image."""
    import hashlib

    exporters = executor.slot_state_exporters()
    layout = SlotStateLayout(exporters)
    segments = [
        (type(owner).__name__, begin, end) for owner, begin, end in layout.segments
    ]
    executor.sampling_backend.prepare_step(["r2"], [2], [_sampling_params("r2")])
    torch.manual_seed(1)
    _randomize([row for owner in exporters for row in owner.slot_state_rows(2)])
    image = torch.zeros(layout.nbytes, dtype=torch.uint8)
    layout.export(2, image, None, request_id="r2")
    digest = hashlib.sha256(image.numpy().tobytes()).hexdigest()
    return segments, layout.nbytes, digest


# (segments, nbytes, sha256 of the slot-2 image) per executor, recorded on F.
_RECORDED_BLOBS = {
    "mtp": (
        [
            ("RuntimeStates", 0, 656),
            ("InklingAttnBackend", 656, 832),
            ("_Leaf", 832, 832),
            ("Mtp", 832, 896),
            ("FlashInferFullSamplingBackend", 896, 6288),
        ],
        6288,
        "e1687bc19b77fe34cdccaf276813a737c4f30f0075e210dbef2497c5a6b4347f",
    ),
    "eagle": (
        [
            ("RuntimeStates", 0, 672),
            ("InklingAttnBackend", 672, 848),
            ("_Leaf", 848, 848),
            ("InklingAttnBackend", 848, 1024),
            ("_Leaf", 1024, 1024),
            ("Eagle", 1024, 1040),
            ("TritonFullSamplingBackend", 1040, 1376),
        ],
        1376,
        "76da900930377f7934f303342aa00b2447a20e71c3b0e9511395acb878ece62e",
    ),
    "plain": (
        [
            ("RuntimeStates", 0, 144),
            ("_Leaf", 144, 144),
            ("GreedySamplingBackend", 144, 144),
        ],
        144,
        "82394803c62847d6ead5831577204a0001c3e032c2cefc848510600b20cc2436",
    ),
}
_EXECUTORS = {"mtp": _mtp_executor, "eagle": _eagle_executor, "plain": _plain_executor}


@pytest.mark.parametrize("name", sorted(_EXECUTORS))
def test_non_whisper_executor_blob_is_byte_identical_to_the_pre_fix_one(name):
    """**Validates: Requirements 3.4, 3.5** (Property 2, slot-state layout)."""
    segments, nbytes, digest = _blob_record(_EXECUTORS[name]())
    print(f"{name}: segments={segments} nbytes={nbytes} sha256={digest}")
    assert (segments, nbytes, digest) == _RECORDED_BLOBS[name]


def test_model_owned_exporters_follow_the_runtime_owners_in_the_blob():
    """**Validates: Requirements 3.4, 3.7.** A model's own owners (Whisper's
    cross KV) are appended after the runtime's: the runtime segments keep the
    offsets a model without slot state has, and the model's rows round-trip
    through the executor's layout."""
    from tokenspeed.runtime.models.whisper import CrossAttentionCache

    cross = CrossAttentionCache(
        num_slots=POOL_ROWS,
        num_layers=2,
        encoder_len=6,
        kv_heads=2,
        head_dim=4,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    plain_segments, plain_nbytes, _ = _RECORDED_BLOBS["plain"]
    executor = _plain_executor()
    executor.model_slot_state_exporters = (cross,)
    exporters = executor.slot_state_exporters()
    assert exporters[-1] is cross
    layout = SlotStateLayout(exporters)
    segments = [
        (type(owner).__name__, begin, end) for owner, begin, end in layout.segments
    ]
    assert segments[:-1] == plain_segments
    assert segments[-1] == (
        "CrossAttentionCache",
        plain_nbytes,
        plain_nbytes + cross.slot_state_bytes(),
    )

    torch.manual_seed(3)
    _randomize(cross.slot_state_rows(2))
    expected = [row.clone() for row in cross.slot_state_rows(2)]
    cross.claim_slot(2, "r2")
    image = torch.zeros(layout.nbytes, dtype=torch.uint8)
    layout.export(2, image, None, request_id="r2")
    layout.import_(7, image, None, request_id="r2")
    assert all(torch.equal(a, b) for a, b in zip(cross.slot_state_rows(7), expected))
    assert cross.owner_of(7) == "r2"


# ----------------------------------------------------------------------
# Whisper's cross-attention KV: a model-owned, request-keyed owner.
# ----------------------------------------------------------------------

# Small enough to stay off the slot domain: padded_len 8 (2 pages of 4),
# 12 KV boundaries, 2 layers, 2 heads of 4.
CROSS_ENCODER_LEN = 6


def _cross_cache():
    from tokenspeed.runtime.models.whisper import CrossAttentionCache

    return CrossAttentionCache(
        num_slots=POOL_ROWS,
        num_layers=2,
        encoder_len=CROSS_ENCODER_LEN,
        kv_heads=2,
        head_dim=4,
        page_size=4,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def _whisper_executor(cross) -> ModelExecutor:
    """A Whisper deployment's executor: plain runtime owners plus the model's
    cross-attention cache, listed through the generic model hook."""
    executor = _plain_executor()
    executor.model_slot_state_exporters = (cross,)
    return executor


def test_cross_attention_cache_slot_state_is_complete():
    """**Validates: Requirement 3.7.** Every slot-sized tensor of the cross
    cache is exported (K, V) or declared: the page table is rewritten by every
    forward, the lengths and page offsets are the same for every slot."""
    cross = _cross_cache()
    _assert_slot_state_complete(cross, exporter=cross, slot_domains={POOL_ROWS})
    per_slot = _per_slot_tensors(cross, {POOL_ROWS})
    assert {"k", "v", "page_table", "cache_seqlens"} <= set(per_slot)
    # The declared constant rows really are identical for every slot.
    for name in type(cross).constant_slot_state:
        tensor = vars(cross)[name]
        if name in per_slot:
            assert torch.all(tensor == tensor[0]), name
    # A slot-sized buffer added without an exporter is caught.
    cross.rogue_per_slot = torch.zeros(POOL_ROWS, 3, dtype=torch.int32)
    with pytest.raises(AssertionError, match="rogue_per_slot"):
        _assert_slot_state_complete(cross, exporter=cross, slot_domains={POOL_ROWS})


def test_every_owner_a_whisper_executor_lists_is_complete():
    """**Validates: Requirements 3.4, 3.7.** The owners a Whisper executor
    lists for the blob -- the runtime's and the model's cross KV -- each cover
    every slot-sized tensor they hold."""
    cross = _cross_cache()
    exporters = _whisper_executor(cross).slot_state_exporters()
    assert exporters[-1] is cross
    for owner in exporters:
        assert isinstance(owner, SlotStateExporter)
        _assert_slot_state_complete(owner, exporter=owner, slot_domains={POOL_ROWS})


def test_cross_attention_cache_round_trip_restores_rows_and_claims_the_slot():
    """**Validates: Requirement 3.7.** A slot prepared for the victim images
    the marker and its encoder rows; the restore into another slot writes the
    rows there and claims it for the victim. Nothing else moves."""
    cross = _cross_cache()
    torch.manual_seed(5)
    _randomize([cross.k, cross.v])
    cross.claim_slot(3, RESTORED)
    cross.claim_slot(9, "req-stale")
    expected = [row.clone() for row in cross.slot_state_rows(3)]
    k_before, v_before = cross.k.clone(), cross.v.clone()

    image = torch.full((cross.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    cross.export_slot_state(3, image, None, request_id=RESTORED)
    assert read_prepared_marker(image)
    for row in cross.slot_state_rows(3):
        row.zero_()
    cross.import_slot_state(9, image, None, request_id=RESTORED)

    assert all(torch.equal(a, b) for a, b in zip(cross.slot_state_rows(9), expected))
    assert cross.owner_of(9) == RESTORED
    # Only slot 9's written rows changed (slot 3 was wiped by the test): the
    # padding rows and every other slot keep their bytes.
    others = [s for s in range(POOL_ROWS) if s not in (3, 9)]
    for now, before in ((cross.k, k_before), (cross.v, v_before)):
        assert torch.equal(now[:, others], before[:, others])
        assert torch.equal(
            now[:, 9, CROSS_ENCODER_LEN:], before[:, 9, CROSS_ENCODER_LEN:]
        )


@pytest.mark.parametrize("previous_occupant", [None, "req-previous"])
def test_cross_attention_cache_images_an_unowned_slot_as_nobodys(previous_occupant):
    """**Validates: Requirement 3.7.** A victim retracted before its audio was
    encoded into its slot (the slot is nobody's, or a previous occupant's)
    images the marker alone; the restore copies nothing and leaves the new
    slot unowned, so the request's next extend encodes its own audio."""
    cross = _cross_cache()
    torch.manual_seed(6)
    _randomize([cross.k, cross.v])
    if previous_occupant is not None:
        cross.claim_slot(3, previous_occupant)
    cross.claim_slot(9, "req-stale")
    dst_rows = [row.clone() for row in cross.slot_state_rows(9)]

    image = torch.full((cross.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    cross.export_slot_state(3, image, None, request_id="req-victim")
    assert not read_prepared_marker(image)
    assert torch.all(image[PREPARED_MARKER_BYTES:] == 0xEE), "no payload imaged"
    cross.import_slot_state(9, image, None, request_id="req-victim")

    assert cross.owner_of(9) is None
    assert all(torch.equal(a, b) for a, b in zip(cross.slot_state_rows(9), dst_rows))
    assert cross.owner_of(3) == previous_occupant, "export does not change ownership"


def _audio_context(requests: list[tuple[str, int]], prefix_lens: list[int]):
    """An extend's multimodal context: one audio window per ``(rid, slot)``."""
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
                        modality=Modality.AUDIO,
                        feature=torch.full((2, 4), float(index)),
                    )
                ]
            )
            for index in range(len(requests))
        ],
        request_ids=[rid for rid, _ in requests],
        request_pool_indices=[slot for _, slot in requests],
        extend_prefix_lens=list(prefix_lens),
        extend_seq_lens=[1] * len(requests),
        max_encoder_tokens=4096,
    )


def _to_encode(cross, requests, prefix_lens) -> list[tuple[str, int]]:
    """The ``(rid, slot)`` pairs the model's extend would run the encoder for."""
    from tokenspeed.runtime.models.whisper import _audio_features_to_encode

    features, slots, rids = _audio_features_to_encode(
        _audio_context(requests, prefix_lens), cross
    )
    assert (features is None) == (not slots)
    if features is not None:
        assert len(features) == len(slots)
    return list(zip(rids, slots))


def test_cross_attention_encodes_a_request_iff_its_slot_is_not_its_own():
    """**Validates: Requirement 3.7.** Ownership, not the prefix length,
    decides whether an extend runs the encoder: a fresh request encodes; a
    continuation of the slot's owner skips it at any prefix; a new request
    in a slot a finished one encoded re-encodes even with a prefix hit (no
    finish event releases the slot -- the different request id is the
    signal); a restored image claims its slot, a not-prepared one leaves it
    for the next extend to encode."""
    cross = _cross_cache()
    # Fresh admits encode.
    assert _to_encode(cross, [("a", 2), ("b", 5)], [0, 0]) == [("a", 2), ("b", 5)]
    for rid, slot in (("a", 2), ("b", 5)):
        cross.claim_slot(slot, rid)  # what encode_audio records
    # Their continuations (chunked prefill, recompute into the same slot) do not.
    assert _to_encode(cross, [("a", 2), ("b", 5)], [4, 0]) == []
    # "a" finished; "c" lands in slot 2 with a prefix hit: it does not own it.
    assert _to_encode(cross, [("c", 2), ("b", 5)], [3, 7]) == [("c", 2)]
    cross.claim_slot(2, "c")
    # "b" is resumed by recompute into slot 2, now "c"'s: it encodes again.
    assert _to_encode(cross, [("b", 2)], [6]) == [("b", 2)]

    # A snapshot of "b" (slot 5) restored into slot 7 claims slot 7 for "b".
    image = torch.empty((cross.slot_state_bytes(),), dtype=torch.uint8)
    cross.export_slot_state(5, image, None, request_id="b")
    cross.import_slot_state(7, image, None, request_id="b")
    assert _to_encode(cross, [("b", 7)], [6]) == []
    # A victim whose audio never ran here restores as nobody's and encodes.
    cross.export_slot_state(4, image, None, request_id="d")
    cross.import_slot_state(8, image, None, request_id="d")
    assert _to_encode(cross, [("d", 8)], [2]) == [("d", 8)]

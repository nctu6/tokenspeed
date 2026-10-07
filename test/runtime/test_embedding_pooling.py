"""Tests for the embedding (pooling) serving path.

CPU-only: no engine, no GPU. They cover the two places where a pooling bug
produces a *well-formed but wrong* answer, which is the failure mode nothing
downstream can catch:

  1. the reduction itself (LAST/CLS/MEAN over a packed batch), and
  2. the decision of whether this server pools at all, and how.

The checkpoint-detection tests use fixture directories rather than a real
model, so they run anywhere.
"""

import json
import os
import tempfile
import unittest

import torch

from tokenspeed.runtime.configs.model_config import (
    PoolingConfig,
    read_sentence_transformers_pooling,
    resolve_pooling_config,
)
from tokenspeed.runtime.layers.pooler import (
    Pooler,
    PoolingType,
    pool_hidden_states,
    seq_lens_from_gather_ids,
)


def _packed_batch(lengths, hidden=6, seed=0):
    torch.manual_seed(seed)
    lens = torch.tensor(lengths)
    states = torch.randn(int(lens.sum()), hidden)
    gather_ids = torch.cumsum(lens, dim=0) - 1
    return lens, states, gather_ids


class PoolReductionTest(unittest.TestCase):
    LENGTHS = [3, 1, 4, 2]

    def test_last_picks_each_sequence_final_row(self):
        lens, states, gather_ids = _packed_batch(self.LENGTHS)
        pooled = pool_hidden_states(PoolingType.LAST, states, gather_ids=gather_ids)
        expected = torch.stack([states[2], states[3], states[7], states[9]])
        self.assertTrue(torch.equal(pooled, expected))

    def test_cls_picks_each_sequence_first_row(self):
        lens, states, _ = _packed_batch(self.LENGTHS)
        pooled = pool_hidden_states(PoolingType.CLS, states, extend_seq_lens=lens)
        expected = torch.stack([states[0], states[3], states[4], states[8]])
        self.assertTrue(torch.equal(pooled, expected))

    def test_mean_averages_within_sequence_boundaries(self):
        lens, states, _ = _packed_batch(self.LENGTHS)
        pooled = pool_hidden_states(PoolingType.MEAN, states, extend_seq_lens=lens)
        expected = torch.stack(
            [
                states[0:3].mean(0),
                states[3:4].mean(0),
                states[4:8].mean(0),
                states[8:10].mean(0),
            ]
        )
        self.assertTrue(torch.allclose(pooled, expected, atol=1e-5))

    def test_missing_index_argument_raises_rather_than_defaulting(self):
        _, states, gather_ids = _packed_batch(self.LENGTHS)
        with self.assertRaises(ValueError):
            pool_hidden_states(PoolingType.LAST, states)
        with self.assertRaises(ValueError):
            pool_hidden_states(PoolingType.MEAN, states, gather_ids=gather_ids)

    def test_seq_lens_round_trip_through_gather_ids(self):
        lens, _, gather_ids = _packed_batch(self.LENGTHS)
        self.assertTrue(torch.equal(seq_lens_from_gather_ids(gather_ids), lens))

    def test_normalize_produces_unit_vectors(self):
        _, states, gather_ids = _packed_batch(self.LENGTHS)
        out = Pooler(PoolingType.LAST, normalize=True)(states, gather_ids=gather_ids)
        norms = out.embeddings.float().norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_no_normalize_leaves_magnitude_alone(self):
        _, states, gather_ids = _packed_batch(self.LENGTHS)
        out = Pooler(PoolingType.LAST, normalize=False)(states, gather_ids=gather_ids)
        self.assertTrue(torch.equal(out.embeddings, states.index_select(0, gather_ids)))


class _ServerArgsStub:
    def __init__(self, is_embedding=None, pooling_type=None, pooling_normalize=None):
        self.is_embedding = is_embedding
        self.pooling_type = pooling_type
        self.pooling_normalize = pooling_normalize


def _checkpoint(pooling: dict | None, normalize_module: bool = False) -> str:
    path = tempfile.mkdtemp()
    if pooling is not None:
        os.makedirs(os.path.join(path, "1_Pooling"))
        with open(os.path.join(path, "1_Pooling", "config.json"), "w") as handle:
            json.dump(pooling, handle)
    modules = [{"idx": 0, "type": "sentence_transformers.models.Transformer"}]
    if normalize_module:
        modules.append({"idx": 1, "type": "sentence_transformers.models.Normalize"})
    with open(os.path.join(path, "modules.json"), "w") as handle:
        json.dump(modules, handle)
    return path


class PoolingConfigDetectionTest(unittest.TestCase):
    def test_generative_checkpoint_is_not_a_pooling_server(self):
        path = _checkpoint(None)
        self.assertIsNone(read_sentence_transformers_pooling(path))
        self.assertIsNone(resolve_pooling_config(path, _ServerArgsStub()))

    def test_lasttoken_plus_normalize_module_is_read_off_the_checkpoint(self):
        path = _checkpoint({"pooling_mode_lasttoken": True}, normalize_module=True)
        self.assertEqual(
            read_sentence_transformers_pooling(path),
            PoolingConfig(pooling_type=PoolingType.LAST, normalize=True),
        )

    def test_normalize_is_read_from_modules_not_assumed(self):
        path = _checkpoint({"pooling_mode_cls_token": True}, normalize_module=False)
        config = read_sentence_transformers_pooling(path)
        self.assertEqual(config.pooling_type, PoolingType.CLS)
        self.assertFalse(config.normalize)

    def test_unsupported_pooling_mode_is_refused_not_silently_mapped(self):
        path = _checkpoint({"pooling_mode_weightedmean_tokens": True})
        with self.assertRaises(ValueError):
            read_sentence_transformers_pooling(path)

    def test_ambiguous_pooling_config_is_refused(self):
        path = _checkpoint(
            {"pooling_mode_lasttoken": True, "pooling_mode_mean_tokens": True}
        )
        with self.assertRaises(ValueError):
            read_sentence_transformers_pooling(path)

    def test_explicit_off_beats_a_pooling_checkpoint(self):
        path = _checkpoint({"pooling_mode_lasttoken": True})
        self.assertIsNone(
            resolve_pooling_config(path, _ServerArgsStub(is_embedding=False))
        )

    def test_forcing_embedding_without_a_pooling_config_is_refused(self):
        path = _checkpoint(None)
        with self.assertRaises(ValueError):
            resolve_pooling_config(path, _ServerArgsStub(is_embedding=True))

    def test_overrides_win_over_the_checkpoint(self):
        path = _checkpoint({"pooling_mode_lasttoken": True}, normalize_module=True)
        config = resolve_pooling_config(
            path,
            _ServerArgsStub(pooling_type="mean", pooling_normalize=False),
        )
        self.assertEqual(
            config, PoolingConfig(pooling_type=PoolingType.MEAN, normalize=False)
        )

    def test_override_supplies_the_mode_a_bare_checkpoint_lacks(self):
        path = _checkpoint(None)
        config = resolve_pooling_config(
            path, _ServerArgsStub(is_embedding=True, pooling_type="cls")
        )
        self.assertEqual(
            config, PoolingConfig(pooling_type=PoolingType.CLS, normalize=False)
        )


if __name__ == "__main__":
    unittest.main()

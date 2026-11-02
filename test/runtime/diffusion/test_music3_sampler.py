# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""CPU: the counter-hash Gumbel sampler is batch-invariant and samples top-k softmax."""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.diffusion.music3_ar import (  # noqa: E402
    AR_TOP_K,
    _sample_top_k,
    column_keys,
    hash_gumbel,
)


def test_noise_depends_only_on_seed_counter_and_column():
    columns = column_keys(300, "cpu")
    seeds = torch.tensor([7, 9, 7], dtype=torch.int64)
    counters = torch.tensor([3, 3, 3], dtype=torch.int64)
    noise = hash_gumbel(seeds, counters, columns)
    assert torch.equal(noise[0], noise[2])  # same key -> same noise
    assert not torch.equal(noise[0], noise[1])
    # A row's noise does not depend on the rest of the batch.
    alone = hash_gumbel(seeds[1:2], counters[1:2], columns)
    assert torch.equal(alone[0], noise[1])
    assert not torch.equal(
        hash_gumbel(seeds[:1], counters[:1] + 1, columns)[0], noise[0]
    )


def test_samples_follow_the_top_k_softmax():
    torch.manual_seed(0)
    vocab = 200
    logits = torch.randn(vocab) * 2.0
    # The Diffusers sampler's distribution: softmax over the top-50 logits.
    top = torch.topk(logits, AR_TOP_K)
    probs = torch.zeros(vocab)
    probs[top.indices] = torch.softmax(top.values, dim=0)

    draws = 60_000
    counters = torch.arange(draws, dtype=torch.int64)
    seeds = torch.full((draws,), 11, dtype=torch.int64)
    samples = _sample_top_k(
        logits.expand(draws, vocab), seeds, counters, column_keys(vocab, "cpu")
    )
    counts = torch.bincount(samples, minlength=vocab).double()
    assert counts[probs == 0].sum() == 0  # never outside the top-k
    expected = probs.double() * draws
    keep = expected > 20
    chi2 = (((counts - expected) ** 2) / expected)[keep].sum().item()
    dof = int(keep.sum()) - 1
    # Mean dof, sd sqrt(2 dof): 6 sigma is far outside chance.
    assert chi2 < dof + 6 * (2 * dof) ** 0.5

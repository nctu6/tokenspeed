"""Block verification (Sun et al., 2024) of a DSpark draft chain.

Instead of testing each draft token against the target on its own, the whole
block is accepted jointly: with ``p_i = min(1, p_{i-1} * P(x_i) / Q(x_i))`` and
``h_i = R_i / (R_i + 1 - p_i)`` where ``R_i = sum_x max(p_i * P(x | x_<=i) -
Q(x | x_<=i), 0)``, the longest prefix whose own uniform ``u_i <= h_i`` is
accepted, and the next token is drawn from ``max(p * P - Q, 0)`` (``P`` after a
full block). It is lossless and accepts at least as many tokens as the
token-wise test. Reference: vLLM ``rejection_sampler_utils.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

# Verify on each rank's vocab shard and exchange per-request scalars (two small
# all-gathers) instead of all-gathering full-vocab FP32 draft logits.
_BLOCK_VERIFY_TP = os.environ.get("TOKENSPEED_DSPARK_BLOCK_VERIFY_TP", "0") == "1"


@dataclass
class DSparkBlockVerifyState:
    """Draft-side tensors the verifier needs to rebuild every step's Q."""

    # [pool_slots, steps, local_vocab] base logits of the last drafted block.
    pool: torch.Tensor
    embedding: torch.Tensor  # [vocab, rank] replicated Markov bigram table
    projection: torch.Tensor  # [local_vocab, rank] this rank's shard
    num_valid: int  # real columns of this rank's shard
    tp_size: int
    tp_group: object
    vocab_start: int = 0  # first vocab id of this rank's shard


def draft_logits(
    state: DSparkBlockVerifyState, slots: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Full-vocab FP32 draft logits ``[bs, steps, tp * local_vocab]``.

    Step ``s`` conditions on ``candidates[:, s]`` (the anchor, then draft
    ``s - 1``): base logits plus the Markov bigram bias of that token.
    """
    base = state.pool.index_select(0, slots)  # [bs, steps, local]
    bs, steps, local = base.shape
    prev = candidates[:, :steps].long().clamp_(0, state.embedding.shape[0] - 1)
    bigram = state.embedding[prev].float()  # [bs, steps, rank]
    bias = bigram @ state.projection.float().T  # [bs, steps, local]
    mine = base + bias
    mine[..., state.num_valid :] = float("-inf")
    if state.tp_size == 1:
        return mine
    from tokenspeed.runtime.distributed.comm_ops import all_gather_single

    out = torch.empty(
        (state.tp_size, bs, steps, local), dtype=mine.dtype, device=mine.device
    )
    all_gather_single(out.view(-1), mine.contiguous().view(-1), state.tp_group)
    return out.permute(1, 2, 0, 3).reshape(bs, steps, state.tp_size * local)


def _local_draft_logits(
    state: DSparkBlockVerifyState, slots: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """This rank's ``[bs, steps, local_vocab]`` FP32 draft logits (padding -inf)."""
    base = state.pool.index_select(0, slots)
    steps = base.shape[1]
    prev = candidates[:, :steps].long().clamp_(0, state.embedding.shape[0] - 1)
    mine = base + state.embedding[prev].float() @ state.projection.float().T
    mine[..., state.num_valid :] = float("-inf")
    return mine


def _gather_small(state: DSparkBlockVerifyState, x: torch.Tensor) -> torch.Tensor:
    """All-gather a small FP32 tensor across TP ranks: ``[tp, *x.shape]``."""
    from tokenspeed.runtime.distributed.comm_ops import all_gather_single

    out = torch.empty((state.tp_size, *x.shape), dtype=x.dtype, device=x.device)
    all_gather_single(out.view(-1), x.contiguous().view(-1), state.tp_group)
    return out


def block_verify_chain_tp(
    state: DSparkBlockVerifyState,
    slots: torch.Tensor,
    target_logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
) -> None:
    """``block_verify_chain`` computed on this rank's vocab shard.

    Same outputs and distribution as gathering the full draft logits, but the
    only cross-rank traffic is two all-gathers of a few floats per request:
    (1) the shard's draft log-normalizer and the draft-token logit it owns;
    (2) per stopping position, the shard's residual mass and its Gumbel-max
    candidates from the residual and from ``P``. The next token for every
    possible accept length is preselected, so the second exchange already
    carries the winner. Gumbel noise is drawn over the full vocab and sliced,
    so shards never reuse noise.
    """
    bs, n = candidates.shape
    steps = n - 1
    vocab = target_logits.shape[-1]
    local = state.pool.shape[-1]
    lo = state.vocab_start
    w = max(0, min(local, vocab - lo))
    t = temperature.view(bs, 1, 1).clamp_min(1.0e-6)
    tl = target_logits.view(bs, n, vocab)
    lse_p = torch.logsumexp(tl / t, dim=-1)  # [bs, n]
    drafts = candidates[:, 1:].long()
    lp_x = (
        tl[:, :steps].gather(-1, drafts[..., None])[..., 0] / t[..., 0]
        - lse_p[:, :steps]
    )
    mine = _local_draft_logits(state, slots, candidates) / t  # [bs, steps, local]
    rel = drafts - lo
    own = (rel >= 0) & (rel < local)
    xq = mine.gather(-1, rel.clamp(0, local - 1)[..., None])[..., 0]
    xq = torch.where(own, xq, torch.full_like(xq, float("-inf")))
    g1 = _gather_small(state, torch.stack([torch.logsumexp(mine, dim=-1), xq], -1))
    lse_q = torch.logsumexp(g1[..., 0], dim=0)  # [bs, steps]
    lq_x = g1[..., 1].amax(dim=0) - lse_q
    ratio = lp_x - lq_x
    cum = []
    log_p = torch.zeros(bs, device=ratio.device)
    for i in range(steps):
        log_p = torch.clamp(log_p + ratio[:, i], max=0.0)
        cum.append(log_p)
    p = torch.stack(cum, dim=1).exp()  # [bs, steps]
    # Shard-local P and Q; residual for stopping after a = 0 .. steps-1 tokens.
    lp_loc = tl[..., lo : lo + w] / t - lse_p[..., None]  # [bs, n, w]
    q_loc = (mine[..., :w] - lse_q[..., None]).exp()  # [bs, steps, w]
    pa = torch.cat([torch.ones_like(p[:, :1]), p[:, : steps - 1]], dim=1)
    res = torch.clamp(pa[..., None] * lp_loc[:, :steps].exp() - q_loc, min=0.0)
    u_noise = torch.rand(bs, vocab, device=tl.device)[:, lo : lo + w]
    gum = -torch.log(-torch.log(u_noise.clamp(1.0e-7, 1.0 - 1.0e-7)))[:, None]
    r_val, r_idx = (torch.log(res.clamp_min(1.0e-38)) + gum).max(-1)  # [bs, steps]
    p_val, p_idx = (lp_loc + gum).max(-1)  # [bs, n]
    g2 = _gather_small(
        state,
        torch.cat(
            [res.sum(-1), r_val, (r_idx + lo).float(), p_val, (p_idx + lo).float()],
            dim=1,
        ),
    )  # [tp, bs, 3 * steps + 2 * n]
    rsum = g2[..., :steps].sum(0)
    rv, ri = g2[..., steps : 2 * steps], g2[..., 2 * steps : 3 * steps]
    pv, pi = g2[..., 3 * steps : 3 * steps + n], g2[..., 3 * steps + n :]
    r_tok = ri.gather(0, rv.argmax(0, keepdim=True))[0].long()  # [bs, steps]
    p_tok = pi.gather(0, pv.argmax(0, keepdim=True))[0].long()  # [bs, n]
    resid = rsum[:, 1:]
    denom = resid + 1.0 - p[:, : steps - 1]
    h = torch.where(denom > 0, resid / denom.clamp_min(1.0e-30), torch.ones_like(resid))
    h = torch.cat([h, p[:, steps - 1 :]], dim=1)
    u = torch.rand(bs, steps, device=h.device).clamp_min(1.0e-7)
    idx = torch.arange(1, steps + 1, device=h.device)
    acc = torch.where(u <= h, idx, torch.zeros_like(idx)).amax(dim=1)
    rows = torch.arange(bs, device=h.device)
    a_res = acc.clamp_max(steps - 1)
    use_p = (acc == steps) | (rsum[rows, a_res] <= 0)
    final = torch.where(use_p, p_tok[rows, acc], r_tok[rows, a_res])
    pos = torch.arange(n, device=h.device)[None, :]
    toks = torch.cat([drafts, drafts[:, :1]], dim=1)
    out = torch.where(pos < acc[:, None], toks, torch.zeros_like(toks))
    out = torch.where(pos == acc[:, None], final[:, None], out)
    predict.view(bs, n).copy_(out.to(predict.dtype))
    accept_index.copy_(
        torch.where(
            pos <= acc[:, None],
            rows[:, None] * n + pos,
            torch.full_like(pos, -1) + rows[:, None] * 0,
        ).to(accept_index.dtype)
    )
    accept_length.copy_(acc.to(accept_length.dtype))


def block_verify_chain(
    target_logits: torch.Tensor,
    draft_logits_full: torch.Tensor,
    candidates: torch.Tensor,
    temperature: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
) -> None:
    """Fill ``predict``/``accept_index``/``accept_length`` like
    ``verify_chain_target_sampled`` (``accept_length`` excludes the bonus).

    Args:
        target_logits: ``[bs * n, V]`` FP32 target logits, ``n = steps + 1``.
        draft_logits_full: ``[bs, steps, >= V]`` FP32 draft logits.
        candidates: ``[bs, n]`` int32 anchor + drafts.
        temperature: ``[bs]`` sampling temperature (> 0).
    """
    bs, n = candidates.shape
    steps = n - 1
    vocab = target_logits.shape[-1]
    t = temperature.view(bs, 1, 1).clamp_min(1.0e-6)
    lp = torch.log_softmax(target_logits.view(bs, n, vocab) / t, dim=-1)
    lq = torch.log_softmax(draft_logits_full[..., :vocab] / t, dim=-1)
    drafts = candidates[:, 1:].long()
    ratio = (
        lp[:, :steps].gather(-1, drafts[..., None])[..., 0]
        - lq.gather(-1, drafts[..., None])[..., 0]
    )
    cum = []
    log_p = torch.zeros(bs, device=ratio.device)
    for i in range(steps):
        log_p = torch.clamp(log_p + ratio[:, i], max=0.0)
        cum.append(log_p)
    cum = torch.stack(cum, dim=1)  # [bs, steps]
    p = cum.exp()
    # Residual mass at positions 0 .. steps-2 looks one row ahead.
    resid = torch.clamp(
        p[:, : steps - 1, None] * lp[:, 1:steps].exp() - lq[:, 1:].exp(), min=0.0
    ).sum(-1)
    denom = resid + 1.0 - p[:, : steps - 1]
    h = torch.where(denom > 0, resid / denom.clamp_min(1.0e-30), torch.ones_like(resid))
    h = torch.cat([h, p[:, steps - 1 :]], dim=1)  # last position: h = p
    u = torch.rand(bs, steps, device=h.device).clamp_min(1.0e-7)
    ok = u <= h
    idx = torch.arange(1, steps + 1, device=h.device)
    acc = torch.where(ok, idx, torch.zeros_like(idx)).amax(dim=1)  # [bs]
    # Next token: residual after acc accepted tokens, or P after a full block.
    rows = torch.arange(bs, device=h.device)
    pa = torch.where(
        acc > 0,
        p.gather(1, (acc - 1).clamp_min(0)[:, None])[:, 0],
        torch.ones_like(acc, dtype=p.dtype),
    )
    p_row = lp[rows, acc].exp()  # [bs, V]
    q_row = lq[rows, acc.clamp_max(steps - 1)].exp()
    res = torch.clamp(pa[:, None] * p_row - q_row, min=0.0)
    res = torch.where(
        (acc == steps)[:, None] | (res.sum(-1, keepdim=True) <= 0), p_row, res
    )
    g = -torch.log(-torch.log(torch.rand_like(res).clamp(1.0e-7, 1.0 - 1.0e-7)))
    final = (torch.log(res.clamp_min(1.0e-38)) + g).argmax(-1)  # [bs]
    # Pack like verify_chain_target_sampled.
    pos = torch.arange(n, device=h.device)[None, :]
    toks = torch.cat([drafts, drafts[:, :1]], dim=1)  # predict[j] = draft j+1
    out = torch.where(pos < acc[:, None], toks, torch.zeros_like(toks))
    out = torch.where(pos == acc[:, None], final[:, None], out)
    predict.view(bs, n).copy_(out.to(predict.dtype))
    accept_index.copy_(
        torch.where(
            pos <= acc[:, None],
            rows[:, None] * n + pos,
            torch.full_like(pos, -1) + rows[:, None] * 0,
        ).to(accept_index.dtype)
    )
    accept_length.copy_(acc.to(accept_length.dtype))


def block_verify_chain_chunked(
    state: DSparkBlockVerifyState,
    slots: torch.Tensor,
    target_logits: torch.Tensor,
    candidates: torch.Tensor,
    temperature: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    chunk: int,
) -> None:
    """``draft_logits`` + ``block_verify_chain`` over ``chunk`` requests at a time.

    The full-vocab FP32 temporaries scale with the batch (about 6 x bs x
    steps x vocab x 4 bytes); captured graphs keep them in their memory pool,
    which comes out of the KV cache budget. Chunking bounds them by ``chunk``.
    """
    bs, n = candidates.shape
    pred = predict.view(bs, n)
    aidx = accept_index.view(bs, n)
    tp = _BLOCK_VERIFY_TP and state.tp_size > 1
    for s in range(0, bs, chunk):
        e = min(bs, s + chunk)
        if tp:
            block_verify_chain_tp(
                state,
                slots[s:e],
                target_logits[s * n : e * n],
                candidates[s:e],
                temperature[s:e],
                pred[s:e],
                aidx[s:e],
                accept_length[s:e],
            )
            sub = aidx[s:e]
            sub.copy_(torch.where(sub >= 0, sub + s * n, sub))
            continue
        dl = draft_logits(state, slots[s:e], candidates[s:e])
        block_verify_chain(
            target_logits[s * n : e * n],
            dl,
            candidates[s:e],
            temperature[s:e],
            pred[s:e],
            aidx[s:e],
            accept_length[s:e],
        )
        sub = aidx[s:e]
        sub.copy_(torch.where(sub >= 0, sub + s * n, sub))

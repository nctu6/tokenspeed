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

"""Shadow n-gram draft statistics (diagnostic only; never changes outputs).

At request finish, replays the generated tokens as a speculative loop that drafts
by suffix matching against the prompt and the output so far, and reports how
many tokens such a draft would have had accepted per step. It answers whether an
n-gram drafter can add to the model drafter on this workload.
"""

from __future__ import annotations

import os

SHADOW_NGRAM = os.environ.get("TOKENSPEED_SHADOW_NGRAM", "0") == "1"
SHADOW_NGRAM_N = int(os.environ.get("TOKENSPEED_SHADOW_NGRAM_N", "4"))
SHADOW_NGRAM_K = int(os.environ.get("TOKENSPEED_SHADOW_NGRAM_K", "5"))
# Only the tail of a long prompt is indexed to bound the scheduler-thread cost.
SHADOW_NGRAM_PROMPT_TAIL = int(
    os.environ.get("TOKENSPEED_SHADOW_NGRAM_PROMPT_TAIL", "16384")
)


def shadow_ngram_summary(prompt: list[int], output: list[int]) -> str | None:
    """Replay ``output`` as an n-gram speculative loop; return a one-line summary.

    Args:
        prompt: Prompt token ids.
        output: Generated token ids, in order.

    Returns:
        A log line, or None when the output is too short to replay.
    """
    n, k = SHADOW_NGRAM_N, SHADOW_NGRAM_K
    if len(output) < 8:
        return None
    ctx = list(prompt[-SHADOW_NGRAM_PROMPT_TAIL:]) + list(output)
    base = len(ctx) - len(output)
    # Last position at which each n-gram ended (exclusive end index).
    index: dict[tuple[int, ...], int] = {}
    for end in range(n, base + 1):
        index[tuple(ctx[end - n : end])] = end
    steps = hits = tokens = full = 0
    hit_accepted = 0
    p = 1  # output[0] comes from prefill: it is the known token opening step 1.
    indexed_to = base  # n-grams ending at or before this offset are indexed
    while p < len(output):
        cur = base + p  # context length at this step's start
        # Index n-grams that ended strictly before the current suffix, so the
        # suffix never matches itself.
        for end in range(max(indexed_to + 1, n), cur):
            index[tuple(ctx[end - n : end])] = end
        indexed_to = max(indexed_to, cur - 1)
        steps += 1
        accepted = 0
        match_end = index.get(tuple(ctx[cur - n : cur])) if cur >= n else None
        if match_end is not None and match_end < cur:
            hits += 1
            for j in range(k):
                if match_end + j >= cur or p + j >= len(output):
                    break
                if ctx[match_end + j] != output[p + j]:
                    break
                accepted += 1
            hit_accepted += accepted
            full += accepted == k
        tokens += accepted + 1
        p += accepted + 1
    return (
        f"ShadowNGram n={n} k={k} out={len(output)} steps={steps} "
        f"hit={hits / steps:.3f} acc_on_hit={hit_accepted / max(hits, 1):.2f} "
        f"full_k={full / steps:.3f} tok_per_step={tokens / steps:.2f}"
    )

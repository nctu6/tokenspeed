# DeepSeek V4 attention

`dsv4_prefill` attends BF16 queries `[tokens, heads, 512]` over a dense BF16
workspace of 512-wide latent rows selected per token (`indices`, `lens`), with
an optional per-head sink that joins the softmax denominator only. A V4 latent
row is key and value at once. `dsv4_decode` does the same from the FP8
page-planar caches. The V4.1 `dsv41_selected_attention` calls `dsv4_prefill`
after its bounded cache gather, and the portable `dsv4_decode` dequantizes
into a workspace and calls it too, so both models and both phases share this leaf.

| Leaf | Where | Notes |
|---|---|---|
| `flashmla_dsv4_prefill` / `flashmla_dsv4_decode` | SM90-SM109 | FlashMLA sparse kernels (wgmma/tcgen05); capped off SM120. |
| `cute_dsl_dsv4_mma_prefill` | SM120, more than 16 heads | CuTe DSL port of the next row (same contract, traits and batch invariance), `PERFORMANT + 1`. |
| `triton_dsv4_mma_prefill` | SM120 | Tensor-core kernel, any head count, int32/int64 metadata, sink and/or LSE, batch-invariant. |
| `triton_dsv4_prefill`, `triton_dsv4_decode` | any NVIDIA/AMD | CUDA cores; decode dequantizes into a BF16 workspace first. |

## SM120 (`_triton/sparse_mma.py`)

FlashMLA ships no SM120 kernels, so SM120 ran the portable kernel: one program
per (token, head) on CUDA cores, re-reading the selected rows for every head.
`triton_dsv4_mma_prefill` uses the tile of the SM120 DSA kernel
(`dsa/_triton/sparse_mma.py`, whose `plan_launch` picks the head tile): a
program owns 16-32 heads of one token, multiplies each gathered tile of 32 rows
by all of them on `mma.sync` (`S = Q K^T`, `O += P K`, BF16 `m16n8k16`) with an
FP32 online softmax, and loads the next tile's row ids before the current
math (task 17: prefetching the whole BF16 tile into registers spilled; gathering
the rows where they are used is 1.24-1.31x faster from 128 tokens, 2447 vs
3212 us at 128 heads and 2048 tokens, and equal below 32 tokens). The
sink is one extra zero-valued key. Rows that are negative or past the
workspace are skipped. Token count, head count, width, row stride and chunk
count are runtime arguments.

Batch invariance. A prefill chunk replayed after a prefix-cache hit must
reproduce the original rows exactly (FlashMLA's sparse prefill does not split,
and `test_deepseek_v41_model` checks this). A split count that follows the
token count breaks that, so the selection is cut into at most 4 chunks whose
number depends on the width only. Each chunk runs its own online softmax and
`_merge` (explicit `fma`) folds the chunk states in order. An unsplit program
folds them in registers; when the batch cannot fill the GPU, every chunk gets
a program and the combine kernel folds the stored states with the same helper
in the same order. Rows are bitwise the same in any batch
(`test_rows_do_not_depend_on_the_batch`). This costs 5-15% against a
token-count split (1427 vs 1618 us at 2048 tokens and 64 heads; 9.9 vs 14.6 us
for one token). Folding in one tile loop (merging at chunk ends) was slower at
32-head tiles (2352 us), and 8 chunks was faster below 8 tokens but 4-25%
slower above.

Measured on an RTX PRO 6000 (CUDA graph, 640 selected rows = 128 window + 512
top-k; `portable` is `triton_dsv4_prefill`):

| heads | tokens | leaf | TFLOP/s | portable |
|---:|---:|---:|---:|---:|
| 16 | 1 | 14.6 us | 1 | 57 us (4x) |
| 16 | 128 | 45.2 us | 59 | 268 us (6x) |
| 16 | 2048 | 518 us | 83 | 3764 us (7x) |
| 64 | 1 | 14.6 us | 6 | 56 us (4x) |
| 64 | 16 | 22.3 us | 60 | 162 us (7x) |
| 64 | 128 | 134 us | 80 | 993 us (7x) |
| 64 | 2048 | 1618 us | 106 | 14496 us (9x) |
| 64 | 2048 (width 128) | 446 us | 77 | 3078 us (7x) |
| 128 | 128 | 210 us | 102 | 1862 us (9x) |
| 128 | 2048 | 3080 us | 112 | |

The gather workspace in the decode path costs 17-25% of
`dsv41.selected_attention` at 64 tokens (25 of 141 us at 64 heads); the rest is
the attention itself plus torch mask construction. Compute bound shapes of the
Triton leaf sit at about 35% of cuBLAS's 413 TFLOP/s BF16. Open: a decode
kernel that reads the FP8 page-planar rows directly (dequantizing the 64-wide
UE8M0 blocks in registers) would drop the BF16 workspace; the V4/V4.1 formats
differ per cache role, so it needs one reader per format.

## CuTe DSL leaf (`_cute_dsl/sparse_mma.py`)

`cute_dsl_dsv4_mma_prefill` runs the BF16 path of the DSA CuTe DSL core
(`dsa/_cute_dsl/sparse_mma.py`: `cp.async` two-stage ring of 32-row tiles,
4 channel warps per 16-head group with one 16 x 128 FP32 accumulator each,
channel-split scores, `ldmatrix(.trans)` fragments, `mma.sync` `m16n8k16`) in a
V4 variant: rank 512 and no RoPE columns, a flat workspace row table (a row id
addresses `kv + id * row_stride`), int32 or int64 row ids checked against the
workspace in the kernel, strided query and output, and the sink as chunk 0's
initial state (`m` = sink logit, `l` = 1), so a row with nothing live still
returns the sink-only state. The chunked fold and its helpers are the DSA
core's: at most 4 chunks cut by the width alone (`num_chunks_for`), folded in
order with explicit `fma` in the CTA or, for small batches, by the combine
kernel; rows are bitwise the same for any batch, head tile and split. Layouts
the vector loads cannot address (`supports`) run the Triton leaf. Binaries are
keyed by head tile, metadata dtypes and sink; token count, width, row count and
strides are runtime values.

`plan_launch` aims at two thirds of the SMs (one chunk per CTA when split) and
uses 16-head CTAs only up to 32 of them. Selection: the leaf declares
`num_q_heads_min = 17`. A 16-head CTA has 4 warps and one per SM, and loses to
the Triton leaf from 64 tokens (0.70-0.79x at 64-4096 tokens, 640 rows); the
rule follows the model's head count, never the batch, so one model never mixes
the two leaves. Measured against the Triton leaf (CUDA graph, sink, 1-4096
tokens, widths 128 / 640 / 1152; 117 points):

| heads | width 128 | width 640 | width 1152 |
|---:|---|---|---|
| 32 | 1.10x (0.89-1.60x) | 1.13x (0.99-1.31x) | 1.13x (1.02-1.38x) |
| 64 | 1.13x (0.90-1.63x) | 1.13x (0.99-1.32x) | 1.15x (1.02-1.37x) |
| 128 | 1.13x (0.89-1.60x) | 1.15x (1.00-1.32x) | 1.16x (1.04-1.39x) |

1.13x geomean overall. 128 heads at 4096 tokens: 4.38 ms at 640 rows (157
TFLOP/s, Triton 4.85 ms) and 7.30 ms at 1152 (170 TFLOP/s, 42% of the BF16
floor). The only losses are window-only selections (128 rows) of 1-2 tokens,
7.2 vs 6.4 us (launch-bound); choosing the leaf by token count would change a
row's bits with its batch, so they stay. More chunks (8) help only below 16
tokens (10.9 vs 13.7 us at one token) and cost from 16 tokens, and the count
may not follow the batch; fewer (1-2) lose 2-3x below 16 tokens. Kept 4.

Tests: `test/nvidia/ops/attention/test_dsv4_mma_sm120.py` (both leaves: float32
reference to one BF16 ulp, LSE to 2e-5, strided q/out, bitwise batch
invariance; Triton recompile guard; CuTe sink-with-LSE and empty rows, compile
cache guard; selection by head count) and
`ops/test_attention_dsv4*.py`; `bench_dsv4_mma_sm120.py` reproduces the tables.

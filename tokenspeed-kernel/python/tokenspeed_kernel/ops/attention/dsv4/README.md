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
| `triton_dsv4_mma_decode` | SM120, 16 heads from 32 tokens | Dequantizes the FP8 page-planar rows in the tile loop of the Triton kernel; no BF16 workspace. |
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

Compute bound shapes of the Triton leaf sit at about 35% of cuBLAS's 413
TFLOP/s BF16.
## Direct reader of packed caches (`_triton/paged_mma.py`)
The portable decode gathered every selected cache row into a BF16 workspace
(640 KB per query at 640 rows, written and read back) through a launch per
segment, a `cat` and mask and index construction (17-25% of
`dsv41.selected_attention` at 64 tokens, and most of it at 1-16 tokens, where
the glue and launches dominate). `_dsv4_sparse_mma_kernel` with the tile loaded
by dequantizing the packed rows in registers does the same work without the
workspace: FP8 with UE8M0 scales (V4.1 `swa`; V4 `swa_v4` / `global_v4` with
the BF16 RoPE tail) and FP4 with E4M3 scales (V4.1 `global`), page-planar,
strided cache views included. An E4M3 or E2M1 value times a power of two or an
E4M3 scale has at most four significant bits, so the BF16 tile is exact and the
result is bit-identical to the workspace route (and batch-invariant, with the
same chunking: the selection is `swa_width` columns then the compressed ones, a
tile belongs to one segment, so `swa_width` is a multiple of 32).
- `triton_dsv41_mma_selected_attention` (V4.1 `dsv41_selected_attention`; SM120
  has no FlashMLA): 128 window + 512 compressed rows, CUDA graph, 16/64/128
  heads x 1-256 tokens, 21 points: 1.66x geomean over the portable route, 1.01x
  worst (128 heads, 128 tokens), 3.7x best (16 heads, 256 tokens); 41 vs 24 us
  at one token. It runs the portable route for what it does not read (workspace
  prefill, a window width off the tile, odd layouts).
- `triton_dsv4_mma_decode` (`dsv4_decode`, V4 FP8 page-planar caches): selected
  only for 16 heads from 32 tokens (1.23x at 32, 1.79x at 128, 2.28x at 256).
  Above 16 heads the workspace route reaches the CuTe DSL dense kernel, 1.1-1.6x
  faster than this Triton one, and beats the direct reader (64 heads 0.69-1.11x,
  128 heads 0.72-0.79x of the workspace route; one token 17 vs 22 us); a few
  tokens at 16 heads is 0.75-0.8x. With 16 heads both routes run the Triton
  kernel, so the choice never changes a bit. Open: the same dequantizing tile in
  the CuTe DSL kernel, which would win above 16 heads too.
`bench_dsv41_paged_sm120.py v41|v4` reproduces the tables;
`test_dsv4_paged_mma_sm120.py` checks bit equality with the portable route (both
formats, 16-128 heads, holes, out-of-cache and short lengths), strided page
views, the no-sink LSE, batch invariance and recompilation.
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
`plan_launch` was re-swept (task 24, 3 head counts x 14 token counts x both
head tiles x 1/2/4 splits, widths 640 and 128): the best (tile, split) per
point is only 1.01x better than the shipped rule, and the DSA wave model loses
(0.97x geomean at width 128), so only 4 chunks of 5 tiles are planned by the
two-thirds rule. 16 heads with an 8-warp CTA (two channel splits per head
group, 32-row tiles, 16 KB exchange) is correct and 1.08-1.10x faster than the
4-warp CTA at 1-16 tokens, 0.92-0.93x of Triton at 64-128 tokens and 0.86x at
2048 (0.76x for window-only rows): better than the 4-warp CTA (0.77x at 2048)
but still behind Triton from 64 tokens, and a leaf that follows the batch would
change rows, so 16 heads stay on Triton. Its score sum has 8 partials, so it
would also not equal the 4-warp CTA bit for bit.

V4.1 prefill reaches this core too: the SM120 selected-attention leaf
(`triton_dsv41_mma_selected_attention`) hands its gathered BF16 workspace to
`dsv4_prefill` with an open solution, so the registry picks this core above 16
heads. The portable V4.1 leaf keeps `solution="triton"`; before the SM120 leaf
had its own route, that pin kept every SM120 V4.1 prefill on the Triton core
(about 7% of end-to-end prefill at 64 heads on real V4.1 weights).
Tests: `test/nvidia/ops/attention/test_dsv4_mma_sm120.py` (both leaves: float32
reference to one BF16 ulp, LSE to 2e-5, strided q/out, bitwise batch
invariance; Triton recompile guard; CuTe sink-with-LSE and empty rows, compile
cache guard; selection by head count) and
`ops/test_attention_dsv4*.py`; `test_dsv4_paged_mma_sm120.py::test_prefill_workspace_runs_the_registry_choice`
covers the V4.1 prefill route; `bench_dsv4_mma_sm120.py` reproduces the tables.

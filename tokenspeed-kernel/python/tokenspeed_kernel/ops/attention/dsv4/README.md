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
| `triton_dsv4_mma_prefill` | SM120 | Tensor-core kernel, 64/128/any heads, int32/int64 metadata, sink and/or LSE. |
| `triton_dsv4_prefill`, `triton_dsv4_decode` | any NVIDIA/AMD | CUDA cores; decode dequantizes into a BF16 workspace first. |

## SM120 (`_triton/sparse_mma.py`)

FlashMLA ships no SM120 kernels, so SM120 ran the portable kernel: one program
per (token, head) on CUDA cores, re-reading the selected rows for every head.
`triton_dsv4_mma_prefill` is the same tile design as the SM120 DSA kernel
(`dsa/_triton/sparse_mma.py`, whose `plan_launch` it reuses): a program owns
16-32 heads of one token, multiplies each gathered tile of 32 rows by all of
them on `mma.sync` (`S = Q K^T`, `O += P K`, BF16 `m16n8k16`) with an FP32
online softmax, issues the next gather before the current math, and splits the
selection across programs (merged by a second kernel) when the batch cannot
fill the GPU. The sink is one extra zero-valued key owned by split 0. Rows that
are negative or past the workspace are skipped. Token count, head count, width,
row stride and split count are runtime arguments.

Measured on an RTX PRO 6000 (CUDA graph, 640 selected rows = 128 window + 512
top-k; `portable` is `triton_dsv4_prefill`):

| heads | tokens | leaf | TFLOP/s | portable |
|---:|---:|---:|---:|---:|
| 16 | 1 | 9.8 us | 2 | 56 us (6x) |
| 16 | 128 | 44.7 us | 60 | 269 us (6x) |
| 16 | 2048 | 485 us | 89 | 3766 us (8x) |
| 64 | 16 | 31.4 us | 43 | 160 us (5x) |
| 64 | 128 | 107 us | 100 | 993 us (9x) |
| 64 | 2048 | 1431 us | 120 | 14542 us (10x) |
| 64 | 2048 (width 128) | 355 us | 97 | 3075 us (9x) |
| 128 | 128 | 177 us | 121 | 1865 us (11x) |
| 128 | 2048 | 2725 us | 126 | |

The gather workspace in the decode path costs 17-25% of
`dsv41.selected_attention` at 64 tokens (25 of 141 us at 64 heads); the rest is
the attention itself plus torch mask construction. Compute bound shapes sit at
25-30% of cuBLAS's 413 TFLOP/s BF16, the same ceiling as the other SM120
`mma.sync` Triton kernels. Open: a decode kernel that reads the FP8 page-planar
rows directly (dequantizing the 64-wide UE8M0 blocks in registers) would drop
the BF16 workspace; the V4/V4.1 formats differ per cache role, so it needs one
reader per format.

Tests: `test/nvidia/ops/attention/test_dsv4_mma_sm120.py` (float32 reference
to one BF16 ulp, LSE to 2e-5, strided q/out, recompile guard) and
`ops/test_attention_dsv4*.py`; `bench_dsv4_mma_sm120.py` reproduces the table.

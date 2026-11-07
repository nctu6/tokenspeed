# MLA kernels

`mla_decode_with_kvcache` attends absorbed MLA queries over a paged latent
cache (`kv_lora_rank` 512 + RoPE 64). Callers: MLA models served by the
generic `mla` backend (for example DeepSeek V3-style models and Kimi-K3) and
the draft/verify blocks of speculative decoding (`q_len` 2-8, or a non-causal
proposal block for DFLASH/DSPARK drafters).

| Leaf | Where | Notes |
|---|---|---|
| `tokenspeed_mla_decode_with_kvcache` | SM100-SM107 | CuTe DSL tcgen05 kernel; non-causal proposal blocks on the query axis (`q_len` 2-8). |
| `triton_mla_mma_decode_with_kvcache` | SM120 | Tensor-core split-K kernel; `q_len` 1-8, causal and non-causal blocks, window, LSE. |
| `triton_mla_decode_with_kvcache` | any NVIDIA/AMD | One program per (row, head) on CUDA cores; `q_len` 1, logit cap. |
| `gluon_*` | AMD | |

`tokenspeed_mla` needs tcgen05 and stays capped at 10.7, as does the
`tokenspeed_mla` attention backend; Kimi-K3's default history consumer
therefore resolves to the generic `mla` backend on SM120.

## SM120 (`_triton/decode_mma.py`)

The portable kernel re-reads the whole cache for every head and multiplies on
CUDA cores: 14-131x off the memory roofline at 16 heads and 21-49x at 128
heads. The SM120 leaf tiles 16 or 32 `(query, head)` rows of one request over a
single pass of the cache; `S = Q K^T` and `O += P V` run on `mma.sync` (E4M3
`m16n8k32` when query and cache are both E4M3, BF16/FP16 otherwise) with an
FP32 online softmax, and the latent row is both K and V. The next tile's load
is issued before the current tile's math. When a batch has too few row tiles
to fill the GPU (`plan_launch`), the cached range is split across up to 128
programs and a small kernel merges the (max, sum, accumulator) states; a split
past a request's end exits at once.

Masks follow the Triton kernel's contract. With `L` cached tokens and `Q`
query rows, row `i` sees the first `L - (Q - 1 - i)` tokens (causal block,
`noncausal_block_size == 1`) or all `L` (non-causal block); a window keeps
tokens from `visible - block - window_left + position`. The leaf declares the
query axis for `q_len` 1-8, so verify and proposal blocks no longer fan out to
one flattened row per position: with `Q = 4`, batch 32 and 8192 tokens it is
1.37x faster at 16 heads (383 vs 523 us) and a tie at 128 heads, where the
kernel is compute bound. `logit_cap` stays
on the portable kernel; plans that need `batch_invariant` do too, because the
split count follows the batch size (never the cached lengths or
`max_seqlen_k`). Batch size, `q_len`, head count, the page-table width and the
split count are runtime arguments; the binaries are keyed by tile width, the
combine kernel's power-of-two split bound and the dtype.

Measured on an RTX PRO 6000 (CUDA graph; memory roofline is the cache bytes at
1461 GB/s, the compute reference is cuBLAS at 413 TFLOP/s BF16 and 582 FP8):

| dtype | heads | batch x length | `Q` | leaf | roofline | TFLOP/s | portable |
|---|---:|---|---:|---:|---:|---:|---:|
| BF16 | 16 | 1 x 4096 | 1 | 20 us | 3.2 us | 7 | 629 us (31x) |
| BF16 | 16 | 1 x 32768 | 1 | 38 us | 25.9 us | 30 | 4999 us (131x) |
| BF16 | 16 | 8 x 8192 | 1 | 45 us | 51.7 us | 50 | 1255 us (28x) |
| BF16 | 16 | 32 x 8192 | 1 | 236 us | 207 us | 39 | 3309 us (14x) |
| BF16 | 128 | 1 x 32768 | 1 | 122 us | 25.9 us | 75 | 5017 us (41x) |
| BF16 | 128 | 8 x 8192 | 1 | 218 us | 51.7 us | 84 | 6066 us (28x) |
| BF16 | 128 | 32 x 8192 | 1 | 713 us | 207 us | 102 | 22365 us (31x) |
| BF16 | 16 | 8 x 8192 | 4 | 114 us | 51.7 us | 80 | |
| E4M3 | 16 | 32 x 8192 | 1 | 122 us | 103 us | 75 | 3406 us (28x) |
| E4M3 | 128 | 32 x 8192 | 1 | 513 us | 103 us | 142 | 24928 us (49x) |
| E4M3 | 128 | 8 x 8192 | 4 | 650 us | 25.9 us | 112 | |

At 16 heads it is 0.9-1.5x of the memory roofline from 8 requests up. At 128
heads the kernel is compute bound: 100-142 TFLOP/s is 25-35% of what cuBLAS
reaches, which `mma.sync` through Triton does not close (sweeps of the key tile,
row tile, warp count, stages and split limit are flat around the shipped
values; the DSA notes show the same ceiling). A CuTe
DSL kernel with `ldmatrix`/`cp.async` staging is the open route, worth up to
about 3x on the compute-bound shapes. Prefill (`triton_mla_prefill`, a
`tl.dot` flash kernel) already runs at 104-146 TFLOP/s, the same efficiency, so
it is not ported.

Tests: `test/nvidia/ops/attention/test_mla_decode_mma_sm120.py` (float32
reference to one output ulp, LSE to 2e-5, graph replay, recompile guard) and
`test/ops/test_batch_shape_recompiles.py`; `bench_mla_decode_sm120.py`
reproduces the table.

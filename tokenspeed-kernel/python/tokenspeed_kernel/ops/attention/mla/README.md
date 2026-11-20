# MLA kernels

`mla_decode_with_kvcache` attends absorbed MLA queries over a paged latent
cache (`kv_lora_rank` 512 + RoPE 64). Callers: MLA models served by the
generic `mla` backend (for example DeepSeek V3-style models and Kimi-K3) and
the draft/verify blocks of speculative decoding (`q_len` 2-8, or a non-causal
proposal block for DFLASH/DSPARK drafters).

| Leaf | Where | Notes |
|---|---|---|
| `tokenspeed_mla_decode_with_kvcache` | SM100-SM107 | CuTe DSL tcgen05 kernel; non-causal proposal blocks on the query axis (`q_len` 2-8). |
| `cute_dsl_mla_mma_decode_with_kvcache` | SM120, BF16 / E4M3 caches | CuTe DSL port of the next row (same contract and traits), `PERFORMANT + 1`. |
| `triton_mla_mma_decode_with_kvcache` | SM120 | Tensor-core split-K kernel; `q_len` 1-8, causal and non-causal blocks, window, LSE; FP16 caches. |
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
heads the Triton kernel is compute bound at 100-142 TFLOP/s, 25-35% of what
cuBLAS reaches; sweeps of the key tile, row tile, warp count, stages and split
limit are flat around the shipped values. Prefill (`triton_mla_prefill`, a
`tl.dot` flash kernel) runs at 104-146 TFLOP/s, the same efficiency, and is not
ported.

## CuTe DSL leaf (`_cute_dsl/decode_mma.py`)

`cute_dsl_mla_mma_decode_with_kvcache` subclasses the DSA CuTe DSL core
(`dsa/_cute_dsl/sparse_mma.py`) and keeps its tile math: `cp.async` two-stage
ring, 4 channel warps per 16 rows with one 16 x 128 FP32 accumulator each,
E4M3 64-row tiles with the query in XOR-swizzled shared memory and key-split
scores, BF16 32-row tiles with channel-split scores, `ldmatrix(.trans)` and
`mma.sync`, the shared softmax/PV step. What changes is the gather and the
mask, both the Triton leaf's: a CTA walks the contiguous position range its
rows can see, one tile of consecutive positions at a time, resolving each
through the request's page table (`pos >> log2(page_size)`), and masks each row
to `first <= p < visible` (causal or non-causal block, window). The split, the
partial states and `plan_launch` are the Triton leaf's (shape by batch size
only, up to 128 splits). The merge kernel gives each split a thread, reduces
the maximum and the weights in a fixed order and skips splits of weight zero;
an empty split stores only `l = 0`. Like the Triton leaf it does not declare
`batch_invariant`. FP16 caches keep the Triton leaf; layouts the vector loads
cannot address (`supports`) run the Triton launch. Binaries are keyed by cache
dtype and row tile; batch, `q_len`, heads, page size, table width, masks and
split count are runtime values.

Against the Triton leaf (CUDA graph, 1-128 requests x 1K-32K tokens, `Q` = 1;
32 points per row):

| dtype | heads | geomean | range | best |
|---|---:|---|---|---|
| BF16 | 16 | 1.09x | 0.97-1.40x | 1.38x at 4 x 16K (36 vs 50 us) |
| BF16 | 128 | 1.67x | 1.23-1.95x | 213 TFLOP/s at 128 x 32K (5.48 vs 10.17 ms) |
| E4M3 | 16 | 1.29x | 1.01-1.96x | 1.96x at 8 x 16K (34 vs 67 us) |
| E4M3 | 128 | 1.94x | 1.23-2.31x | 355 TFLOP/s at 128 x 32K (3.29 vs 7.60 ms) |

1.46x geomean. With `Q` = 4 and 8 (16 and 128 heads, 1-128 requests, 4K and
32K): 1.84x (1.13-2.34x; E4M3 128 heads up to 380 TFLOP/s). The closest points
are BF16 16 heads from 16 requests of 16K up (0.97-1.00x), where both leaves
stream the cache at about 1.45 TB/s, the DRAM roofline (smaller caches stay in
the 128 MB L2 across graph replays, which is where the 16-head wins come
from). The first version lost at one request of
16 heads (0.75x at 1K, 23.6 vs 17.8 us): its merge kernel looped over 128
splits serially; the parallel merge with weight-zero skipping made it 1.23x.

Tests: `test/nvidia/ops/attention/test_mla_decode_mma_sm120.py` (both leaves:
float32 reference to one output ulp, LSE to 2e-5, causal / non-causal blocks and
windows on the query axis, empty and short requests in split and unsplit
launches, strided output, graph replay; Triton recompile guard; CuTe compile
cache guard over batch, lengths, page size, masks and `q_len`; FP16 selection)
and
`test/ops/test_batch_shape_recompiles.py`; `bench_mla_decode_sm120.py`
reproduces the table.

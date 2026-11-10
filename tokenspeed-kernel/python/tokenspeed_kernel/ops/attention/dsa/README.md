# Deep sparse attention kernels

The DSA operators separate sparse-history selection from attention:

- `dsa_prefill_topk` and `dsa_decode_topk` score the index cache and produce
  padded global KV slots plus a live length for every query.
- `dsa_prefill` and `dsa_decode` consume those selected slots and read the
  latent KV cache.
- `dsa_plan` owns optional decode planning state.

AMD CDNA4/gfx950 and CDNA5/gfx1250 provide Gluon implementations for full
selected attention. The gfx1250 implementation supports BF16, E4M3, and E5M2
queries; dense or packed KV rows; page size 64; latent ranks 128 and 512; and
optional 64-wide RoPE. A zero RoPE width selects the same kernel path with the
RoPE storage, gathers, and score term compiled out.

GLM-5.3-Flash uses the zero-RoPE specialization with
`qk_nope_head_dim=256`, `kv_lora_rank=512`, and 16 local heads at TP4. Its
KPool selection has a configured width of 2048 and can append three tail
positions, so both prefill and decode accept padded slot widths 2048 through
2051. The live `topk_lens` value determines how many entries participate in
the online softmax; unused entries remain `-1`.

Portable Triton implementations remain the fallback for trait combinations
without a matching native registration.

## Materialized Triton row top-k

`triton_topk_from_logits(logits, topk, seq_lens=..., enable_pdl=...)` selects
FP32 rows with descending scores and smaller column ids first on ties.
`seq_lens` is an int32 or int64 vector on the scores' device, clamped per row
to `[0, width]`; explicit `None` selects the full row. Scores and lengths may
be strided. Invalid scores inside the visible prefix still use `-inf` and
produce `-1` output padding. Entries beyond the prefix are never read and may
be uninitialized.

The sorting and radix implementations both bound input reads by these device
lengths. Radix histogram reduction excludes unwritten tiles and selects at
most `min(topk, seq_lens[row])` entries, including empty and short rows.
CUDA Graph replay may change the lengths without changing allocation or launch
capacity. QSA passes its complete compressed-block counts to both CUDA and
Triton selection, so its scorer writes only valid prefixes.

## Persistent CUDA row top-k

`persistent_topk` selects unsorted indices for `k` of 512, 1024, or 2048.
It reads only the visible prefix, applies causal lengths for grouped queries,
and pads short rows with `-1`. Equal scores may select any distinct tied indices.
Its ordered keys rank `-0.0` below `+0.0`; it does not promise lowest-index tie
selection or declare `batch_invariant`. The lowest-index tie contract in
`docs/design/numerics.md` applies to leaves selected with `batch_invariant=True`.
NaNs in the visible prefix are unsupported: value selection is unspecified,
but output indices and writes remain in bounds. The unread suffix may contain
NaNs. Empty or unmatched refinement histograms never reuse stale thresholds.

The register, streaming, and cluster implementations share a coarse FP16
histogram followed by exact FP32 selection. If the threshold bucket exceeds
the 2048-entry candidate buffer, additional FP32 radix passes narrow it before
collection. Refinement skips the leading bits fixed by the coarse bounds and
reuses the existing histogram storage, without increasing shared memory. It stops
when candidates fit or all remaining keys are identical. Truncation is safe only
in the latter case. Ordered integer bounds preserve nearest-even FP16 bucket
boundaries, including negative midpoints, signed zero, and infinities. Candidates
retain their integer keys through tie selection so subnormal scores remain
distinct under CUDA fast math. Cluster ranks reduce every refinement histogram
together before collecting candidates. Equal reduced histograms and counts keep
every rank's refinement rounds and cluster barriers in sync.

## Index-K plane formats

`dsa_prefill_topk` and `dsa_decode_topk` read the storage of the index-key
plane off its dtype and pass it to selection as the `index_k_format` and
`index_k_layout` traits; a top-k leaf declares the planes it scores and
selection never hands it another one. One layout per dtype, and the facades
never convert a plane:

| dtype | `index_k_format` | `index_k_layout` | row |
| --- | --- | --- | --- |
| `uint8` | `fp8_scaled` | `packed` | `[slots, head_dim + 4 * head_dim / 128]`: FP8 E4M3 keys followed by one fp32 scale per 128 elements |
| `uint8` | `fp8_scaled` | `page_planar` | any other uint8 shape: per-page planes of keys and scales, the outer page stride possibly padded |
| `bfloat16` | `bf16` | `packed` | `[slots, head_dim]`: the keys as the indexer produced them, no scale plane |

The in-tree DeepGEMM, Triton, and Gluon leaves score `fp8_scaled` planes; a
leaf scoring the checkpoint's bf16 keys (an indexer in the RL trainer's
order) registers `index_k_format={"bf16"}`, `index_k_layout={"packed"}`, and
the `batch_invariant` and `forced_initial_local` features, so a bf16 plane
selects it and nothing else. A plane of any other dtype is a `TypeError`.

`dsa_prefill_topk` also takes the index keys as rows already in
workspace-row order instead of a plane (the query-context-parallel history
gather over page-sharded caches assembles them): `index_k_fp8` +
`index_k_scale` are the rows of an `fp8_scaled` plane (`[workspace_rows,
head_dim]` uint8 or float8_e4m3fn, `[workspace_rows, head_dim / 128]` fp32),
`index_k_bf16` the rows of a `bf16` one (`[workspace_rows, head_dim]` bf16),
each one row per entry of `kv_workspace_slots` and selecting with that
`index_k_format` and `index_k_layout="packed"`, never together and never with
`index_k_cache`; a call with neither a plane nor rows is a `ValueError`.
Rows additionally require the `index_k_workspace_rows` feature
(`dsa.INDEX_K_WORKSPACE_ROWS_FEATURE`): a leaf declares it exactly when its
launcher takes the row keywords for its format (DeepGEMM does, for the FP8
pair; a bf16 leaf declares it and takes the `index_k_bf16` keyword), the
facade hands the keywords to declaring leaves only, and a leaf that only
reads planes -- the portable Triton leaf, the Gluon wrappers -- is never
selected for rows, not by ranking and not by a kernel override (an override
skips traits but not required features). The failure is a
`NoKernelFoundError` at selection; a host whose sharded prefill hands
rows probes that selection at construction with
`dsa.select_dsa_prefill_topk_for_rows(index_k_format=, ...)`, which makes a
platform without a declaring leaf a startup error.

A `dsa_decode_topk` leaf bounds every query row itself: row `j` of a request
scored with `q_len_per_req` rows (spec verify, a multi-depth draft's k-row
window) selects over the first `seq_lens[req] - (q_len_per_req - 1) + j`
positions, derived from `seq_lens`. The `seq_lens_2d` rows the facade hands
every leaf (`[tokens, 1]`, each carrying the request's full length) are the
scoring extent the `plan` was built from, not per-row bounds; a leaf that read
them as bounds would let a verify or draft row select its window's later
rows, and a sparse core that trusts the selection for causality (`kv_seq_lens`
is optional on `dsa_decode`) would attend them.

`candidate_lens_cpu` (the CPU mirror of each prefill token's candidate count)
goes to every selected leaf registered with the `candidate_lens_cpu` feature
(`dsa.CANDIDATE_LENS_CPU_FEATURE`) and to no other: a leaf that can size its
launches from it declares the feature alongside the keyword, and the facade
reads the registration rather than probing call signatures, so a
`*args, **kwargs` wrapper never receives a keyword its launcher cannot take.

## Slot order of the sparse cores

`dsa_decode` and `dsa_prefill` take a required `slot_order` in
`SLOT_ORDERS = ("selection", "sorted")`, passed to selection as the
`slot_order` trait: `selection` reduces a token's selected slots in the order
the top-k leaf emitted them and is what every core does by default (a core
need not declare the trait); `sorted` reduces them in ascending slot order,
so the reduction is batch-invariant whenever the selected set is, and is
served only by cores declaring `slot_order={"sorted", ...}`, which receive
the choice as the `slot_order` keyword. Asking a silent core for `sorted`
is a `ValueError`, not a silent fallback.

## Row top-k selection in CuTe DSL (`_cute_dsl/deep_select.py`)

`deepselect_topk(scores, ends, topk, capacity=..., cluster_size=...)` selects
the `topk` largest entries of every FP32 row and returns their unsorted
column indices and values. It is an in-tree CuTe DSL rendition of the
DeepSeek DeepSelect algorithm for Hopper and SM120, where the upstream package
ships no cubin; the V4.1 indexer uses it on sm90 and SM120 for both row
selection and block-maxima selection, and the V3.2 DeepGEMM leaves use it on
SM120 (see "DSA indexers on SM120" below).

Contract:

- `scores` is CUDA FP32 `[rows, width]` with unit column stride, a row
  stride that is a multiple of 4 elements, and a 16-byte aligned base;
  `ends` is int32 `[rows]` and bounds each row (clamped to `[0, width]`).
- `capacity` is the compiled survivor capacity (512, 1024, or 2048) and
  bounds `topk`; `capacity_for` gives the smallest one that serves a `topk`.
- A row with `ends[r] <= topk` yields `0 .. ends[r]-1` then `-1` slots
  scoring `-inf`. Otherwise every slot is a distinct index below `ends[r]`.
  Ties at the k-th value break arbitrarily but deterministically, so `-inf`
  masking may be selected when fewer than `topk` finite entries exist;
  consumers filter by value. NaN entries are never selected and leave `-1`.
- `cluster_size` in `{1, 2, 4, 8}` splits a row across a thread-block
  cluster (non-leader CTAs ship their survivors to the leader over
  distributed shared memory). `choose_cluster_size` picks it: rows of at
  least 32K entries, and only while every cluster stays resident (about 60%
  of the SMs for 4- and 8-CTA clusters, 80% for pairs; on SM120, where a CTA
  fills an SM, 85% and 95%). Every configuration a call may pick must be
  compiled with `warmup` before CUDA-graph capture. Rows, width and top-k are
  runtime arguments: only the capacity and cluster size select a binary.

Algorithm per CTA: an 8-bit radix select over order-preserving keys selects
the first 8192 entries (the row's in-order tail of up to 4096 entries plus
the first pseudo-randomly ordered 512-entry segments) exactly and sets the
running threshold; the remaining segments stream through a three-stage
`cp.async.bulk` ring, and the CTA appends only entries above the threshold
as `(index, value)` pairs; once 4032 candidates accumulate, and once more at
the end, a radix select over survivors plus candidates re-selects `k` pairs
and raises the threshold. Per-pass histograms send non-matching keys to a
sink slot so the passes stay branch-free; one warp locates the pivot bucket
while the others wait, since the search is a latency chain.

The kernel above needs 207 KB (capacity 512) to 232 KB (capacity 2048) of
shared memory, and SM120 allows 99 KiB per block. The same code runs there in
a compact profile selected at import from the platform: 256 threads (rounds of
4096 entries, whose in-order tail is the row's last full round), a two-stage
ring, and a 3072-pair candidate buffer. A round's hits are appended in two
groups of warps, and a group that would overflow the buffer first triggers a
compaction, so the buffer only has to hold half a round (this is also why
compaction runs from 1024 candidates instead of 4032). That is 93 KB at
capacity 2048 and leaves the selected entries unchanged; the correctness tests
run unchanged on both profiles. At capacity 2048 the buffer gathers one peer,
so clusters are limited to pairs (quads at 1024 and below). An RTX PRO 6000
selects 2048 of one 4K row in 4.2 us and of one 128K row in 43 us (pair), against
41 and 198 us for the Triton radix select.

## DSA indexers on SM120

DeepGEMM's SM120 MQA-logits kernels serve both indexers; what differs is the
index-key row format and the selector that follows.

- **V3.2 `dsa_*_topk` / `kpool_prefill_topk`.** The DeepGEMM leaves are offered
  on SM120 by builds that report SM120 kernels (`deepgemm_supports_sm120`) and
  outrank the Triton leaves. Prefill selects with DeepSelect (1.3-1.9x faster
  than the Triton radix select it replaces; the TRT-LLM prefill top-k has no
  SM120 image). Decode keeps the TRT-LLM CuTe cluster runner, which is as fast
  or faster up to a cliff where its clusters stop fitting (rows x clusters over
  the SMs) and it falls back to a non-cluster kernel; DeepSelect takes over past
  it (`_use_deepselect_decode_topk`: 32K+ entries and 96+ rows, or 64K+ entries
  and 48+ rows; 2.1-2.7x faster on 128 rows of 64K-128K). Both selectors return
  exact top-k sets, so the choice only affects speed.
  Tiny prefill chunks are bound by the host: the GPU work of a 128-token chunk
  is ~40 us, and the leaf used to spend ~350 us issuing about 100 framework ops
  around it (277 vs 144 us for the Triton leaf). On SM120 `dsa_prefill_prepare`
  (`_triton/prefill_prepare.py`) does the query quantization, the weight
  folding, the candidate counts and the key gather in one launch, bit-identical
  to the framework ops it replaces (`test_dsa_prefill_prepare_sm120.py`; other
  platforms keep the ops because the quantizer they select is not checked
  against it), and DeepSelect reports workspace rows itself (`index_base`, the
  row start added to each selected column; `-1` padding stays). Host time per
  call drops from 360 to 118-124 us at 1-1024 tokens; the leaf beats the
  Triton leaf at every chunk size (back-to-back eager calls, 1-512 tokens
  1.35-6.0x, 1024 tokens 21x, 2K-8K tokens with prefixes 54-98x;
  `scripts/sm120_dsa_vs_tokenspeed.py` in DeepGEMM). The launch rule is "one
  path": no chunk-size rule exists or is needed.
- **V4.1 `dsv41_index_topk`.** SM120 stores the checkpoint's 68-byte rows
  (`cache_format="v41"`: 64 bytes of E2M1 pairs, then one UE8M0 word with the
  four 32-element scales, page-planar). One leaf, `deep_gemm_sm120_dsv41_index_topk`
  (window 12.0-12.9, registered only by builds with SM120 kernels), scores
  them with DeepGEMM's block-scaled SM120 MQA kernels and selects with
  DeepSelect: the paged kernel for Full passes, the contiguous one for one
  request's prefill chunk (a zero-stride table), and the sparse paged kernel
  for Reindex passes over long histories. Queries are packed like the keys and
  the tensor core applies both scales, so Full logits are FP32 sums of exact
  products and the selected sets match an exact reference; the sparse kernel
  writes BF16 logits, so Reindex picks may swap within one BF16 rounding of the
  k-th score. The sm100 packed leaf (`deep_gemm_dsv41_index_topk`) and the
  upstream DeepSelect leaves stay on SM100/SM103, since that package does not
  fit SM120's shared memory. The same leaf scores the 132-byte FP8 `index_v4`
  rows (the `v4` cache format, not selected on SM120) like Hopper does.
  Measured on an RTX PRO 6000 (32 heads, top-k 512, 2048 candidate blocks, CUDA
  graph, median of 3), Triton time over native time on packed rows:

  | pass | decode, 1-128 requests x 1K-32K rows | chunked prefill, 128-4096 queries x 4K-32K rows |
  |---|---|---|
  | Full | 5.6-16x | 11-46x |
  | Full that sources the candidate pool | 7.7-33x | 22-88x |
  | Reindex over the pool | 2.2-22x | 6.8-22x |

  Every pass wins, so packed rows have no shape rule against the Triton leaf.
  Inside the leaf, a Reindex pass scores the whole history and gathers the pool
  unless the pool-only sparse kernel wins (`_sparse_reindex_wins`): that kernel
  runs in one wave of CTAs and costs about 85 us at any size, while the packed
  history scorers are cheap (dense over sparse time 0.31-0.70 up to the pool's
  16384 rows; sparse wins 1.05-3.5x on paged histories past the pool from 2^21
  scores, and on one request's chunk from four times the pool at 128 queries or
  eight times otherwise). For FP8 rows no pool-only scorer exists on SM120 (the
  sparse kernel reads MX rows only), so `fp8_indexer_wins` still keeps long FP8
  Reindex passes on Triton. TP-sharded index scoring (`index_candidates`) still
  takes FP8 rows only.

## Tensor-core sparse attention on SM120 (`_triton/sparse_mma.py`)

`triton_dsa_mma_decode` and `triton_dsa_mma_prefill` (priority `PERFORMANT`,
window 12.0-12.9) are the SM120 cores for the dense latent cache the runtime
stores (`has_kv_cache=True`, BF16 or E4M3 rows of 512 + 64 channels, page 64,
top-k 512-2051, `kv_lora_rank` 128 or 512, 64-wide RoPE, `return_lse` either
way, `q_len` 1-6 for decode). FlashMLA and the trtllm-gen cores are capped off
SM120. The packed FP8 sparse cache (`has_sparse_kv_cache=True`) has no runtime
user on SM120 (the backend always passes `sparse_kv_cache=None`) and keeps
`triton_dsa_decode` / `triton_dsa_prefill`. The leaves declare
`batch_invariant` and `cuda_graph`, and the `slot_order` trait with both
orders (they reduce the slots as they arrive, so they keep the top-k leaf's
order); they carry no `topk` trait, because a `KernelPlan` leaves `topk`
variable and drops any leaf that restricts it.
A zero RoPE width is the KPool layout of GLM-5.3-Flash (`qk_nope_head_dim` 256,
`kv_lora_rank` 512, 2048 slots plus up to three tail slots, cache rows of 512
channels): the kernel compiles the RoPE gather and score term out, and the
same leaves serve it (29-134x over `triton_dsa_*` at 16 and 64 heads, 1-1024
tokens, 2051 slots; up to 125 TFLOP/s).

Why it is fast: all heads of a token read the same selected rows, so a program
owns a tile of heads of one token and multiplies each gathered tile of rows by
all of them. The portable kernels run one program per (token, head) and
reduce on CUDA cores. Here `S = Q K^T` runs on `mma.sync` (FP8 `m16n8k32` when
query and cache are both E4M3, BF16 `m16n8k16` otherwise) and `O += P V` on
FP16 probabilities for an E4M3 cache (its values are exact in FP16) or BF16
ones, with an FP32 online softmax. One gather of a row serves both products
(the 512 latent channels are both K and V; the 64 RoPE channels only K). The
next tile's loads are issued before the current tile's math: the whole E4M3
tile is prefetched into registers, while a BF16 tile (twice the registers)
prefetches only the next slots and gathers its rows where they are used; a
whole BF16 tile in flight spilled and cost 1.0-1.6x (task 17).

Batch invariance: a token's selection is cut into at most `MAX_CHUNKS` (8)
chunks of whole `BLOCK_N` tiles, a function of the selection width alone
(`num_chunks_for`). Each chunk runs its own online softmax and the chunk states
are folded in order by one helper (`_merge`, explicit `fma`). One program folds
them in registers; when a batch has too few (token, head tile) pairs to fill
the GPU, several programs share a token, each storing the state of every chunk
it ran, and a small second kernel folds all stored states with the same helper
in the same order. A token's row (and LSE) is therefore bitwise the same for any
batch size, head-tile width and split count, as for FlashMLA's unsplit sparse
prefill; `test_rows_do_not_depend_on_the_batch` and
`test_split_count_does_not_change_the_result` guard it, and
`test_bitwise_envelope_plans_the_mma_leaves` the plan. This is the scheme of the
V4 leaf (`dsv4/README.md`).
One path with parameters: `plan_launch(tokens, heads, num_sms)` picks the
head-tile width (16 or 32) and `num_splits`, the number of programs that share
one (token, head tile) and each take a run of chunks (at most one per chunk).
The launch parameters depend on the token count only, so a graph captured per
batch size replays fixed values. Token count, head count, top-k width, strides,
chunk count and split count are runtime arguments (`do_not_specialize`); the
binaries are keyed by tile width and split-or-not, which
`test_batch_size_does_not_recompile` guards.

Numerics match the portable kernels: output BF16 for E4M3 queries (the query
dtype otherwise), `k_scale` folded into the softmax scale, natural-log FP32
LSE, `-1` slots and `topk_lens` masked, all-invalid rows zero with
negative-infinite LSE. Against float32 attention over the stored values the
LSE is exact to 2e-5 absolute, and the output is within one BF16 ulp.

Measured on an RTX PRO 6000 (CUDA graph, 128 heads, top-k 2048, history 16K
per request for decode and 32K for prefill), against `triton_dsa_*`: decode
batch 1/8/32/128 is 39x/76x/97x/129x faster for E4M3 (13.5, 46, 136, 537 us),
prefill chunks 128/512/2048/8192 is 108x/115x/113x/115x (0.48, 1.8, 7.4,
29.5 ms); 16, 32 and 64 heads per GPU gain 27-146x on every bucket. The
tile is within about 4x of the tensor-pipe roofline (E4M3 `QK` at 779 TFLOP/s
and 16-bit `PV` at 404 TFLOP/s, as cuBLAS reaches on this GPU): 136 TFLOP/s
for 128 requests of decode and 158 TFLOP/s for 8192-token prefill (before
batch invariance; see the cost below). Ablating the
`QK` product alone removes 40-45% of the time. Tried and rejected, all slower
on 128 requests of 128 heads (shipped: 540 us): a compiler-pipelined
`tl.range` loop (640-739 us over 4 or 8 warps and 1-3 stages), 16 warps with
a 128-register cap (812 us), `BLOCK_N` of 16 (797 us), and register caps at 8
warps (1547 us at 168, 2045 us at 128; the kernel used 246 registers with no
spills, so a cap forces spilling). `BLOCK_N` of 64 was slower under every cap
tried (2750-3711 us). The CuTe DSL cores below replace it where they are built.
Cost of batch invariance (same 64 buckets, CUDA graph): the chunk states need
a second accumulator, so the 32-head tile now holds 255 registers and spills 34
(E4M3) to 52 (BF16) of them (BF16: 4 since the slot-only prefetch above, which
made BF16 1.24x faster on geomean and up to 1.64x at 32-128 heads; E4M3
unchanged). The kernel is 1.21x slower (geomean) and 1.86x at
worst; against `triton_dsa_*` it is still 22-114x faster on every bucket (72x
geomean; 86x before). E4M3 prefill costs 2-9% (up to 35% for 128-token chunks
of 64-128 heads), BF16 prefill 5-9% at 16 heads and 28-37% at 32-128 heads,
decode of 8-128 requests 0-49% (E4M3) and 0-36% (BF16), and decode of one
request takes 17-24 us instead of 10-14 us (8 chunks bound the programs of one
token; the old split went to 16). Tried: 16 chunks (prefill the same, one
request unchanged, but 1.4-2.2x slower for 8-128 requests of 64-128 heads from
the stored chunk states), 4 chunks (1.32x geomean, 3.6x at worst), and 16 warps
(slower, 58-68 spills at 128 registers). Task 17 also tried, without a gain:
keeping the running accumulator in an L2-resident scratch (no spills, but the
fold traffic made E4M3 13% slower), one flat tile loop with the fold at chunk
ends (more spills), a `tl.range` pipeline (slower), and 16 chunks (0.6-0.7x for
one request, 1.1-1.5x slower for 32-128 requests from the stored states).

### CuTe DSL cores (`_cute_dsl/sparse_mma.py`)

`cute_dsl_dsa_mma_decode` / `cute_dsl_dsa_mma_prefill` (solution `cute_dsl`,
`PERFORMANT + 1`) take the contract, traits and features of the Triton mma
leaves and outrank them on SM120 wherever the CuTe DSL imports. A query and
cache of different dtypes (which the backend never sends) runs the Triton core;
the choice depends on the dtypes only. Design:

* A CTA owns 16 or 32 heads of one token; the selected rows of a tile (64 for
  E4M3, 32 for BF16) are gathered with `cp.async` into a two-stage ring, so one
  tile is in flight while the previous one is multiplied.
* Warps split the output channels: each of four warps per 16-head group
  multiplies a quarter of the 512 latent channels in `O += P V`, so a warp's
  FP32 accumulator is 16 x 128 (one accumulator plus the running fold). V
  fragments come from `ldmatrix.trans`; E4M3 values are converted to FP16
  with `cvt.rn.f16x2.e4m3x2` (exact), BF16 values go to the MMA as loaded.
  The V4 variant also runs a 16-head CTA in 8 warps (`cwarps`; the output
  channels split 8 ways, the score rows between the two warps of a channel
  quarter, bitwise the 4-warp result; `dsv4/README.md`).
* Scores: for E4M3 the query tile sits in shared memory and each warp scores
  16 of the 64 rows over all 576 channels (`ldmatrix` + `m16n8k32`); rows are
  unpadded and XOR-swizzled in 16-byte units, so the CTA fits 99 KiB with no
  bank conflicts. For BF16 a 64-row tile and the query tile do not fit, so each
  warp keeps its quarter of the query channels in registers and the four
  partial score tiles are added in a fixed order.
* FP32 online softmax in registers; the probabilities become the PV A
  fragments directly. The chunked fold and the split combine are the Triton
  scheme (chunks by width, ordered fold with explicit `fma`, the same helper in
  both kernels), so rows are bitwise the same for any batch, head tile or
  split.
* Stream-K launch. The grid has one row of blocks per whole token (a CTA per
  head tile, chunks folded in registers) followed by `num_splits` rows per split
  token (each CTA a run of chunks, states stored, folded by the combine
  kernel); the hardware hands CTAs out in grid order, so the short split CTAs
  fill the SMs that the last wave of whole CTAs leaves idle.
  `plan_launch(tokens, heads, num_sms, fp8)` returns `(block_h, num_splits,
  full_tokens)`. A CTA fills an SM, so one CTA more than SMs costs a whole
  second wave (BF16, 32-head programs: 188 take 149 us, 189 take 287 us). The
  base plan splits every token alike: the split (1, 2, 4 or 8 CTAs per token)
  with the fewest chunk times, `waves x chunks per CTA`, taken only if it needs
  1.25x (16-head CTAs) or 1.5x (32-head CTAs) fewer than unsplit; 32-head CTAs
  start once they alone fill the SMs. A stream-K plan (whole tokens first, a
  tail of 1/8-2 waves split 2, 4 or 8 ways) replaces it only if a wave model
  (unsplit CTA 115 / 138 us BF16, 62 / 81 us E4M3 at 16 / 32 heads, +4 us per
  split CTA, +4 us for the combine) puts it 8% ahead. Against the uniform plan
  alone, over 140 decode shapes of 1-192 tokens at 16-128 heads and both
  dtypes: 1.032x geomean (BF16 1.034x, E4M3 1.029x), up to 1.38x, worst 0.97x;
  64 buckets (16/32/64/128 heads, decode 1/8/32/128, prefill 128-8192) 1.017x,
  worst 0.986x, up to 1.30x (64 heads, 128 tokens: 291 -> 216 us BF16). The
  gain is at token counts a little past a whole number of waves, where the old
  plan ran a second wave or split every token. Rows and LSE are bitwise equal
  across plans (`test_cute_rows_do_not_depend_on_a_stream_k_schedule`,
  `test_cute_split_count_and_head_tile_do_not_change_the_result` over every
  `full_tokens`). Whole and split tokens are separated by token, not by head
  tile, because the state store of the first stream-K kernel took one more
  live register and cost the E4M3 core 3-6%; keeping the token-indexed store
  keeps the one-request decode at 20.2 us BF16 / 13.3 us E4M3 (was 20.3 / 13.1).
* Binaries are keyed by dtype, head tile, rank and RoPE width; token count,
  heads, width, strides, chunk and split counts are runtime arguments
  (`test_cute_batch_size_does_not_recompile`). Compilation happens on first
  use, which the CUDA-graph warm-up runs cover.

What bounds the BF16 core (task 24 ablations, 128 heads, 2048-token prefill,
6.1 ms, BF16 CuTe core): removing the whole partial-score exchange (store,
barrier, sum) gives 5.1 ms, every MMA 4.9 ms, the score MMAs alone 4.9 ms, the
PV MMAs alone 5.3 ms, the gathers alone 6.0 ms (no change), gathers + exchange
+ MMAs 2.6 ms. So the exchange is 17%, not the cause of the 48%: the CTA is
bound by a chain per tile (wait, scores, exchange barrier, softmax, PV) that
two warps per scheduler cannot overlap, and by the L2 -> SM gather (4.8 ms
floor with `cp.async`, 3.2-4.0 TB/s). Tried and lost: a bulk copy per row
(`cp.async.bulk` + mbarrier, zero-fill of absent rows by the owner thread) has
a 3.8 ms gather floor but the whole kernel is 1.12x slower (6.85 vs 6.13 ms);
a 1-stage ring with 2 CTAs per SM (V4, 16 heads, 8 warps) is 0.55-0.82x of the
Triton leaf (the 2-stage 8-warp CTA: 0.76-1.10x); 16 chunks
per selection (a CuTe-only chunk count) makes one-request decode 1.3-1.46x
faster (BF16 20.3 -> 14.2 us, E4M3 13.1 -> 9.8 us at 16-32 heads) but costs
0.8-0.9x at 4-64 tokens, where every chunk state is stored, so it cannot be
chosen without following the batch. Task 30 repeated this with a brute-force
plan per shape (block_h x splits x whole tokens) for both chunk counts: 16
chunks are 1.31x (BF16) / 1.23x (E4M3) faster at 1-4 tokens and 0.95x / 0.91x
at 13-64 tokens (worst 0.85x / 0.81x), 1.06x / 1.01x geomean over the 140
shapes. A split CTA costs 7-10 us (state store at the end of its chain plus the
combine; BF16 32 heads: 2 / 4 / 8 splits are 79 / 43 / 25 us against 138 / k)
and the extra states are L2 / DRAM traffic comparable to the gathers at 64
tokens x 128 heads. A persistent CTA that prefetches the next token's query
and first tile has no work to chain at one request (8-16 CTAs) and at most
3.4-5% to gain elsewhere: a launch whose selections are all empty (launch,
query load, store, combine) takes 17 of 441 us (BF16, 128 heads, 128 tokens)
and 8 of 248 us (E4M3); BF16 holds the query in 32 of its 255 registers and
E4M3 stages it in 18-36 KB next to the 99 KiB ring, so there is nowhere to
prefetch it. Keeping chunk states on chip across a run is not bit-exact: the
fold is a left fold, so a CTA that owns chunks 4-7 needs the running state of
chunk 3, and a chain of CTAs through global memory or DSMEM pays one hop per
CTA against the 2 us combine. Prefetching the combine's chunk loads (1-8
states ahead) gained nothing. A 3-stage ring or a key split for BF16
needs the query next to the ring in 99 KiB, which does not fit.
Measured (same 64 buckets, CUDA graph, `bench_dsa_mma_sm120.py`), CuTe against
the Triton core: E4M3 1.91x geomean (1.49-2.34x), BF16 1.16x (1.00-1.42x), no
bucket slower; against the Triton core before task 17 1.92x / 1.44x, against
the one before batch invariance 1.62x / 1.12x (BF16 decode of one request is
20 us, 1.5x the pre-invariance 13-14 us: 8 chunks of 32-row tiles bound its
CTAs). 128 heads: E4M3 decode batch 1/8/32/128 13.0/30.2/80.6/238 us, prefill
128/512/2048/8192 0.24/0.87/3.47/13.7 ms (342 TFLOP/s, 63% of the tensor-pipe
floor); BF16 decode 20.4/45.3/140/421 us, prefill 0.42/1.55/6.13/24.3 ms
(192 TFLOP/s, 48% of the 16-bit floor). Ablation on E4M3 prefill 2048: removing
every MMA saves only 9% and removing the gathers 13%, so the rest is the
per-tile softmax and fragment work of two warps per scheduler; 64-row tiles and
lane-constant swizzled addresses took it from 5.1 to 3.5 ms. Tests: the shared
`test_dsa_mma_sm120.py` cases run for both solutions (float32 reference,
bitwise rows across batches, any width, zero RoPE), plus split and head-tile
invariance, recompile guard, `out=` and cache layouts, and the mixed-dtype
route. The V4 prefill and MLA decode CuTe leaves reuse this kernel class
(`dsv4/_cute_dsl/`, `mla/_cute_dsl/`). Still open: the BF16 path at 42-48% of
the 16-bit floor (channel-split partial scores; a register-resident query with
a key split as in E4M3, or a smaller ring, are untried), and decode of one
request, bound by its 8 width chunks (20 us BF16 / 13 us E4M3: 5.3 / 4.6 us
of launch, query and combine, then about 1.9 / 2.2 us per tile of the chain). Sharing gathered rows across
a 2-CTA cluster over DSMEM is ruled out on SM120: DSMEM pulls run at about
33 GB/s per CTA (DeepGEMM `deepgemm-analysis.md` 9.28). Sharing them between
tokens of one CTA is bounded for V4 at 16 heads (`deepgemm-analysis.md` 9.35).

Strided caches. A core that reads `kv_cache` in place through its own page and
token strides declares `STRIDED_KV_CACHE_FEATURE` (`strided_kv_cache`; the
Triton portable and mma cores and both CuTe cores do). Cores that reshape the
cache into flat rows would copy a strided view in full on every call, so a
caller whose cache is such a view requires the feature through
`dsa_decode` / `dsa_prefill(features=...)` and the kernel plan; dots3-note's
compact latent cache does (`Dots3NoteDSABackend.core_features`) instead of
pinning a solution, so SM120 reaches the CuTe cores and other platforms keep a
core that does not copy.
The portable Triton attention implementations support `return_lse=True` for
both dense and packed latent caches. They return `(output, lse)`, with FP32
natural-log LSE shaped `[tokens, heads]`; all-invalid rows produce zero output
and negative-infinite LSE. This permits exact softmax-weighted merging of
context-partitioned sparse attention. Supplying `out` preserves the supplied
output buffer even when returning LSE. Omitting `topk_lens` uses the full padded
slot width, with negative slots still excluded.

DeepGEMM index scoring registers for 16, 32, or 64 index heads. The
16-head case pads queries and weights to its native 32-head ABI with zeros;
caller-provided scoring scales and forced initial/local candidate policies
remain unchanged. Kernel traits exclude other head counts.

FlashMLA sparse prefill (regular BF16 KV) and sparse decode (packed FP8 KV)
also support `return_lse=True`. Both return natural-log LSE `[tokens, heads]`
and normalize empty rows to zero output / negative-infinite LSE for the shared
DCP reduction. `topk_lens` masks excluded columns before dispatch, and supplied
output buffers retain their identity. Decode transposes the vendor's
`[batch, heads, query]` LSE to token-major order. This does not reinterpret
regular BF16 KV as packed FP8 or change its Triton decode selection.

## Runtime backend selection

The runtime selects DSA with `--attention-backend dsa`, or automatically from
the DSA architecture. The same name resolves to `AscendDSABackend` on Ascend
and `DSABackend` on GPU platforms. The Ascend implementation owns its device
metadata and operator adaptation; model projections and indexer selection
parameters remain model-owned. Its current constraints are BF16 KV and index
keys, 128-token kernel pages, target verification of at most eight tokens,
and no draft-model attention. Target verification can still consume draft
candidates. The Ascend backend checks these constraints.

## Sharded Index-K candidates

`dsa_index_candidates` scores a bounded query tile against rank-local Index-K
for prefill and decode. Inputs are a position-preserving page table (`-1` for
absent pages), request IDs, and global causal lengths. Outputs are global logical
offsets and FP32 scores: invalid candidates use `-1`/`-inf`, and forced
initial/local candidates use `+inf`. Cross-rank merging belongs to runtime DCP.
Runtime setup validates a positive power-of-two candidate count when the KVP
group has multiple ranks, using the cache-parallel topology (`kvp_group` /
`kvp_size`). Single-rank selection keeps its existing top-k choices.

DeepGEMM compacts owned, causally visible pages and maps results back to global
offsets, masking padding and handling partial tails and empty shards. Query
quantization and head padding match the unsharded path; portable Triton provides
the same interface. Query tiling bounds scratch memory, and fixed-shape GPU
metadata supports CUDA graph replay without host reads.

On SM120 the DeepGEMM leaf selects each row's candidates with DeepSelect over
the row's own compact length (the first `lengths` columns; a short row yields
all of them, then `-1`), not `torch.topk` over the padded row. At the logits a
DCP rank produces (top-k 2048, 64 heads, 1/4 of the pages owned) it is faster
than `torch.topk` at every shape (2-8x from 128 rows) and than the TRT-LLM
cluster runner from 128 rows or at 1/4 ownership; the runner is 1.1-1.7x faster
only for full-length rows (no sharding) with up to 32 rows of 16K+ candidates,
which a sharded rank does not see, so there is no rule. The whole leaf over
1-2048 query rows, 16K-128K contexts and 2/4/8 ranks is 1.25-4.7x faster than
the `torch.topk` leaf (geomean about 2) and 3-64x faster than the Triton leaf
(`scripts/sm120_bench_index_candidates.py` in DeepGEMM).

The Triton FP8 indexer skips Q/K scoring for tiles with no causally visible,
owned entries. Empty tiles still write negative-infinite logits, including
when lengths or ownership change during graph replay. Partially valid tiles
keep the per-entry mask and global forced-window policy. This avoids redundant
arithmetic without changing the launch grid, logits capacity, or Top-K scan.

### Persistent CUDA top-k dependencies

`ragged_decode_topk` needs no workspace and requires an explicit `enable_pdl`.
With PDL, selection waits before reading lengths or scores, including empty and
short rows. Cluster and single-CTA launches both honor the PDL flag.

For non-cluster PDL grids with no more rows than SMs, the launcher selects a
one-block register budget to reduce packing of waiting CTAs. Larger grids keep
the two-block budget. Launch bounds guide register allocation; they are not a
hard limit on resident CTAs. Both variants use the same radix algorithm.

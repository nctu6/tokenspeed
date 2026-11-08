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
- **V4.1 `dsv41_index_topk`.** DeepGEMM has no paged FP4 MQA kernel on SM120, so
  the packed-FP4 leaf (`deep_gemm_dsv41_index_topk`) and the upstream DeepSelect
  leaves stay on SM100/SM103. SM120 scores the FP8 `index_v4` rows like Hopper
  does (`deep_gemm_sm120_dsv41_*`, registered by builds with SM120 kernels) and
  selects with DeepSelect. The sparse candidate scorer is Hopper-only, so a
  Reindex pass scores the whole history; `fp8_indexer_wins` keeps it on the
  Triton scorer, which reads only the candidate pool, where the history is
  long (over 16384 rows and more than 2^19 row-history pairs). The leaves are
  selected only for 132-byte `index_v4` rows. SM120 still stores the 68-byte
  V4.1 rows (`cache_format="v41"` in the runtime attention config), because the
  `v4` rows hold about 45% fewer tokens in the same memory; the config does not
  select `v4` on SM120 yet, so serving reaches the native route only once it
  does. Measured on an RTX PRO 6000 (32 heads, top-k 512, CUDA graph), Triton
  time over native time is 2.3-7.2x for decode Full passes over 1K-32K rows and
  1-128 requests, and 1.4-5.1x for chunked-prefill Full passes of 128-4096
  queries.

## Tensor-core sparse attention on SM120 (`_triton/sparse_mma.py`)

`triton_dsa_mma_decode` and `triton_dsa_mma_prefill` (priority `PERFORMANT`,
window 12.0-12.9) are the SM120 cores for the dense latent cache the runtime
stores (`has_kv_cache=True`, BF16 or E4M3 rows of 512 + 64 channels, page 64,
top-k 512-2051, `kv_lora_rank` 128 or 512, 64-wide RoPE, `return_lse` either
way, `q_len` 1-6 for decode). FlashMLA and the trtllm-gen cores are capped off
SM120. The packed FP8 sparse cache (`has_sparse_kv_cache=True`) has no runtime
user on SM120 (the backend always passes `sparse_kv_cache=None`) and keeps
`triton_dsa_decode` / `triton_dsa_prefill`, as do batch-invariant plans (the
leaves do not declare `batch_invariant`: the split count follows the batch).
A zero RoPE width is the KPool layout of GLM-5.3-Flash (`qk_nope_head_dim` 256,
`kv_lora_rank` 512, 2048 slots plus up to three tail slots, cache rows of 512
channels): the kernel compiles the RoPE gather and score term out, and the
same leaves serve it (29-134x over `triton_dsa_*` at 16 and 64 heads, 1-1024
tokens, 2051 slots; up to 125 TFLOP/s). The split count of these leaves follows
the token count, so a token's prefill row depends on its batch; the V4 leaf
(`dsv4/README.md`) shows the fixed-chunk scheme that removes this, not yet
applied here.

Why it is fast: all heads of a token read the same selected rows, so a program
owns a tile of heads of one token and multiplies each gathered tile of rows by
all of them. The portable kernels run one program per (token, head) and
reduce on CUDA cores. Here `S = Q K^T` runs on `mma.sync` (FP8 `m16n8k32` when
query and cache are both E4M3, BF16 `m16n8k16` otherwise) and `O += P V` on
FP16 probabilities for an E4M3 cache (its values are exact in FP16) or BF16
ones, with an FP32 online softmax. One gather of a row serves both products
(the 512 latent channels are both K and V; the 64 RoPE channels only K). The
next tile's gather is issued before the current tile's math (register
double-buffering; `num_stages=2`).

One path with parameters: `plan_launch(tokens, heads, num_sms)` picks the
head-tile width (16 or 32) and `num_splits`, the number of programs that share
one (token, head tile) and each take a slice of the selected columns. Few
tokens split up to 16 ways, and a small second kernel merges the partial
(max, sum, accumulator) states; one split normalizes in place. The launch
parameters depend on the token count only, so a graph captured per batch size
replays fixed values. Token count, head count, top-k width, strides and
split count are runtime arguments (`do_not_specialize`); the binaries are
keyed by tile width and the combine kernel's power-of-two split bound, which
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
for 128 requests of decode and 158 TFLOP/s for 8192-token prefill. Ablating the
`QK` product alone removes 40-45% of the time. Tried and rejected, all slower
on 128 requests of 128 heads (shipped: 540 us): a compiler-pipelined
`tl.range` loop (640-739 us over 4 or 8 warps and 1-3 stages), 16 warps with
a 128-register cap (812 us), `BLOCK_N` of 16 (797 us), and register caps at 8
warps (1547 us at 168, 2045 us at 128; the kernel uses 246 registers with no
spills, so a cap forces spilling). `BLOCK_N` of 64 was slower under every cap
tried (2750-3711 us). A CuTe DSL kernel with TMA or `cp.async` gathers
and a warp-specialized softmax is the next step if this becomes the bottleneck;
it is not written.

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

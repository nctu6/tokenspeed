# MoE ops

Per-op notes for `ops/moe/`.

## Deferred route outputs

`bf16_moe_routes` returns individual, unweighted expert outputs. Deferred MXFP4
execution also returns a map from each original route to its output row, with
`-1` for remote experts. Consumers normalize each route before weighting and
summation. Combining routes before normalization changes the operation.

The solution modules register these operations. Shared Triton helpers live in
`_triton/`, and the CuTe DSL routed reduction lives in `_cute_dsl/`. They use
the same registry and public dispatchers as the other MoE operations.

## Expert placement dispatch

With redundant experts (`--ep-num-redundant-experts R`), a logical expert has
several physical replicas across the EP ranks, and routing must emit physical
ids. Under replicated-input EP, every rank runs the same routing over the same
all-gathered tokens, so the replica choice has to be a pure function of the
token row and the route rank for exactly one rank to own each (token, expert)
pair: `replicas[logical, (row + rank) % count]`.

`ExpertDispatch` holds one layer's tables (`replicas [num_logical, X]` int32,
`-1` padded to a fixed width; `num_replicas [num_logical]` int32). The runtime
keeps them as views of the placement's device tables so an in-place
rebalance reaches captured graphs without reallocating.

`dispatch_topk_ids(topk_ids, dispatch)` is the production implementation of
the replica choice: two gathers and a modulo over `[tokens, top_k]`, in
tensor ops, graph-capturable. Every id must be a real expert in
`[0, num_logical)`; the runtime masks zero experts and padded slots around
the call and counts the mapped ids into its load counters afterwards.

Follow-up: a fused Triton kernel for the same contract (one launch instead of
four tensor ops per MoE layer per forward). Its arguments must follow the
compile-time parameter rule: token count, top-k, and table width are runtime
arguments or bucketed, never `constexpr`.

## FlashInfer adapters (`flashinfer.py`)

### Opt-in SM90 block-FP8 grouped MoE

`prepare_sm90_fp8_moe(mode=...)` prepares a callable before warmup and graph
capture. Choose `default` for upstream FlashInfer, `expected_m` for the per-call
hint backport, or `optimized` to also enable expert-tile prefix sums and direct
tile-to-expert lookup. Both private modes accept independent pipeline stage
caps for FC1 and FC2.
Existing `moe_plan` selection stays unchanged.

```python
from tokenspeed_kernel.ops.moe import prepare_sm90_fp8_moe

expert_apply = prepare_sm90_fp8_moe(
    mode="optimized", max_num_stages_fc1=4, max_num_stages_fc2=-1
)
# Each omitted stage bound defaults to -1 and keeps upstream selection.
# Pass the usual FlashInfer CUTLASS MoE arguments plus:
# expert_apply(..., expected_m=expected_tokens_per_expert)
```

Private modes apply `sm90_moe.patch` to a cached source copy of FlashInfer
0.7.2 and compile isolated modules, operators, and tuning caches. They require
matching upstream sources, JIT enabled, and both CUDA toolkit and runtime
12.8 or newer. Initialization reports unsupported sources or build settings.
The installed FlashInfer package is unchanged. To switch settings, prepare a
new callable, warm it up, and recapture graphs.

`expected_m` is a positive uint32 hint for both expert GEMMs in one call;
omitting it preserves the native estimate. For complete local expert groups,
use `max(1, ceil(local_rows * top_k / local_experts))`; replicated-input EP
must account for remote routes using the global expert count. Routing,
workspace bounds, and the runner's stored estimate are unchanged.

`max_num_stages_fc1` and `max_num_stages_fc2` independently bound the two
expert GEMMs. Each defaults to `-1`, preserving upstream stage selection for
that GEMM. A positive value only excludes candidates above its limit, in
both regular and SwapAB layouts. Tile selection, stage candidates, and
shared-memory checks follow upstream logic. Bounds must be at least four,
the backend's minimum candidate depth; zero is invalid. Different bound
pairs use isolated module, operator, tuning, and kernel caches.

Prefix/lookup scheduling supports SM90 DeepSeek block-FP8 SwapAB grouped GEMMs.
The prefix scan handles expert counts independent of block and warp size and
refreshes metadata on every replay. A bounded lookup covers the first tiles,
then falls back to prefix binary search. Configurations whose scheduler
metadata would exceed the shared-memory budget retain the upstream scheduler.
Regression tests compare eager and graph outputs with upstream at
`atol=rtol=0`. Performance depends on shape and routing; the mean hint alone
can regress on concentrated routes, so benchmark the intended settings.

### NVFP4 routing-map padding

The pinned FlashInfer version (0.7.2) initializes unused rows inside active
expert tiles of `permuted_idx_to_token_idx` to `-1` in its routing producers.
This prevents unnecessary activation loads in the first GEMM without a separate
fill-kernel launch. Routing refreshes live mappings and tile padding on every
invocation, including CUDA graph replay and both logits and precomputed top-k
entry points. Allocation slack beyond the active tiles and the launcher's extra
guard entry are not part of this initialization contract.

`thirdparty/flashinfer/trtllm_moe.py` uses the upstream native module while
retaining the private runner for the small-batch tactic policy described
later. Warmup must finish before graph capture.

Regression coverage includes live and padded mapping entries, local expert
partitions, PDL on/off, changing routing within one captured shape, graph replay
after workspace corruption, untouched allocation slack/guard, and output
equality with the upstream operator.
Expert-partition tests retain the loader's global activation input scales while
sharding expert weights, and select SiTU through FlashInfer's activation enum.

### BF16 intermediate sizes that are multiples of 64

FlashInfer's BF16 TRT-LLM launcher (`Bf16MoeLauncher::check_moe`) rejects an
intermediate size per partition that is not a multiple of 128. Its BF16 cubins
need only 64 for gated activations: GEMM1's rows, twice the intermediate size,
are tiled by 128, and GEMM2 reads its K, the intermediate size, in 128-byte
blocks of 64 elements. Non-gated activations still need 128.

`thirdparty/flashinfer/trtllm_bf16_moe.py` builds a private copy of the
installed launcher in which only that check requires 64 for gated and 128 for
non-gated activations. It uses a source-keyed JIT
module, private operator names, and cloned entry points, and refuses a launcher
whose check it does not find exactly once. The BF16 TRT-LLM SiLU/SwiGLU
kernels declare `ispp_alignment` 64 when the adapter applies to the installed
FlashInfer and FlashInfer's JIT can compile the private module, and 128
otherwise, with a warning that names the reason. The ReLU2 kernels in that file
keep 128. The private module is not in FlashInfer's AOT jit-cache, and ninja
rebuilds a module left in the JIT workspace whenever the build changes, so this
needs `FLASHINFER_DISABLE_JIT` unset and FlashInfer's nvcc (`FLASHINFER_NVCC`,
else `bin/nvcc` under its CUDA home). The adapter decides this once per
process at import, without compiling, so kernel selection and layer padding
agree. Only sizes that are not multiples of 128 run on the private launcher;
the first such layer JIT-compiles the whole private TRT-LLM MoE module during
warmup. Other sizes keep FlashInfer's stock module. These layers keep
FlashInfer's BF16 routing behavior. Once the minimum supported FlashInfer
accepts these sizes, remove the adapter.

### FP32 correction bias for in-kernel DeepSeekV3 routing

FlashInfer's grouped DeepSeekV3 router (`routingMainKernel` in
`trtllm_fused_moe_routing_deepseek.cu`, used when `n_group > 1`) casts the
correction bias to the BF16 output type before adding it to the FP32 sigmoid
score, and its tanh-form sigmoid returns 0 for strongly negative logits.
Reference routers such as DeepSeek-V3's add the bias in FP32, and so do
TokenSpeed's top-k routers. A model opts in per layer with
`routing_config["fp32_correction_bias"] = True`; nothing enables it by default.

`moe_plan(fp32_correction_bias=True)` keeps a kernel that routes from logits
only if its `_tokenspeed_fp32_correction_bias` hook returns True, and otherwise
plans precomputed top-k. Only the BF16 SiLU/SwiGLU TRT-LLM kernel that routes
from logits has the hook; the BF16 ReLU2 kernels do not. That hook builds,
when the plan is made, a copy of FlashInfer's TRT-LLM MoE module whose routing
source carries the edit of flashinfer-ai/flashinfer#5557 (FP32 bias,
`1 / (1 + exp(-x))` sigmoid), together with the 64-aligned launcher above when
that applies, under private operator names. Each edit must be found exactly
once in its stock form or in #5557's form; any other source, or a failed
build, logs a warning and the layer plans precomputed top-k. A FlashInfer
whose routing already has both edits is used as is, so this module is no
longer built once the minimum supported FlashInfer includes #5557; remove the
adapter then.

### Qwen3.8 low-batch tactic

For Qwen3.8's 2,560-hidden, 640-intermediate, 512-expert NVFP4 MoE with
TP4/EP4 and top-k 10, the private runner restricts inputs of at most 32 tokens
to FlashInfer's valid tile-32 tactics. If the native launcher does not offer
tile 32 for a profile, all native tactics remain available. Other models and
larger token counts use the full tuner. Only the matching shape gets a separate
tuning-cache key, so a previously cached tile-8 choice cannot bypass this
policy without forcing unrelated shapes to retune. If FlashInfer removes
either tuning hook, changes the cache-key builder signature, or moves runner
construction outside the cloned entry points, the adapter raises an error.

The cache-key builder derives the policy tag from the target profile.
During autotuning, this is the profile that `p.get_opt_shapes()` selects; during
serving it is the bucket matched to the request. FlashInfer passes caller
tensors when checking a profile, but synthesized tensors when storing its
winner, so the cache-key builder must not compute token-dependent tags from
those tensors. A scoped hook on FlashInfer's shared cache-key builder adjusts
only the private runner's keys. Other runners, cache persistence, and
measurement remain unchanged. Different token profiles continue to store
independent winners.

This is a temporary workaround for FlashInfer 0.7's MoE tactic selection.
When upstream tuning handles the full decode graph, remove it.

The policy targets CUDA-graph latency across routing, shared experts, and MoE
GEMMs. On the measured BS1 MTP3 graph, FlashInfer 0.7's isolated-kernel tuner
selected tile 8, leaving an idle interval before routing. This policy keeps
tile 32 available for that shape while retaining upstream routing and GEMM
implementations.

### NVFP4 squared ReLU

Nemotron-H experts use a non-gated `relu(x)**2` activation. GEMM1 holds only
the up projection, so `w13` is `[E, I, H]` rather than `[E, 2I, H]`. The weight
preprocessor skips the gate/up half swap and permutes with
`is_gated_act_gemm=False`. FlashInfer tiles non-gated GEMM1 rows by 128, so
the kernels declare an `ispp_alignment` of 128 instead of 64.

The kernel applies `output1_scale_gate_scalar` to the GEMM1 accumulator before
squaring and `output1_scale_scalar` after. Squaring is not linear, so the
GEMM1 dequant `a13 * w13_scale_2` must go in the first scale and the second
scale carries only the GEMM2-input requant `1 / a2`. This is the SiTU recipe.
The SwiGLU recipe folds the up-half dequant into the second scale, which would
be wrong here. The kernel test checks non-unit input scales against a
dequantized reference, and checks in-kernel sigmoid-plus-bias routing with 512
experts and top-22.

### Unquantized CUTLASS MoE workspace

`flashinfer_cutlass_unquant_moe_apply` hands `cutlass_fused_moe` one
persistent scratch buffer per device (`cutlass_unquant_moe_workspace`),
sized through `cutlass_fused_moe_workspace_size`, zero-filled when it is
allocated or grown, and reused across layers and calls. Left to allocate its
own scratch per call, the SM90 chain read bytes it never wrote: for some
(EP rank, routing) combinations the rank's routed output was NaN for finite
inputs, reproducibly for that call, while the identical call with any
caller-provided buffer matched the FP32 reference. Stream ordering serializes
consecutive MoE layers through their activations, so one buffer per device
is never in use by two calls at once; a model that overlapped two MoE calls on
separate streams would need one buffer per stream. PDL stays off in this
chain for the race described in the BF16 CUTLASS section of `flashinfer.py`.

### FP8 per-channel experts

The FP8 per-channel TRT-LLM section of `flashinfer.py` runs `fp8_channel`
experts (compressed-tensors FP8 dynamic W8A8: one FP32 scale per weight row,
activations quantized per token) on FlashInfer's
`trtllm_fp8_per_channel_scale_moe`. The kernel reads the per-row scales in the
weights' own resident row order, so the weight preprocessor moves every scale
with its row: the `[gate | up]` to `[up | gate]` swap, the gated-activation
interleave, and the epilogue-tile shuffle. Its per-expert output scalars stay 1.
Each rank pads its own intermediate partition to a multiple of 64 with zero rows
and columns whose scales are 1, so a rank's channels do not move and the padding
adds exactly zero, also to the per-route FC2-input quantization. Partitions that
are a multiple of 64 need no padding: on GB200 the kernel output for a multiple
of 64 is bit-identical to the same experts padded to the next multiple of 128.
The FC2 input is quantized per routed row by TRT-LLM's `perTokenQuantization`
(scale `max(amax / 448, 1 / (448 * 512))`, codes `x * min(448 / amax, 448 *
512)` in FP32, rounded to BF16 and then to E4M3, `amax` over the BF16 row). The
cubins need `usePerTokenSfA` and `usePerTokenSfB`; they are not among the BF16
and NVFP4 MoE cubins, so an offline cache must hold them.

The kernel takes precomputed routes only as BF16 weights packed with the
expert ids. With FP32 route weights the routed kernel stops before its
finalize and `moe_finalize_fuse_shared` combines the per-route outputs with the
FP32 weights, accumulating in FP32 and rounding once to BF16.

## FP8 per-channel experts

Per-channel FP8 experts store weights in FP8 E4M3 with FP32 scales per output
channel and quantize activations per token. FP32 routing weights remain FP32
during combination. Native FP8 matrix arithmetic uses FP32 accumulators but
need not match sequential FP32 arithmetic.

## Shared MoE precision policies

The unquantized Triton MoE normally keeps GEMM1 accumulators in FP32 through
SiLU and combines route outputs with FP32 weights without intermediate rounding.
Existing callers retain that behavior.

An execution plan with `precision_policy="bf16_route_accum_fp32"` and an explicit
positive finite `output_scale` requests these distinct boundaries:

1. Round each GEMM1 gate/up projection to BF16; compute SiLU and multiplication
   in FP32, then store the intermediate in BF16.
2. Accumulate GEMM2 in FP32 and store each expert output in BF16.
3. Multiply each expert output by its unscaled FP32 route weight, then round
   each weighted route to BF16.
4. Sum those routes in ascending expert-ID order in FP32, multiply by
   `output_scale` in FP32, and only then store the final BF16 output.

The result is rank-local: any tensor-parallel reduction is owned by the caller
and occurs after this final BF16 boundary. Scaling an already-finalized BF16
sum, or scaling routing weights before the per-route rounding, is not equivalent.
Routing must independently implement the caller's selection and normalization
contract. The expert kernel does not renormalize or rescale route weights.

The runtime's existing `MoELayer` accepts this policy through `routing_config`
keys `precision_policy` and `output_scale`. It requires unquantized BF16 SiLU,
precomputed FP32 top-k weights, unit `routed_scaling_factor`, no expert bias, and
no expert-parallel/all-to-all mode. Automatic backend selection chooses the
shared Triton BF16 implementation. An explicitly requested unsupported backend
fails closed; CUTLASS is not declared equivalent to these rounding boundaries.
No new scheduling or model execution path is introduced.

This correctness-first policy sorts the small selected-expert list before the
existing dispatch. It is not a performance claim; check its numerics and graph
replay before using its results as an optimization baseline.

## Vendor policy `bf16_vendor_fp32_route`

`precision_policy="bf16_vendor_fp32_route"` runs FlashInfer's TRT-LLM BF16
experts (`--moe-backend flashinfer_trtllm`) with the DeepSeekV3 correction bias
added in FP32. It is a different arithmetic contract, not an equivalent of
`bf16_route_accum_fp32`:

| Step | `bf16_route_accum_fp32` | `bf16_vendor_fp32_route` |
| --- | --- | --- |
| Route weights | Unscaled FP32 | Routed scale applied, rounded to BF16 for the kernel |
| GEMM1 gate/up | Rounded to BF16 | Not rounded |
| Weighted route | Each route rounded to BF16 | Not rounded |
| Combine | Ascending expert ID, FP32, times `output_scale` | Vendor slot order, FP32 |
| Output scale | `output_scale` | Must be 1 |

`MoELayer` accepts it for BF16 unquantized SiLU experts with DeepSeekV3
routing, without expert bias, all-to-all, `--moe-combine-order slot`, or an
output scale. The policy asks `moe_plan` for `fp32_correction_bias`: the plan
routes inside the kernel (`flashinfer_trtllm_unquant_moe_apply`) when the
kernel's hook builds FlashInfer's routing with the FP32 bias
(`thirdparty/flashinfer/trtllm_bf16_moe.py`), and otherwise plans precomputed
top-k (`flashinfer_trtllm_unquant_routed_moe_apply`), where the caller's
router adds the bias in FP32. Under expert parallelism, the layer plans
precomputed top-k. In the kernel, `routing_config["routed_scaling_factor"]`
is applied to the BF16 route weights; on precomputed routes the caller folds
it into the FP32 route weights (Beam: in its route epilogue). The intermediate size per partition must be a
multiple of the kernels' `TRTLLM_UNQUANT_ISPP_ALIGNMENT` (64 where FlashInfer's
launcher is relaxed, else 128); the policy refuses a size the layer would pad.
The kernel finalizes its output itself; deferred finalize is refused.

The two routings are not byte-identical: the kernel computes
`fl32(fl32(s * scale) / fl32(sum + 1e-20))` with its fast-math division, the
precomputed path `fl32(fl32(s / fl32(sum + 1e-20)) * scale)`
(Beam's route epilogue with the routed scale, the bytes of TopK's fold). An IEEE
CPU emulation of the two orders on Beam's routing shape differs in about 5 of
10^6 BF16 weights. A
model whose ranks must agree on one routing checks the plan's
`support_routing` across its MoE group.

## Beam's grouped route

`beam_route_weights` (`python/tokenspeed/runtime/models/beam.py`) routes each
token with a grouped sigmoid: the FP32 sigmoid of the router logits plus the
correction bias, the sum of the top two scores in each of 8 groups, the top 4
groups, the top 8 experts of the kept groups, their unbiased scores renormalized
(sum plus `1e-20`) and multiplied by the routed scale.

- `grouped_route.torch_grouped_route` is those statements in PyTorch (any device).
- `_triton/router_select.triton_router_route_lanepar` runs them in one launch,
  one row per program on one warp, byte for byte: the FP32 sigmoid with Torch's
  CUDA recipe (libdevice `exp`, inline `add.rn.f32` / `div.rn.f32`, no FTZ), the
  grouped selection with Torch's radix keys, fill order, and trimmed bitonic
  network (`_triton/grouped_select.py`; the tie permutation is the pinned
  Torch's), and the epilogue with the sum in Torch's CUDA K8 order
  `((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))` (`_triton/route_epilogue.py`). Its host
  guard `router_route_rejection` admits Beam's routing shape only (256 experts,
  8 groups, top 2 per group, 4 groups, top 8, contiguous FP32 CUDA tensors, at
  least one row, a scale that rounds to a finite zero or normal FP32). It
  launches as a programmatic dependent where `pdl_enabled()`.

Beam calls the Triton kernel on NVIDIA SM100 to SM103 where the guard admits the
call and the Torch statements otherwise. The GPU tests
(`test/nvidia/ops/moe/test_router_select_cuda.py`) pin Torch's FP32 sigmoid and
K8 sum order first (a mismatch fails, never skips), then compare the two byte for
byte on random rows, tie, edge, and NaN corpora and in CUDA-graph replay.

## Routing weight storage and deferred finalize

`moe_plan` records the selected expert kernel's `topk_weights_dtype`. DeepSeek V4
routing stores its weights in that dtype after FP32 scoring, normalization, and
scaling.

On NVIDIA, the DeepSeek V4 MoE defers finalization for BF16 activations
whenever the selected kernel supports it (TRT-LLM MXFP4 SwiGLU, Triton MXFP4,
and the other deferred-capable kernels). The kernel returns expert rows and row
indices, and `moe_finalize_fuse_shared` combines them with the precomputed
route weights and the shared expert output in FP32, trimming to the original
hidden width. Some kernels (Triton MXFP4) return no weights; `MoELayer` then
returns the precomputed route weights it passed in. FP16 inputs, ROCm builds
without the fused finalizer, and kernels without deferred finalize keep the
kernel's own finalized output.

## MegaMoE on SM120
`deep_gemm_sm120_mxfp4_mega_moe_apply` serves the V4 / V4.1 MXFP4 experts on
SM120 (arch 12.x) for EP 2. It is registered only when the DeepGEMM build
reports `get_mega_moe_max_ranks(12) >= 2` (`deepgemm_sm120_mega_moe_max_ranks`,
behind `deepgemm_supports_sm120`). The fused SM100 leaf
`deep_gemm_mxfp4_mega_moe_apply` stays capped at 10.9 (TMEM, 2-CTA UMMA, NVLink).
`test_sm120_excludes_sm100_only_leaves.py` checks both rules.

SM120 has no NVLink, so DeepGEMM runs MegaMoE there as a sequence of kernels.
The ranks write into each other's torch symmetric-memory buffers with peer
stores over PCIe (CUDA IPC): tokens go out to the owners of their experts, the
owners run local FP8 x MXFP4 grouped GEMMs and SwiGLU, and the expert rows come
back to be summed in top-k order. A system-scope epoch barrier in the buffer
orders the steps, so the call replays in CUDA graphs. Results are bitwise equal
to a single rank holding all experts. Calls that share the buffer must be
ordered on one stream per rank. The SM120 leaf keeps the scale rows in order
(no UTCCP transpose). V4.1's 2304-wide experts are not padded to 2560 there.

MegaMoE stays opt-in: `--moe-backend mega_moe`. Against what the default path
runs at EP 2 on SM120,
Marlin W4A16 plus its all-reduce or all-gather / reduce-scatter, at the same
served tokens (cold weights, CUDA graph, GPUs on one PCIe switch):

| Tokens per rank | V4 | V4.1 |
|---|---|---|
| 1 | 0.95x | 0.96x |
| 2-1024 | 1.00-1.05x | 0.99-1.02x |
| 2048-4096 | 1.30x | 1.28-1.35x |

At EP 2 a V4 rank holds 70 GB of routed experts; V4.1 needs at least 4 ranks.
Details and the per-size table are in DeepGEMM `deepgemm-analysis.md`
section 9.26.

### Token layouts (`moe_plan(ep_token_layout=...)`)

The plan names how the tokens meet the experts, and the leaf runs the same
kernels either way. `ep_token_layout` is a required argument of `moe_plan` (and
of `MoELayer`); `None` means no kernel of the plan owns the token exchange, and
this leaf refuses it:

- `scattered` (attention DP): each rank brings its own tokens. The call is the
  push / barrier / dispatch / GEMMs / return / combine protocol above and returns
  the complete output of this rank's tokens. EP 2 only.
- `replicated` (attention TP): every rank of the MoE group holds the same
  tokens, so nothing is exchanged. The kernel runs as one rank (a one-rank
  process group, no push, no barrier, no return) over this rank's shard of the
  experts and returns a partial, which the caller sums with the all-reduce a TP
  layer already does (`combine_order="rank"`). The shard is either a range of
  experts (`--ep-size 2`; the staged top-k masks the other rank's routes, see
  `stage_mxfp4_mega_moe_inputs`) or a slice of every expert's intermediate
  width (MoE TP 2, the default with `--tensor-parallel-size 2` and no
  `--ep-size`). A slice keeps SwiGLU and the per-32 activation scales inside
  it, so the activations are quantized exactly as unsharded, and the second GEMM
  sums over the slice. V4.1's 2304 splits into 1152 = 9 x 128. Widths of one and
  two ranks are tested; mixed TP x EP is refused.

Real V4.1 (40 layers, first 128 of 384 experts, GPUs 6+7, 2 x 2004 tokens,
prefill graphs off, median of 9, TPOT from 257 tokens):

| Path | Prefill tok/s | TPOT ms |
|---|---|---|
| Marlin TP2 (`auto`) | 15.1K | 10.69 |
| MegaMoE, attention TP2 + MoE TP2 | 17.0K (1.12x) | 10.66 |
| MegaMoE, attention TP2 + EP2 | 16.8K (1.11x) | 11.22 |
| MegaMoE, attention DP2 + EP2 | 19.1K (1.26x) | 14.02 |

The TP2 attention path removes the DP penalty (every DP rank reads the whole
dense model, +2.6 ms) and the barrier / push / return glue, but an EP range
leaves one rank with more routed experts than the other at 2 requests (12
routes over 128 experts), and the all-reduce waits for it: 0.8-1.1 ms per step
against 0.25-0.3 ms for Marlin. MoE TP splits every expert in half, so both ranks
read the same bytes, and the all-reduce wait is back to Marlin's. Its grouped
GEMMs tie Marlin's (3.17 vs 3.3 ms per step) and its glue (staging, dispatch,
gather, SwiGLU, combine, 0.73 ms) is a little under Marlin's (SiLU, align,
reduce, fill, 0.8 ms), which makes TPOT a tie, not a win. `auto` therefore still
selects Marlin: a MegaMoE leaf is selected only for `--moe-backend mega_moe`.
With several ranks the intermediate rows are sized for the worst routing of
the largest batch on every rank (about 2.6 GB at 8192 tokens per rank at V4.1
shapes). DeepGEMM keeps them in one persistent buffer per device, allocated by
the first call and grown by a larger one; the leaf's warmup runs the largest
token count before the KV planner measures memory, so the planner counts the
buffer. Serving on real V4.1 weights (40 layers, 128 experts, GPUs 6+7, 2 x 2K
tokens): MegaMoE DP2/EP2 prefills 1.25x faster than Marlin TP2 (18.1-18.6K vs
14.5-14.8K tok/s) and decodes 1.31x slower (13.9 vs 10.6 ms TPOT): attention DP
reads every dense weight on each GPU (+2.6 ms per step at 2 requests), and the
MegaMoE glue around the grouped GEMMs (barriers, push, dispatch, gather,
SwiGLU, return, combine) adds 2.3 ms per step while the GEMMs tie Marlin
(DeepGEMM `deepgemm-analysis.md` 9.29). A graph timeline of one call at 2
tokens per rank (V4.1 shapes, cold weights, 195 us) puts the glue on the
critical path at about 25 us: 13 us before the first GEMM (input staging 2,
push 2, barrier 3, dispatch 2, gather 4), 2 us between the GEMMs (SwiGLU runs
under the GEMM's PDL tail) and 10 us after the second (return 4, barrier 3,
combine 3). Fusing push + barrier + dispatch + gather and return + barrier +
combine would save about 9 us per layer, 0.36 ms per step (2.6% of the 13.9 ms
TPOT); the GEMMs are 165 us. Not done: the barrier skew between ranks, which
is most of the 0.8-1.05 ms the serving trace shows for the two barriers, does
not shrink with fusion.

## Triton MXFP4 expert offsets
The Triton MXFP4 stage kernels advance every expert base (weights, scales,
biases, route ids) in int64. At the V4.1 shapes (384 experts, hidden 5120,
intermediate 2304) a 32-bit `expert_id * rows * columns` wraps from expert 182
(stage 1: another expert's weights, silently) and from expert 364 (stage 2: an
illegal memory access). `test_triton_mxfp4_large_expert_offsets.py` covers both
boundaries.

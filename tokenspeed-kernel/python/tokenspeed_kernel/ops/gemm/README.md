# GEMM weight preparation

## Small dense projections

`mv` selects the registered small-row projection implementation. Its dispatcher
and registry both use the `("gemm", "mv")` operation. The gfx950 row-CTA GEMV
declares its minimum N and K through the shared shape traits.

On CDNA4, ordinary GEMM dispatch admits the measured single-row
`[1,8192] @ [4608,8192].T` projection alongside the existing multirow cases.
Callers that publish directly into a collective buffer pass that buffer as
`out` to the same `mv` operation.

## Skinny dense BF16 (`triton_mm_bf16_skinny`, NVIDIA SM120)

Dense BF16 `mm` with 1 to 16 rows (a decode step) on SM120 selects
`triton_mm_bf16_skinny` over `torch_mm`. Each CTA owns 16 weight rows and
streams them once over K (tile 128, or 256 for K >= 4096); the activation rows
pad to one 16-row `tl.dot` tile, so every M up to 16 runs the same binary (M
and the row strides are runtime arguments without specialization; N and K
are the model's static widths). Products accumulate in FP32, the bias is added
before the single rounding. Traits: `m` 1..16, N a multiple of 16, K a
multiple of 128, unit inner strides, `NK` weight layout, BF16 output, and a
2-D A (`a_ndim`: `mm` takes `m` from `A.shape[0]`, which is not the row count
of a batched N-D A).

Measured on an RTX PRO 6000 (SM120) over the 6 x 32 bias projections of a
whisper-large-v3 decode step (1.47 GB of weights, one CUDA graph): cuBLAS
1.58 ms at M = 1 (GEMV) and 2.00-2.01 ms at M = 2..16 (16x16-tile tensor-core
GEMM); this kernel 1.41 ms at every M. Other architectures keep their
existing choice; it is not measured there.

## Per-token/per-channel FP8

Per-channel FP8 projections use FP8 E4M3 weights with FP32 scales per output
channel and dynamically quantize activations per token. This scale format is
distinct from block scaling. Native FP8 matrix arithmetic uses FP32
accumulators but need not match sequential FP32 arithmetic.

## DeepSeek V4 grouped output projection

The grouped-projection plan describes the projection geometry and source formats.

After loading and sharding, the operation selects preprocessing using the source
formats and operation traits. The layer runs that preprocessing through the
weight broker, which installs the transformed tensors and records their storage
layout. When no transformation is needed, the layer enrolls canonical storage.

Execution and warmup select kernels using the supplied tensors' recorded layouts,
formats, and operation traits. Participating layers require known storage layouts;
raw canonical callers can explicitly allow unknown storage. This policy is
independent of kernel overrides. Python validation runs during graph capture,
not device-side replay.

## Block-FP8 DeepGEMM (Hopper, SM120)

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged on Hopper; routed experts use their own
backend.

On SM120 (RTX PRO 6000), `deepgemm_sm120_mm_fp8_blockscale` registers only
when the installed DeepGEMM build has SM120 kernels
(`deep_gemm.get_supported_arch_majors()` contains 12). It ranks above the
FlashInfer and Triton block-FP8 leaves, so `auto` selects it: it is faster on
decode-sized M and on par for prefill-sized M.

This optional DeepGEMM integration requires Hopper or SM120, contiguous E4M3 operands
A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible by 128. It
converts canonical FP32 1x128 activation and 128x128 weight scales to the
required layout without UE8M0 rounding (on SM120 the kernel reads the
activation scales in place, so no transpose runs). Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.

## Block-scaled MXFP4 DeepGEMM (SM120)

SM120 (RTX PRO 6000) also exposes the DeepGEMM block-scaled MXFP4 GEMM, which
has no counterpart among the other NVIDIA leaves (the Triton MXFP4 `mm` is
AMD-only). Two leaves register, both only when the installed DeepGEMM build has
SM120 kernels (`deep_gemm.get_supported_arch_majors()` contains 12):

- `deepgemm_sm120_mm_mxfp4` (op `mm`): packed E2M1 activations `A[M,K/2]` and
  weights `B[N,K/2]` as `uint8`, with raw UE8M0 per-32 scales
  `A_scales[M,K/32]` / `B_scales[N,K/32]`. Requires contiguous 2-D operands,
  BF16 output, N divisible by 64, and K divisible by 128.
- `deepgemm_sm120_mxfp4_linear` (op `mxfp4_linear`, weight layout
  `deepgemm_sm120`): the op the dense MXFP4 linear layer calls. The weight keeps
  checkpoint order; `prepare_mxfp4_linear` packs its UE8M0 scales once into the
  int32 MN-major form DeepGEMM reads (four K-group bytes per word, MN aligned to
  4). Each call quantizes the activation with
  `mxfp4_quantize_mn_major_scales`, which writes the packed bytes and the scales
  in that same layout directly, then runs the GEMM. No scale is widened or
  repacked per call. Both leaves are `SPECIALIZED` priority, so `auto` selects
  them on SM120.

`deepgemm_sm120_mxfp4_swiglu_linear` (op `mxfp4_swiglu_linear`) serves gated MLPs
whose gate_up projection stacks the gate rows over the up rows: the same linear,
then `silu_and_mul` over its BF16 output. Both linear leaves declare `n_align=64`
and `k_align=128`, so selection rejects shapes the GEMM cannot tile.

The `mm` leaf takes raw `uint8` scales `[MN, K/32]` and packs them per call with
`pack_ue8m0_scales`; callers that reuse a weight should prefer `mxfp4_linear`.
Packed operands are tagged `int8` (`kPackedFP4`) when passed to DeepGEMM.

Measured on an RTX PRO 6000 as a whole `mxfp4_linear` (activation quantization
included) against FlashInfer's best `mm_fp4` backend plus its own activation
quantization, in a CUDA graph over weights larger than L2, across the
DeepSeek-V3 TP8 dense linears at M = 1..4096: the leaf is faster on 60 of 63
shapes, about 1.23x geomean. The three losses are `q_b` M = 256 (0.90x, the GEMM
itself is slower there) and M = 1 of `o_proj` and `dense down` (0.97x).

The activation quantizer runs as a PDL producer for the GEMM: it triggers the
dependent launch right after its own wait, so the GEMM prefetches weights while
it runs (the GEMM waits for the quantizer before reading activations). Grids
above 1024 blocks launch as a plain kernel instead, because PDL-launched blocks
that wait on the previous GEMM take its SMs (up to 18% slower at M = 4096).

## Block-scaled MXFP8 DeepGEMM (SM120)

`deepgemm_sm120_mm_mxfp8` (op `mm`, `SPECIALIZED`) serves E4M3 operands with
UE8M0 scales per 32 K on both sides, the format of the DeepSeek V4.1 dense
projections, on DeepGEMM's block-scaled SM120 kernel (`recipe=(1, 1, 32)`).
Before it, the Triton block-FP8 kernel was the only SM120 leaf for this
signature and took most of a V4.1 decode step. Scales arrive as canonical
`uint8 [MN, K/32]` and are packed into DeepGEMM's int32 MN-major words by one
Triton launch (`_triton/ue8m0_pack.py`). For a reused weight,
`prepare_mxfp8_weight_scales` packs the weight scales once on SM120 (it returns
FlashInfer's swizzled scales on SM100), and `mm(prepacked_scales=True)` with
those int32 scales quantizes 16-bit activations straight into the packed layout
(`quantize_fp8_group32_mn_major_scales`, same codes and scale bytes as the
canonical group-32 quantizer). N must be a multiple of 64 and K of 128; other
weights keep the canonical path.

The leaf runs DeepGEMM in deterministic mode, which on SM120 skips stream-K and
the BLOCK_N = 32 K-warp split, so each output is one in-order K loop and a row
does not depend on its batch (`batch_invariant`). It registers only with
DeepGEMM builds that report the mode (`get_deterministic_algorithms`). Measured
on an RTX PRO 6000 (CUDA graph, weights larger than L2) on the six V4.1
projections at M = 1..2048: 2.2-11.7x faster than the Triton leaf. Tests:
`test/nvidia/ops/gemm/test_deepgemm_sm120_mxfp8.py`.

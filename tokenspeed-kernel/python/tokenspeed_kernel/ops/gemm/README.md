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

## Hopper block-FP8 DeepGEMM

`deepgemm_mm_fp8_blockscale` supports explicit `mm(..., override=...)` or
`solution="deepgemm"` selection. Serving can opt in with
`--dense-gemm-backend deepgemm` for standard block-FP8 dense linears. The
default `auto` route remains unchanged; routed experts use their own backend.

This optional DeepGEMM integration requires Hopper, contiguous E4M3 operands
A[M,K]/B[N,K], BF16 output, N divisible by 64, and K divisible by 128. It
converts canonical FP32 1x128 activation and 128x128 weight scales to the
required layout without UE8M0 rounding. Conversion is part of each call,
including graph capture; no persistent weight cache is created. Unsupported
forced calls fail explicitly. It does not claim batch-invariant numerics.

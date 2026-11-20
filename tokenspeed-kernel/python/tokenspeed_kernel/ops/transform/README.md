# Transform kernels

`hadamard_transform` applies a Sylvester Hadamard transform along the last
dimension (`scale` multiplies the result). Callers: the GLM-5.3-Flash indexer
query (`[tokens, 32 heads, 128]`, BF16).

| Leaf | Where | Notes |
|---|---|---|
| `fast_hadamard_transform` | SM90-SM103 | CUDA butterfly from the `tokenspeed-fast-hadamard-transform` wheel, which ships sm_90/sm_100/sm_103 cubins and no PTX. |
| `triton_hadamard_transform_128_mma` | SM120 | One tensor-core matmul per row tile, BF16/FP16, last dim 128. |
| `triton_hadamard_transform_128` | any NVIDIA/AMD | CUDA-core fallback; also FP32. |
| `torch_hadamard_transform` | reference | |

## SM120 (`triton_hadamard_transform_128_mma`)

The wheel has no sm_120 image (`no kernel image` at launch), so its window
stays capped at 10.3. The upstream source does build for sm_120 (nvcc 13.2,
`-gencode arch=compute_120,code=sm_120`) and runs correctly, but it is slower
than the leaf below, so the wheel is not worth rebuilding for SM120.

The leaf multiplies a `[BLOCK_M, 128]` row tile by the 128x128 sign matrix
`(-1) ** popcount(i & j)` on `mma.sync`. The signs are exact in BF16/FP16 and
the products accumulate in FP32, so the stored result is bitwise equal to the
CUDA butterfly on random inputs and within one ulp of the exact transform in
general. The tile is `[16, 128]` with 4 warps up to 2048 rows (decode, latency
bound) and `[64, 128]` with 8 warps above; the row count and `scale` are
runtime arguments, so only those two launch shapes are compiled.

Measured on an RTX PRO 6000 (CUDA graph, 32 heads x 128 BF16; roofline is a
device-to-device copy of the same tensor, 1461 GB/s):

| tokens | copy | portable Triton | upstream CUDA (sm_120 build) | `_mma` |
|---:|---:|---:|---:|---:|
| 1 | 1.0 us | 1.9 us | 1.0 us | 2.1 us |
| 8 | 0.9 us | 6.0 us | 1.1 us | 2.2 us |
| 64 | 1.0 us | 41.6 us | 1.9 us | 2.1 us |
| 512 | 2.0 us | 321 us | 9.7 us | 5.4 us |
| 4096 | 10.8 us | 2507 us | 73 us | 21.4 us |
| 16384 | 183 us | 10097 us | 292 us | 183.7 us |
| 65536 | 734 us | 40380 us | 1165 us | 735 us |

From 16K rows up the leaf runs at the copy roofline (within 0.2%). A
single-token call is launch bound and ties the portable kernel at about 2 us.
`test/nvidia/ops/test_hadamard_sm120.py` checks it against a float64 transform
and guards against recompiles; `bench_hadamard_sm120.py` reproduces the table.

# MHA attention leaves

`mha_prefill`, `mha_extend_with_kvcache` and `mha_decode_with_kvcache` pick a
leaf per platform through the registry. Notes below cover only choices that are
not obvious from the registration.

## FA4 prefill on SM120

`fa4_mha_prefill_sm120` (`cuda.py`, priority `SPECIALIZED`) serves causal
varlen prefill on workstation Blackwell with FlashAttention 4's CuTe DSL
package. SM120 has no `tcgen05` or TMEM, so FA4 runs its SM80-class
`mma.sync.m16n8k16` kernel there, in 99 KB of shared memory. Head dims 64, 80,
96, 112 and 128 are declared; 256 does not fit and fails to compile, so those
calls keep the portable Triton leaf (`triton_mha_prefill`). Sinks and logit
capping are not supported by FA4 and keep the Triton leaf as well; sliding
windows and the LSE output are.

One RTX PRO 6000 Blackwell Server Edition, BF16, 32 query and 8 KV heads,
head dim 128, causal, ragged batches. A microbenchmark ran the Triton leaf and
FA4 on the same inputs; the largest absolute difference was 0.0039-0.0078, one
BF16 ulp:


| batch | Triton | FA4 | speedup |
| --- | ---: | ---: | ---: |
| 16 x 811 tokens | 1.296 ms (67 TFLOP/s) | 0.410 ms (210 TFLOP/s) | 3.16x |
| 32 x 400 tokens | 0.833 ms (50 TFLOP/s) | 0.255 ms (164 TFLOP/s) | 3.26x |
| 4 x 2048 tokens | 1.771 ms (78 TFLOP/s) | 0.554 ms (248 TFLOP/s) | 3.20x |
| 1 x 8192 tokens | 6.764 ms (81 TFLOP/s) | 1.983 ms (277 TFLOP/s) | 3.41x |

FA4's varlen scheduler launches a few surplus tiles after the last sequence and
reads `cu_seqlens[num_batch + 1]` for them, one int32 past the end of the
tensor the engine passes. The leaf hands FA4 a view over a copy that repeats the
last value, so the surplus tiles are empty. Without it the neighbouring bytes
became a bogus sequence length: the embedding server died with `illegal memory
access` in about two of three runs, while offline runs, a fuzz and
`CUDA_LAUNCH_BLOCKING=1` stayed clean because the byte after `cu_seqlens` was
usually mapped and zero.

`test/ops/attention/test_fa4_sm120_prefill.py` compares the leaf with an FP32
reference per head dim, GQA ratio and window, checks that it does not
recompile across batch shapes and pins which leaf SM120 selects at head dims
64, 128 and 256.

## Triton causal prefill and extend

`_triton/prefill.py` (`triton_mha_prefill`, `triton_mha_extend_with_kvcache`)
stops each query tile's key loop at the tile's last query position when the call
is causal and has no custom mask, with or without a sliding window. The skipped
key tiles are fully masked, so the output is unchanged; a prefill chunk that
starts at position 0 does about half the work. Before this the loop ran to the
end of the sequence unless a window was set. On SM120 a 8192-token chunk of
Qwen3.6-35B-A3B (head dim 256, FP8 KV) went from 4.2 ms to 2.5 ms per full
attention layer. `test_mha_prefill_triton_window_bounds` covers windowed and
unwindowed causal calls with a cached prefix and a partial query tile.

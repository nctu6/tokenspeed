# fa4_sm90 — vendored FlashAttention CuTe-DSL kernels (SM90 head_dim 512)

Source: the `vllm/vllm_flash_attn/cute` package bundled in the vLLM v0.31.0
image (`vllm/vllm-openai`), a fork of FlashAttention's CuTe-DSL kernels
(https://github.com/Dao-AILab/flash-attention, `flash_attn/cute`) that extends
the SM90 forward to head_dim up to 512. Upstream FlashAttention is BSD-3-Clause
(`LICENSE`); the vLLM distribution is Apache-2.0 (`LICENSE.vllm`). Per-file
copyright headers are kept unchanged.

Local changes: package imports rewritten from `vllm.vllm_flash_attn.cute` to
`tokenspeed_kernel.thirdparty.fa4_sm90`. TokenSpeed uses only the SM90
forward for head_dim 512 paged extend
(`tokenspeed_kernel/ops/attention/mha/fa4_sm90_hd512.py`).

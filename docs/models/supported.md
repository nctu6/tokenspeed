# Supported Models

UnieInfra serves the model architectures listed on this page. Each one is a
native implementation under
[`python/tokenspeed/runtime/models/`](https://github.com/UnieAI/UnieInfra/tree/main/python/tokenspeed/runtime/models):
a module exports its classes through `EntryClass`, and the model registry
(`registry.py`) matches them against the `architectures` field of the
checkpoint's `config.json`. Plugins can add architectures at runtime through
`register_model` (see `registered_architectures()`); those are not listed here.

For launch commands, parallelism layouts, and backend choices, see
[Model recipes](../recipes/models.md).

## Feature status legend

- ✅ Supported and verified on that platform (see [Verification levels](#verification-levels)).
- ⚠️ Runs, but with known issues or limitations (described in Notes).
- 🚧 Not working yet, or planned.
- Blank: the code is present but has not been verified on that platform.

A ✅ needs evidence on that vendor: a recipe launch command for it, a CI
model test on its runners, or the MI350P measurements below. The Notes
column names the evidence. NVIDIA covers `sm90`/`sm100`/`sm103`/`sm107`;
AMD covers `gfx950` (MI350/MI355 series) unless noted.

## Text-only language models

| Architecture | Models | Example HF models | NVIDIA | AMD | Notes |
| --- | --- | --- | :-: | :-: | --- |
| `BeamForCausalLM` | Beam | `reflection/Beam-1.0-501B-A23B-NVFP4` | ✅ | ✅ | BF16/FP8/NVFP4 on NVIDIA (TP8, two nodes); BF16 and W8A8 FP8 on gfx950 (TP8). DFlash2 draft. [Recipe](../recipes/models.md#beam) |
| `DeepseekV3ForCausalLM` | DeepSeek-V3 family | — | | | Base of the Kimi K2.5 MLA path; loader unit tests only. |
| `DeepseekV32ForCausalLM` | DeepSeek-V3.2 | — | | | |
| `Dots3NoteForCausalLM` | dots3-note | — | | | Text inference only: vision and audio encoder weights are skipped. |
| `Gemma3ForCausalLM` | Gemma 3 (text config) | — | | | Text-only entry of the Gemma 3 module. |
| `Gemma4ForCausalLM` | Gemma 4 (text config) | — | | | Text-only entry of the Gemma 4 module. |
| `GlmMoeDsaForCausalLM` | GLM-5, GLM-5.2, GLM-5.3 | `zai-org/GLM-5.2-FP8`, `nvidia/GLM-5.2-NVFP4`, `zai-org/GLM-5.3` | ✅ | | TP8 + EP, FP8/NVFP4, MTP from the base checkpoint. B200 AIME26 CI. Not verified on MI350P (tiny dims are illegal for the AMD FP8 kernels). [Recipe](../recipes/models.md#glm5-glm5-2) |
| `GptOssForCausalLM` | GPT-OSS 20B / 120B | `openai/gpt-oss-20b`, `openai/gpt-oss-120b` | ✅ | ✅ | MXFP4. GPQA-Diamond CI on B200 and MI35x. gfx950: Gluon Petit MegaMoE (DP8/EP8). [Recipe](../recipes/models.md#gpt-oss-20b-120b) |
| `LlamaForCausalLM` | Llama | — | | | |
| `LongcatFlashForCausalLM` | LongCat-Flash | — | | | Unit tests only. |
| `Mistral3ForConditionalGeneration` | Ministral 3, Mistral 3 (text only) | `mistralai/Ministral-3-14B-Instruct-2512` | | ✅ | The Pixtral vision tower is not supported; image requests are rejected. MI350P: FP8, TP1, GSM8K 96%. |
| `NemotronHForCausalLM` | Nemotron-3 Super (Nemotron-H hybrid) | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | ✅ | | Mamba2 + latent MoE + GQA, NVFP4, single GPU, MTP draft. [Recipe](../recipes/models.md#nemotron-3-super) |
| `Qwen2ForCausalLM` | Qwen2 | `Qwen/Qwen2-1.5B-Instruct` | | | Manual generation test only. |
| `Qwen3ForCausalLM` | Qwen3 dense | `Qwen/Qwen3-0.6B` | | | Verified on Ascend NPU only. [Recipe](../recipes/models.md#qwen3-0-6b-on-ascend-npu) |
| `Qwen3MoeForCausalLM` | Qwen3 MoE | `Qwen/Qwen3-30B-A3B` | ✅ | | TP2 + EP. [Recipe](../recipes/models.md#qwen3-dense-qwen3-30b-a3b) |
| `Qwen3_5MoeForCausalLM` | Qwen3.5 MoE (text config) | — | | | Text-only entry of the Qwen3.5 module. |
| `Qwen4ExpForCausalLM` | Qwen4-Exp (text config) | — | | | Text-only entry of the Qwen3.8 Flash Next module. |

DeepSeek V4-Flash / V4-Pro (`DeepseekV4ForCausalLM`) is no longer registered:
its model implementation was removed upstream, so the V4 recipe sections do not
apply to this tree. DeepSeek V4.1 is listed below.

## Multimodal language models

Inputs: **T**ext, **I**mage, **V**ideo, **A**udio. Modalities joined by `+`
can be combined in one request: `T + I` accepts text only, image only, or
text with images. Modalities separated by `/` are mutually exclusive: `T / I`
accepts text or images, but not both in one request. Image, video, and audio
preprocessing runs in the SMG gateway.

| Architecture | Models | Inputs | Example HF models | NVIDIA | AMD | Notes |
| --- | --- | --- | --- | :-: | :-: | --- |
| `DeepseekV41ForCausalLM` | DeepSeek-V4.1-Flash | T + I | `deepseek-ai/DeepSeek-V4.1-Flash`, `deepseek-ai/DeepSeek-V4.1-Flash-1017` | ✅ | ✅ | GSM8K CI on B200, MI35x, and GB300 1P1D. MI350P: TP4 production, GSM8K 0.9045, KV tiering (Host RAM + NVMe L3), DSpark. Only text input verified on MI350P. [Recipe](../recipes/models.md#deepseek-v4-1-flash) |
| `Gemma3ForConditionalGeneration` | Gemma 3 | T + I | `google/gemma-3-27b-it` | ✅ | | BF16 launch script `scripts/gemma3.sh`. |
| `Gemma4ForConditionalGeneration` | Gemma 4 (dense and MoE) | T + I + V + A | `google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it` | ✅ | ✅ | Audio loads only when the checkpoint has `audio_config`. NVIDIA: `scripts/gemma4.sh` (31B, BF16, TP2). MI350P: 26B-A4B GSM8K 97.7% at TP1, 97.3% at TP4. Gateway needs `--reasoning-parser passthrough`. |
| `Glm53FlashForConditionalGeneration` | GLM-5.3-Flash | T + I + V | `zai-org/GLM-5.3-Flash` | ✅ | ✅ | BF16/block-FP8, MTP from the base checkpoint. gfx950: TP4, EP off. AIME26 CI on B200 and MI35x. Not verified on MI350P. [Recipe](../recipes/models.md#glm-5-3-flash) |
| `InklingForConditionalGeneration` | Inkling | T + I + A | `thinkingmachines/Inkling-NVFP4`, `lightseekorg/Inkling-MXFP4` | ✅ | ✅ | NVFP4 (NVIDIA) / MXFP4 (AMD), TP4, MTP. AIME25 CI on B200 and MI35x. [Recipe](../recipes/models.md#inkling) |
| `KimiK25ForConditionalGeneration` | Kimi K2.5, K2.6 | T + I | `nvidia/Kimi-K2.5-NVFP4`, `nvidia/Kimi-K2.6-NVFP4` | ✅ | | NVFP4, TP4 + EP, EAGLE3 / DFlash drafts. CI on B200 (GSM8K, AIME25, GPQA, MMLU, OCRBench). [Recipe](../recipes/models.md#kimi-k2-5-k2-6) |
| `KimiK3ForConditionalGeneration` | Kimi K3 | T + I | `moonshotai/Kimi-K3` | ✅ | ✅ | MXFP4/NVFP4. NVIDIA: TP8/EP8, multi-node, DSpark/DFlash2, GB300/GB200 CI. AMD: TP8/EP8 recipe and MI35x AIME26 CI; on MI350P a tiny checkpoint matches HF teacher-forced greedy at TP1, TP4 and EP4 (two single-token near-tie differences remain); real weights not yet run on MI350P. [Recipe](../recipes/models.md#kimi-k3) |
| `MiniMaxM3SparseForConditionalGeneration` | MiniMax M3 | T + I + V | `nvidia/MiniMax-M3-NVFP4` | ✅ | | NVFP4, TP4, EAGLE3 / DSpark drafts. AIME CI on B200. [Recipe](../recipes/models.md#minimax-m3) |
| `MuseSparkForConditionalGeneration` | Muse Spark Open | T + I + V + A | `meta-models/Muse-Spark-Open-816B-A42B`, `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`, `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4` | ✅ | ✅ | NVIDIA: NVFP4 on 4× B300. AMD: BF16 TP8/EP8 on MI350X, MXFP4 AIME26 CI on MI35x (TP4/EP4, TP8/EP8). MI350P: kernel tests pass at TP4 and a 2-layer real-dimension dummy serves at TP1/TP4; the full checkpoint has not run. [Recipe](../recipes/models.md#muse-spark-open) |
| `NemotronH_Nano_Omni_Reasoning_V3` | Nemotron-H Nano Omni | T + I + V + A | — | | | RADIO vision and Parakeet audio encoders. Unit tests only. |
| `NemotronH_Omni_Reasoning_V3` | Nemotron-H Omni | T + I + V + A | — | | | Same encoders as above. |
| `Qwen3OmniMoeForConditionalGeneration` | Qwen3-Omni | T + I + V + A | — | | | Text output only. Unit tests only. |
| `Qwen3_5ForConditionalGeneration` | Qwen3.5 dense, Qwen3.8-27B | T + I + V | `Qwen/Qwen3.5-27B`, `Qwen/Qwen3.8-27B-FP8` | ✅ | ✅ | NVIDIA: FP8 single GPU + MTP; GSM8K CI on GB10. MI350P: Qwen3.8-27B BF16 TP4, GSM8K (300) 95.3%, 1462 tok/s at c32; gateway needs `--reasoning-parser qwen3`. [Recipe](../recipes/models.md#qwen3-8-27b) |
| `Qwen3_5MoeForConditionalGeneration` | Qwen3.5 MoE, Qwen3.6 MoE | T + I + V | `Qwen/Qwen3.5-35B-A3B-FP8`, `nvidia/Qwen3.5-122B-A10B-NVFP4`, `nvidia/Qwen3.5-397B-A17B-NVFP4`, `Qwen/Qwen3.6-35B-A3B` | ✅ | ⚠️ | NVIDIA: NVFP4/FP8, DP/EP/DeepEP, PD and EPD disaggregation, MTP; CI on B200/GB200/GB10. MI350P: tiny checkpoint matches HF at TP1/TP4; EP 🚧 (no BF16 EP MoE kernel on AMD); no full checkpoint run. |
| `Qwen4ExpForConditionalGeneration` | Qwen3.8-Flash-Next | T + I | `Qwen/Qwen3.8-Flash-Next-FP8`, `amd/Qwen3.8-Flash-Next-Quark-MXFP4` | ✅ | ✅ | GDN + QSA hybrid, PLE embeddings, MTP. NVIDIA FP8 TP4; AMD Quark MXFP4 TP2. GSM8K CI on GB200 and MI35x. [Recipe](../recipes/models.md#qwen3-8-flash-next) |

## Speech models

| Architecture | Models | Inputs | Example HF models | NVIDIA | AMD | Notes |
| --- | --- | --- | --- | :-: | :-: | --- |
| `WhisperForConditionalGeneration` | Whisper | A | `openai/whisper-large-v3` | ✅ | | Encoder-decoder, served on `/v1/audio/transcriptions`. Launch script `scripts/whisper-large-v3.sh`, TP1 smoke on sm90. Prefix caching and KVStore are off for encoder-decoder models. |
| `Qwen3ASRForConditionalGeneration` | Qwen3-ASR | T + A | — | | | Decoder with the shared Qwen3-Omni audio tower. Unit tests only. |

Parakeet has no standalone architecture: it is the audio encoder of the
Nemotron-H Omni models above.

## Draft models for speculative decoding

Draft architectures load through `--speculative-draft-model-path`, or from the
target checkpoint itself when it carries the draft layers. See
[Speculative decoding](../configuration/server.md#speculative-decoding) for the
flags.

| Architecture | Method | Draft for | NVIDIA | AMD | Notes |
| --- | --- | --- | :-: | :-: | --- |
| `LlamaForCausalLMEagle3` | EAGLE3 | Kimi K2.5 | ✅ | | `lightseekorg/kimi-k2.5-eagle3`; Kimi K2.5 EAGLE3 CI on B200. |
| `Eagle3DeepseekV2ForCausalLM` | EAGLE3 (MLA draft) | Kimi K2.5 | | | `nvidia/Kimi-K2.5-Thinking-Eagle3`; manual test only. |
| `DSparkDraftModel` | DSpark | MiniMax M3 | ✅ | | `nvidia/MiniMax-M3-DSpark`; AIME26 CI on B200. |
| `DeepseekV41ForCausalLMDSpark` | DSpark (same checkpoint) | DeepSeek-V4.1-Flash | ✅ | ✅ | GSM8K CI on B200, MI35x, GB300 1P1D. MI350P production. |
| `K3DSparkModel` | DSpark | Kimi K3 | ✅ | | GB300 two-node AIME26 CI. |
| `DFlashDraftModel` | DFlash | Kimi K2.5 / K2.6 | ✅ | | `z-lab/Kimi-K2.5-DFlash`; AIME25 CI on B200. |
| `DFlash2DraftModel` | DFlash2 | Kimi K3 | ✅ | | `lightseekorg/kimi-k3-dflash2`; GB300 two-node AIME26 CI. |
| `DFlash2BeamDraftModel` | DFlash2 | Beam | ✅ | ✅ | `reflection/Beam-1.0-501B-A23B-DFlash2`. [Recipe](../recipes/models.md#beam) |
| `GlmMoeDsaForCausalLMNextN` | NextN / MTP | GLM-5.2, GLM-5.3 | ✅ | | From the base checkpoint. |
| `Glm53FlashForConditionalGenerationNextN` | NextN / MTP | GLM-5.3-Flash | ✅ | ✅ | From the base checkpoint. |
| `InklingForConditionalGenerationNextN` | NextN / MTP | Inkling | ✅ | ✅ | From the base checkpoint. |
| `NemotronHForCausalLMNextN` | NextN / MTP | Nemotron-3 Super | ✅ | | From the base checkpoint; ReplaySSM on by default. |
| `Qwen3_5ForConditionalGenerationNextN` | NextN / MTP | Qwen3.5 dense, Qwen3.8-27B | ✅ | | From the base checkpoint. |
| `Qwen3_5MoeForConditionalGenerationNextN` | NextN / MTP | Qwen3.5 MoE | ✅ | | From the base checkpoint; Qwen3.5-35B-A3B NVFP4 CI. |
| `Qwen4ExpForCausalLMNextN` | NextN / MTP | Qwen3.8-Flash-Next | ✅ | ✅ | One-layer MTP from the base checkpoint. |
| `DeepseekV3ForCausalLMNextN` | NextN / MTP | DeepSeek-V3 family | | | |
| `DeepseekV32ForCausalLMNextN` | NextN / MTP | DeepSeek-V3.2 | | | |
| `Dots3NoteForCausalLMNextN` | NextN / MTP | dots3-note | | | |
| `KimiK3ForConditionalGenerationNextN` | NextN / MTP | Kimi K3 | | | |
| `NemotronH_Omni_Reasoning_V3NextN` | NextN / MTP | Nemotron-H Omni | | | |
| `Qwen3_5MoeForCausalLMNextN` | NextN / MTP | Qwen3.5 MoE (text config) | | | |

## Verification levels

A ✅ is backed by at least one of these, strongest first:

1. **Matches HF Transformers.** Greedy TokenSpeed output is teacher-forced
   through the Hugging Face Transformers model, and the argmax token and
   logprobs agree (`test/manual/compare_hf_greedy.py`). Tiny random-weight
   checkpoints are enough to check the kernels and the parallel layout.
2. **Accuracy eval.** The served model scores as expected on a benchmark such
   as GSM8K or AIME through the OpenAI-compatible API (the `test/ci/eval/`
   jobs, or the MI350P runs below).
3. **Serves and runs.** The model loads and answers requests with a recipe or
   launch-script command, without an accuracy gate.
4. **Code present.** The architecture is registered, but has not been
   verified on that platform. These cells are blank.

Exact equality with Transformers is not always reachable: low-precision
kernels, quantized checkpoints, and different reduction orders change the
numerics. Compare with identical sampling parameters; Transformers'
`generate` applies the checkpoint's `generation_config.json` defaults, while
the server uses only the request's parameters.

### Verified on 4× AMD Instinct MI350P (gfx950)

PCIe-only host without xGMI. Measured on `main` in 2026-11.

| Model | Architecture | Precision | Parallelism | Level | Result |
| --- | --- | --- | --- | :-: | --- |
| DeepSeek-V4.1-Flash | `DeepseekV41ForCausalLM` | Checkpoint default | TP4 | 2 | ✅ GSM8K 0.9045. Production layout with KV tiering (Host RAM + NVMe L3) and DSpark. AgentX speedups over vLLM in the [README](https://github.com/UnieAI/UnieInfra#readme). |
| Qwen3.8-27B | `Qwen3_5ForConditionalGeneration` | BF16 | TP4 | 2 | ✅ GSM8K (300) 95.3%, 1462 tok/s at c32. Gateway needs `--reasoning-parser qwen3`. |
| Ministral-3-14B-Instruct-2512 | `Mistral3ForConditionalGeneration` | FP8 | TP1 | 2 | ✅ GSM8K 96%, 2622 tok/s. Text only. |
| gemma-4-26B-A4B-it | `Gemma4ForConditionalGeneration` | BF16 | TP1 | 2 | ✅ GSM8K 97.7%, 1846 tok/s. Gateway needs `--reasoning-parser passthrough`. |
| gemma-4-26B-A4B-it | `Gemma4ForConditionalGeneration` | BF16 | TP4 | 2 | ✅ GSM8K (300) 97.3%, 2183 tok/s at c32. The per-rank expert intermediate size (704 / 4 = 176) is padded to 192 for the Triton MoE kernels. |
| Gemma 4 dense, tiny | `Gemma4ForConditionalGeneration` | BF16 | TP1, TP4 | 1 | ✅ Matches HF teacher-forced greedy. |
| Gemma 4 MoE, tiny | `Gemma4ForConditionalGeneration` | BF16 | TP1, TP4 | 1 | ✅ Matches HF teacher-forced greedy. |
| Qwen3.5 dense, tiny | `Qwen3_5ForConditionalGeneration` | BF16 | TP1, TP4 | 1 | ✅ Matches HF teacher-forced greedy. |
| Qwen3.5 MoE, tiny | `Qwen3_5MoeForConditionalGeneration` | BF16 | TP1, TP4 | 1 | ✅ Matches HF teacher-forced greedy, except one router near-tie flip at one prompt length (kernel probe OK). |
| Qwen3.5 MoE, tiny | `Qwen3_5MoeForConditionalGeneration` | BF16 | EP | — | 🚧 No BF16 expert-parallel MoE kernel on AMD. |
| Kimi K3 / Kimi Linear, tiny | `KimiK3ForConditionalGeneration` | BF16 | TP1, TP4 | 1 | ✅ Matches HF teacher-forced greedy, serial and batched, except one near-tie token at TP1 (prompt length 130) and one at TP4 (length 5). |
| Kimi K3 / Kimi Linear, tiny | `KimiK3ForConditionalGeneration` | BF16 | EP4 | 1 | ✅ Matches HF teacher-forced greedy at every tested prompt length. |
| Muse Spark, 2-layer real-dimension dummy | `MuseSparkForConditionalGeneration` | BF16 | TP1, TP4 | 3 | ✅ Kernel tests pass at TP4 and the dummy model serves. The full 816B checkpoint has not run on MI350P. |
| GLM-5.x / GLM-5.3-Flash, tiny | `GlmMoeDsaForCausalLM`, `Glm53FlashForConditionalGeneration` | — | — | 4 | Not verified: the tiny dimensions are illegal for the AMD FP8 kernels. |

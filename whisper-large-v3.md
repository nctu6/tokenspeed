# whisper-large-v3 — TokenSpeed support WIP

**Status:** smoke PASS both hosts; one commit pending/ done. Branch `nctu6` (was tip `0461b8e7`).
**Goal:** load + serve openai/whisper-large-v3 on both hosts (sm120 and
sm90); smoke `POST /v1/audio/transcriptions` on GPU 6, TP=1.
**Standing rules:** one commit when green (or as far as we get); no push;
Signed-off-by: `nctu6 <nctu6.tw@gmail.com>`.

## User ask

1. `scripts/whisper-large-v3.sh` (mirror embedding/qwen3.6 script style).
2. Minimum TokenSpeed-native WhisperForConditionalGeneration support so both
   hosts can load weights and answer a short-clip transcription.
3. Rsync (no push); free GPUs 6/7; smoke; tear down; logs under
   `/tmp/nctu6-whisper/` on remotes.

## Design refs (ideas only — do not copy-paste)

- Local vLLM-unieai: `vllm/model_executor/models/whisper{,_causal,_utils}.py`
- TokenSpeed ASR today: `qwen3_asr.py` / `qwen3_audio.py` (decoder-only + audio
  tower; transcription adapted to chat in SMG gRPC router).
- SMG: `smg/crates/protocols/src/transcription.rs`, multipart,
  `/v1/audio/transcriptions`.
- Model config: `/Users/nctu6/Downloads/ascend/infra/model_config/whisper-large-v3/`
  (`WhisperForConditionalGeneration`, `model_type: whisper`).
- Weights on both remotes: `/models/openai/whisper-large-v3`.

## Prior art in this monorepo (unieai remote — NOT on nctu6 HEAD)

Whisper was already prototyped on `remotes/unieai/unieai` (Roy):

| commit | what |
|---|---|
| `daa5f76f` | model + `cross_attn_varlen` + config (NOT servable yet) |
| `2a2f2fbf` | serve end-to-end; LibriSpeech clip parity vs HF |
| `770fc5f6` | drop per-step cross-attn gather at high occupancy |
| `4164b2c1` | `asr_http.py` `/v1/audio/transcriptions` + translations |

Key design notes from those commits (carry forward):

- Encoder reuses pool-free `MultimodalEncoderAttention`; decoder self-attn stays
  on ordinary paged KV; **cross KV is a fixed per-request buffer** (not in the
  paged arena) — matches vLLM / TRT-LLM, not SGLang's doubled layer-id pool.
- Cost: `decoder_layers × encoder_len × d_model × 2` per slot (~8.8 GiB at
  max_num_seqs=32 on large-v3) — keep max_num_seqs modest for smoke.
- Prefill CUDA graph disabled for encoder-decoder (cross KV addressed by pool
  slot; capture would freeze first batch).
- sm_120 historically forced decode-time gather (paged KV / flash_attn_with_kvcache
  gaps); re-probe rather than inherit.
- unieai entrypoint was a **standalone `asr_http`** (feature extract + decoder
  prompt + resample in-process). nctu6 serves via **SMG + TokenSpeed gRPC**;
  prefer extending Qwen3-ASR transcription adapter / multimodal registry over
  standing up a parallel HTTP stack — but port model + cross-attn first.

## Plan (incremental)

1. [x] Write this WIP md.
2. [x] Port / redesign `python/tokenspeed/runtime/models/whisper.py` +
      `cross_attn_varlen` onto current nctu6 APIs (Mapping, layers, weight
      loader, multimodal interfaces).
3. [x] Config / registry: architecture `WhisperForConditionalGeneration`,
      audio model flag, context length from `max_target_positions`.
4. [x] Transcription path (asr_http; SMG still Qwen3-ASR-only): SMG detect whisper (like qwen3_asr) OR lightweight
      asr path that still fronts SMG — decide after reading serve_smg + gateway.
5. [x] `scripts/whisper-large-v3.sh` (arch detect, GPU 6, TP=1, dtype bf16,
      smoke curl multipart).
6. [x] Rsync both hosts (Python-only, no SMG rematurin); smoke PASS;
      free GPUs; logs.
7. [x] One commit from this md; no push.

## Progress log

- 2026-10-07 Asia/Taipei: tip `0461b8e7` on `nctu6`. No in-tree whisper.py.
  Located unieai whisper lineage (`daa5f76f`…`4164b2c1`). Creating this md;
  next: diff mm_encoder_attention / model_config vs those commits and port.

## Blockers / open questions

- How does nctu6 `tokenspeed serve` + SMG carry mel features for Whisper
  (not MultimodalEmbedder path)? May need SHM feature consume in model +
  multimodal registry Whisper mel preprocessor + gRPC transcription adapter.
- Whether to rematurin SMG (Rust change) for whisper routing — prefer yes if
  we extend grpc router; avoid if we can reuse HTTP multipart→worker path.

## Intended commit body (draft — update before committing)

```
feat: whisper-large-v3 serve + transcription smoke (sm90/sm120)

Add TokenSpeed-native WhisperForConditionalGeneration (encoder + paged
decoder self-attn + fixed cross-KV buffer), wire OpenAI-compatible
POST /v1/audio/transcriptions, and scripts/whisper-large-v3.sh with
arch-aware defaults (GPU 6, TP=1, bf16).

Smoke: health OK + non-empty transcription on both hosts (sm90/H200 and
sm120/PRO 6000); GPUs freed after tear-down.

See whisper-large-v3.md for design notes and remaining gaps.

Signed-off-by: nctu6 <nctu6.tw@gmail.com>
```

## Remaining gaps (fill as we go)

_(none recorded yet)_

## Progress log (continued)

- 2026-10-07 10:59 CST: Ported unieai whisper tip onto nctu6 with API
  adaptations (PagedAttention positions/rotary_emb/qk_norm, LogitsProcessor
  dp_lm_head_tp, drop out_cache_loc). Added `cross_attn_varlen` (+ dtype
  normalize), model_config hooks (decoder layer count, multimodal/audio,
  cross_attn_slots, prefill budget clamp), kv auto→model dtype, model_executor
  enc-dec disable prefill graph + req_pool_indices kwargs. CLI routes Whisper
  → asr_http via runtime_select. Script: scripts/whisper-large-v3.sh.

## Files touched (uncommitted)

- `python/tokenspeed/runtime/models/whisper.py` (new)
- `python/tokenspeed/runtime/entrypoints/asr_http.py` (new)
- `python/tokenspeed/runtime_select.py` (new)
- `python/tokenspeed/cli/__main__.py`
- `python/tokenspeed/runtime/layers/attention/mm_encoder_attention.py`
- `python/tokenspeed/runtime/layers/attention/configs/mha.py`
- `python/tokenspeed/runtime/configs/model_config.py`
- `python/tokenspeed/runtime/utils/hf_transformers_utils.py`
- `python/tokenspeed/runtime/execution/model_executor.py`
- `scripts/whisper-large-v3.sh` (new)
- `test/runtime/entrypoints/test_asr_prompt.py` (new)
- `whisper-large-v3.md` (this file)

## Remaining gaps

- SMG gRPC `/v1/audio/transcriptions` still rejects non-Qwen3-ASR; Whisper
  uses TokenSpeed-native `asr_http` instead (same OpenAI multipart contract).
- Language auto-detect not implemented (defaults to `en` when omitted).
- Cross-KV memory scales with max_num_seqs (script default 4).
- No SMG rematurin required for this path (Python-only).


## Smoke results (2026-10-07 Asia/Taipei)

| host | arch | health | transcription HTTP | text | GPU 6/7 after teardown |
|---|---|---|---|---|---|
| remote B (sm120) | sm120 / PRO 6000 | OK | 200 | ` The quick brown fox jumps over the lazy dog.` | 0 / 0 MiB |
| remote A (sm90) | sm90 / H200 | OK | 200 | ` The quick brown fox jumps over the lazy dog.` | 0 / 0 MiB |

- Serve: `CUDA_DEVICES=6 HOST_PORT=8778 ./scripts/whisper-large-v3.sh`
- Sample: espeak-ng WAV `/tmp/nctu6-whisper/smoke_speech.wav` (copied to sm90 host)
- Logs: `/tmp/nctu6-whisper/serve.{sm120,sm90}.log`, `smoke.*.out` on remotes
- Note: sm90/H200 default `gpu_memory_utilization=0.95` allocates a huge Host L2
  (~257 GiB) and delays readiness ~3–4 min; smoke still passed. Prefer
  `GPU_MEMORY_UTILIZATION=0.3` for faster smoke startups on H200.

## Intended commit body (final)

```
feat: whisper-large-v3 serve + transcription smoke (sm90/sm120)

Add TokenSpeed-native WhisperForConditionalGeneration (encoder + paged
decoder self-attn + fixed cross-KV buffer), route Whisper checkpoints to
asr_http for OpenAI-compatible POST /v1/audio/transcriptions, and ship
scripts/whisper-large-v3.sh (GPU 6, TP=1, bf16).

Smoke PASS on both hosts (sm90/H200 and sm120/PRO 6000):
health OK, HTTP 200 with non-empty transcript for an espeak clip, GPUs
freed after tear-down.

SMG gRPC transcription adapter remains Qwen3-ASR-only; Whisper uses the
in-process asr_http path (same multipart contract). See whisper-large-v3.md.

Signed-off-by: nctu6 <nctu6.tw@gmail.com>
```

# MiniMax-H3-Image: TokenSpeed serve

`./scripts/minimax-h3-image.sh` serves `/models/MiniMaxAI/MiniMax-H3-Image`
(text-to-image and reference-image editing, 33B DiT, 6 distilled steps, no
guidance). It is the image sibling of `minimax-h3.md` (video) and uses the same
diffusion runtime entry: `tokenspeed serve` reads `model_index.json`,
`detect_diffusion_family` returns `h3image` for `MiniMaxH3ImagePipeline`, and
`entrypoints/image_http.py` takes over.

Needs `diffusers>=0.43.0` (`MiniMaxH3ImageModularPipeline`). The text encoder is
the MiniMax-H3 `text_encoder/` (`model_index.json` points at `MiniMaxAI/MiniMax-H3`).

## API

* `POST /v1/images/generations` JSON: `prompt`, `size` (`WxH`, multiples of 32),
  optional `seed`, `output_format` (png|jpeg|webp), `min_generation_area`.
* `POST /v1/images/edits` multipart: same fields plus `image` files (up to 16,
  referred to as `<Picture 1>`, `<Picture 2>` in the prompt).
* Response: `{"data": [{"b64_json": ...}], "size", "seed", "output_format"}`.
* `n`, `num_inference_steps`, `guidance_scale` and `negative_prompt` are rejected:
  the schedule is fixed by the checkpoint.

## Residency

`device_split` (default with 2 GPUs): Qwen3-VL conditioner on `cuda:1`, DiT + VAE
on `cuda:0`, both resident (~65 GB each). `single`: one GPU, use
`ENABLE_CPU_OFFLOAD=1`. Ulysses/ring are not supported for images.

## Verified (2026-11-20, 2x RTX PRO 6000, sm120, GPUs 6,7)

* t2i 1024x1024: 6.4 s; 4736x3552: 30 s (warm).
* edit with one reference at 1024x1024 keeps the subject and changes the scene.
* t2i 4096x4096: 30 s; 2-reference edit at 2048x2048: 18 s (device_split).
* Single GPU (`CUDA_DEVICES=6 NUM_GPUS=1 ENABLE_CPU_OFFLOAD=1`): works up to
  4096x4096 and a 2-reference 2048x2048 edit (peak ~69 GB), but each request
  swaps ~62 GB weights: ~76 s at 1024x1024, ~155 s at 4096x4096.

# Copyright (c) 2026 LightSeek Foundation
# Copyright (c) 2026 UnieAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""``/v1/audio/transcriptions`` and ``/v1/audio/translations`` for Whisper.

The sibling of :mod:`pooling_http`, and for the same reason: the engine can
serve Whisper -- its transcript is verified word-for-word against
HuggingFace -- and nothing could reach it, because ``ts serve``'s gateway
speaks a proto whose only generative RPC is text in, text out.

Three things happen here rather than in the engine, because the engine does
none of them:

* **Feature extraction.** The engine takes log-mel features, not audio. It
  has no decoder for wav/flac/mp3 and no ``WhisperProcessor``; the gateway
  ships precomputed features and so does this.
* **The decoder prompt.** Whisper is told what to do by the tokens it starts
  from -- ``<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`` --
  so ``language`` and the transcribe/translate split are prompt construction,
  not sampling parameters. ``/v1/audio/translations`` is the same endpoint
  with ``<|translate|>``, which is why it is four lines and not a second
  server.
* **Language default.** OpenAI's ``language`` is optional and Whisper can
  detect it, but detection needs a first decode pass this server does not
  do. Absent ``language`` it asks for English and says so in the response,
  rather than silently transcribing Mandarin as though it were English.

Handlers are ``def``: see :mod:`pooling_http` for why awaiting the engine's
async path from uvicorn's loop hangs.
"""

from __future__ import annotations

import io
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve"]

# What Whisper starts decoding from. The language slot is filled per request.
_START = "<|startoftranscript|>"
_NO_TIMESTAMPS = "<|notimestamps|>"


def _prompt_tokens(tokenizer, language: str, task: str) -> list[int]:
    tokens = [_START, f"<|{language}|>", f"<|{task}|>", _NO_TIMESTAMPS]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # Round-trip, not a None check: convert_tokens_to_ids maps an unknown
    # token to unk_token_id rather than None, so `language=zzz` sails through
    # a None check and Whisper decodes from an unknown-token prompt -- which
    # answers 200 with a plausible transcript in some arbitrary language.
    # Measured: the None check returned 200 for `language=zzz`.
    roundtrip = tokenizer.convert_ids_to_tokens(list(ids))
    unknown = [t for t, back in zip(tokens, roundtrip) if back != t]
    if unknown:
        raise ValueError(
            f"this checkpoint's tokenizer has no {unknown} -- check the "
            f"`language` code (Whisper wants 'en', 'zh', 'ja', ...)"
        )
    return list(ids)


def _transcribe(
    engine,
    processor,
    audio: bytes,
    *,
    language: str,
    task: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    import soundfile as sf

    from tokenspeed.runtime.engine.io_struct import GenerateReqInput
    from tokenspeed.runtime.multimodal.inputs import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
    )

    wav, rate = sf.read(io.BytesIO(audio), dtype="float32")
    if getattr(wav, "ndim", 1) > 1:
        # Whisper's feature extractor wants mono; a stereo clip would
        # otherwise be read as twice the frames at half the duration.
        wav = wav.mean(axis=1)

    # Resample to whatever the extractor was configured for (16 kHz for every
    # Whisper checkpoint). It does NOT resample -- handing it 44.1 kHz audio
    # and telling it the true rate yields features covering a third of the
    # intended window, which transcribes to something fluent and wrong. An
    # endpoint that accepts uploads gets whatever rate the caller has.
    target = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    if rate != target:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(int(rate), target)
        wav = resample_poly(wav, target // divisor, int(rate) // divisor)
        rate = target

    features = processor.feature_extractor(
        wav, sampling_rate=rate, return_tensors="pt"
    ).input_features

    request = GenerateReqInput(
        input_ids=_prompt_tokens(processor.tokenizer, language, task),
        sampling_params={
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
    )
    request.precomputed_multimodal_inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.AUDIO, feature=features[0].contiguous()
            )
        ]
    )
    return engine.llm.generate(request)["text"]


def create_app(engine, processor, served_model_name: str) -> FastAPI:
    app = FastAPI(title="TokenSpeed ASR server")
    app.state.engine = engine
    app.state.processor = processor

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "tokenspeed",
                }
            ],
        }

    def _handle(
        file: UploadFile,
        language: str | None,
        response_format: str,
        temperature: float,
        task: str,
    ):
        audio = file.file.read()
        if not audio:
            return JSONResponse({"error": "empty audio file"}, status_code=400)
        # See the module docstring: no detection pass, so an absent language
        # is answered with a stated default rather than a guess.
        resolved = language or "en"
        try:
            text = _transcribe(
                app.state.engine,
                app.state.processor,
                audio,
                language=resolved,
                task=task,
                max_new_tokens=448,
                temperature=temperature,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text, "language": resolved}

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
    ):
        return _handle(file, language, response_format, temperature, "transcribe")

    @app.post("/v1/audio/translations")
    def translations(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
    ):
        # Translation is always *into* English, so the language slot still
        # names the source; Whisper handles that from the audio itself and
        # the task token is what changes.
        return _handle(file, None, response_format, temperature, "translate")

    return app


def serve(argv: list[str], *, host: str, port: int) -> None:
    """Build the engine and its processor from ``argv``, then serve."""
    import uvicorn
    from transformers import AutoProcessor

    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    server_args = prepare_server_args(argv)
    processor = AutoProcessor.from_pretrained(server_args.model)
    engine = Engine(server_args=server_args)
    served = server_args.served_model_name or server_args.model
    if isinstance(served, (list, tuple)):
        served = served[0]
    logger.info("asr server: model=%s", served)
    app = create_app(engine, processor, served_model_name=served)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        engine.shutdown()

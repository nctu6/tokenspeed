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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

**Streaming.** ``stream=true`` answers ``text/event-stream`` in the chunk
schema vLLM 0.32.0 emits, so a benchmark measures TTFT the same way against
both servers. Each event is ``data: {json}`` with ``id``, ``object``
(``transcription.chunk`` or ``translation.chunk``), ``created``, ``model``
and ``choices: [{"delta": {"content": ...}}]``; the last text chunk adds
``finish_reason`` and ``stop_reason``. ``stream_include_usage`` appends a
``choices: []`` chunk carrying ``usage``, ``stream_continuous_usage_stats``
puts ``usage`` on every chunk, and the stream ends with ``data: [DONE]``.
The engine streams the cumulative transcript, already held back at an
incomplete UTF-8 sequence by its incremental detokenizer; each chunk sends
the new suffix and empty suffixes are skipped. Everything that can be
rejected (empty audio, an unknown ``language``) is rejected with a JSON 400
before the stream starts; an engine error after that arrives as a ``data:``
error event followed by ``[DONE]``, as in vLLM.

Handlers are ``def`` (Starlette runs them in its threadpool, where feature
extraction belongs) and never await the engine's async path from uvicorn's
loop: ``AsyncLLM`` belongs to the sync bridge's own loop. A streamed body is
an async generator over :meth:`LLM.generate_stream_async`, which hands each
engine output to uvicorn's loop through a queue that loop owns, so a stream
holds no threadpool thread and costs no thread handoff per chunk. (A sync
generator over :meth:`LLM.generate_stream` costs two per chunk; at 16
concurrent streams that GIL traffic slowed every request's feature
extraction and admission, 80 vs 18 ms upload-to-engine -- see
:meth:`LLM.generate_stream_async`.) A client that disconnects closes the
generator, which aborts the request.

**Feature extraction under concurrency.** Decode + log-mel is ~6 ms of CPU
per clip, but run from every threadpool thread at once it collapses: the HF
extractor's torch STFT releases and re-takes the GIL many times per call,
and each of the process's default intra-op threads (half the cores) piles
onto the same few ops. Measured on a 192-core host with 16 concurrent
callers: 107 ms per call. So at most :data:`_EXTRACT_CONCURRENCY` requests
extract at once, and :func:`serve` caps this process's torch intra-op
threads at :data:`_EXTRACT_TORCH_THREADS` (the engine's GPU work runs in
spawned scheduler processes and is unaffected): 24 ms per call at 16
callers, 3.8 ms alone. Features are bit-identical at any thread count. A
process pool was measured too and lost: ~4-5 ms of IPC per call.
"""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve"]

# What Whisper starts decoding from. The language slot is filled per request.
_START = "<|startoftranscript|>"
_NO_TIMESTAMPS = "<|notimestamps|>"

# Feature extraction tuning (module docstring): the best measured pair among
# concurrency 2 / 4 / 8 / 16 and torch threads 1 / 2 / 4 / 8 / default, at 1, 8
# and 16 concurrent callers.
_EXTRACT_CONCURRENCY = 4
_EXTRACT_TORCH_THREADS = 4


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


def _build_request(
    processor,
    audio: bytes,
    *,
    language: str,
    task: str,
    max_new_tokens: int,
    temperature: float,
    stream: bool,
):
    """Decode ``audio`` into Whisper features and the engine request.

    Raises ``ValueError`` for a ``language`` the tokenizer does not know.
    """
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
        stream=stream,
    )
    request.precomputed_multimodal_inputs = MultimodalInputs(
        mm_items=[
            MultimodalDataItem(
                modality=Modality.AUDIO, feature=features[0].contiguous()
            )
        ]
    )
    return request


def _stop_fields(finish_reason: dict) -> dict:
    """vLLM's ``finish_reason`` / ``stop_reason`` for an engine finish reason.

    This endpoint sets no stop strings or stop token ids, so ``stop`` can
    only be end-of-text, which vLLM reports with ``stop_reason`` null.
    """
    return {"finish_reason": finish_reason["type"], "stop_reason": None}


def _event(payload: dict) -> str:
    # Raw UTF-8 like vLLM's pydantic dump, not \uXXXX escapes.
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _sse_chunks(
    engine,
    request,
    *,
    task: str,
    model: str,
    include_usage: bool,
    continuous_usage: bool,
) -> AsyncIterator[str]:
    """SSE events for one streamed request, in vLLM 0.32.0's schema."""
    chunk_id = f"{task}-{uuid.uuid4().hex}"
    head = {
        "id": chunk_id,
        "object": (
            "transcription.chunk" if task == "transcribe" else "translation.chunk"
        ),
        "created": int(time.time()),
        "model": model,
    }
    usage = None
    sent = 0
    try:
        async for out in engine.llm.generate_stream_async(request):
            meta = out["meta_info"]
            if include_usage:
                # Both counts are request totals, not per-chunk.
                prompt_tokens = int(meta["prompt_tokens"])
                completion_tokens = int(meta["completion_tokens"])
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "completion_tokens": completion_tokens,
                }
            # The engine's text is cumulative; send what is new.
            text = out["text"]
            delta = text[sent:]
            sent = len(text)
            finish_reason = meta.get("finish_reason")
            if not delta and finish_reason is None:
                continue
            choice = {"delta": {"content": delta}}
            if finish_reason is not None:
                choice.update(_stop_fields(finish_reason))
            chunk = {**head, "choices": [choice]}
            if continuous_usage and usage is not None:
                chunk["usage"] = usage
            yield _event(chunk)
        if include_usage and usage is not None:
            yield _event({**head, "choices": [], "usage": usage})
    except Exception as exc:  # noqa: BLE001 -- reported in-band, as vLLM does
        # The 200 and headers are gone; the error can only travel in-band.
        logger.exception(f"asr stream {chunk_id} failed")
        error = {"message": str(exc), "type": type(exc).__name__, "code": 500}
        yield _event({"error": error})
    yield "data: [DONE]\n\n"


def create_app(engine, processor, served_model_name: str | list[str]) -> FastAPI:
    app = FastAPI(title="TokenSpeed ASR server")
    app.state.engine = engine
    app.state.processor = processor
    extract_slots = threading.Semaphore(_EXTRACT_CONCURRENCY)
    if isinstance(served_model_name, str):
        served_names = [served_model_name]
    else:
        served_names = list(served_model_name)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        now = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": now,
                    "owned_by": "tokenspeed",
                }
                for name in served_names
            ],
        }

    def _handle(
        file: UploadFile,
        model: str,
        language: str | None,
        response_format: str,
        temperature: float,
        task: str,
        stream: bool,
        stream_include_usage: bool,
        stream_continuous_usage_stats: bool,
    ):
        audio = file.file.read()
        if not audio:
            return JSONResponse({"error": "empty audio file"}, status_code=400)
        # See the module docstring: no detection pass, so an absent language
        # is answered with a stated default rather than a guess.
        resolved = language or "en"
        try:
            with extract_slots:
                request = _build_request(
                    app.state.processor,
                    audio,
                    language=resolved,
                    task=task,
                    max_new_tokens=448,
                    temperature=temperature,
                    stream=stream,
                )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if stream:
            return StreamingResponse(
                _sse_chunks(
                    app.state.engine,
                    request,
                    task=task,
                    model=model or served_names[0],
                    include_usage=stream_include_usage,
                    continuous_usage=(
                        stream_include_usage and stream_continuous_usage_stats
                    ),
                ),
                media_type="text/event-stream",
            )

        text = app.state.engine.llm.generate(request)["text"]
        if response_format == "text":
            return PlainTextResponse(text)
        return {"text": text, "language": resolved}

    # `stream` and its two usage options are OpenAI/vLLM's optional form
    # fields; omitted means a single JSON (or text) body.
    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        language: str | None = Form(default=None),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
        stream: bool = Form(default=False),
        stream_include_usage: bool = Form(default=False),
        stream_continuous_usage_stats: bool = Form(default=False),
    ):
        return _handle(
            file,
            model,
            language,
            response_format,
            temperature,
            "transcribe",
            stream,
            stream_include_usage,
            stream_continuous_usage_stats,
        )

    @app.post("/v1/audio/translations")
    def translations(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        response_format: str = Form(default="json"),
        temperature: float = Form(default=0.0),
        stream: bool = Form(default=False),
        stream_include_usage: bool = Form(default=False),
        stream_continuous_usage_stats: bool = Form(default=False),
    ):
        # Translation is always *into* English, so the language slot still
        # names the source; Whisper handles that from the audio itself and
        # the task token is what changes.
        return _handle(
            file,
            model,
            None,
            response_format,
            temperature,
            "translate",
            stream,
            stream_include_usage,
            stream_continuous_usage_stats,
        )

    return app


def serve(argv: list[str], *, host: str, port: int) -> None:
    """Build the engine and its processor from ``argv``, then serve."""
    import torch
    import uvicorn
    from transformers import AutoProcessor

    from tokenspeed.runtime.entrypoints.engine import Engine
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    server_args = prepare_server_args(argv)
    processor = AutoProcessor.from_pretrained(server_args.model)
    engine = Engine(server_args=server_args)
    # Process-wide, so set here and not in create_app: this process only
    # extracts features and relays requests; the schedulers are spawned
    # processes with their own thread settings.
    torch.set_num_threads(_EXTRACT_TORCH_THREADS)
    served = server_args.served_model_names
    if isinstance(served, (list, tuple)):
        served = served[0]
    logger.info(f"asr server: model={served}")
    app = create_app(engine, processor, served_model_name=served)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        engine.shutdown()

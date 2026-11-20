# Copyright (c) 2026 LightSeek Foundation
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

"""OpenAI-compatible ``/v1/embeddings`` server for pooling models.

``ts serve`` fronts generative models with the smg gateway, which reaches the
engine over ``tokenspeed_scheduler.proto``. That proto has no embedding RPC,
so an embedding server owns its engine in-process instead and answers HTTP
directly through ``Engine.encode``.

Handlers are plain ``def``: FastAPI runs them in its threadpool, and each
call goes through the engine's sync facade, which owns the event loop the
tokenizer manager is bound to. Awaiting the tokenizer manager from uvicorn's
loop would submit on one loop and complete on another. Concurrent requests
still batch: every worker thread submits its own request to the scheduler.
"""

from __future__ import annotations

import base64
import hmac
import time
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

__all__ = ["create_app", "serve"]

_ENCODING_FORMATS = ("float", "base64")


def _error(status_code: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "code": status_code}},
        status_code=status_code,
    )


def parse_embedding_input(
    value: Any,
) -> tuple[list[str] | None, list[list[int]] | None]:
    """Split OpenAI's ``input`` into texts or token-id lists.

    ``input`` is a string, a list of strings, a list of ints (one tokenized
    input) or a list of lists of ints. The two integer forms are only told
    apart by looking inside; reading one as the other would embed the wrong
    sequence.
    """
    if isinstance(value, str):
        return [value], None
    if not isinstance(value, list) or not value:
        raise ValueError("`input` must be a non-empty string or list")
    if all(isinstance(item, str) for item in value):
        return list(value), None
    # bool is an int subclass; a list of booleans is not a token sequence.
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None, [list(value)]
    if all(
        isinstance(item, list)
        and item
        and all(isinstance(i, int) and not isinstance(i, bool) for i in item)
        for item in value
    ):
        return None, [list(item) for item in value]
    raise ValueError(
        "`input` must be a string, a list of strings, a list of ints, or a "
        "list of non-empty lists of ints"
    )


def _encode_vector(vector: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "base64":
        return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode(
            "ascii"
        )
    return vector


def create_app(engine, served_model_name: str, api_key: str | None) -> FastAPI:
    """Build the HTTP app around a running embedding ``Engine``.

    Args:
        engine: An ``Engine`` serving a pooling checkpoint.
        served_model_name: The name clients pass as ``model``; any other name
            is answered with 404 rather than served silently.
        api_key: When set, every ``/v1/*`` request must carry
            ``Authorization: Bearer <api_key>``. None serves without
            authentication.

    Returns:
        The FastAPI application.
    """
    app = FastAPI(title="TokenSpeed embedding server")

    def _unauthorized(request: Request) -> JSONResponse | None:
        if api_key is None:
            return None
        supplied = request.headers.get("authorization", "")
        if hmac.compare_digest(supplied, f"Bearer {api_key}"):
            return None
        return _error(401, "invalid or missing API key", "authentication_error")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models(request: Request):
        if (denied := _unauthorized(request)) is not None:
            return denied
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

    @app.post("/v1/embeddings")
    def embeddings(request: Request, body: dict = Body(...)):
        if (denied := _unauthorized(request)) is not None:
            return denied
        model = body.get("model")
        if model is not None and model != served_model_name:
            return _error(404, f"model {model!r} is not served here", "model_not_found")
        encoding_format = body.get("encoding_format") or "float"
        if encoding_format not in _ENCODING_FORMATS:
            return _error(
                400,
                f"unsupported encoding_format {encoding_format!r}; "
                f"supported: {list(_ENCODING_FORMATS)}",
                "invalid_request_error",
            )
        if body.get("dimensions") is not None:
            # Truncating a vector changes its meaning; refuse rather than
            # return full-width vectors to a caller that asked for fewer.
            return _error(400, "`dimensions` is not supported", "invalid_request_error")
        try:
            texts, input_ids = parse_embedding_input(body.get("input"))
            outputs = engine.encode(prompt=texts, input_ids=input_ids)
        except ValueError as exc:
            return _error(400, str(exc), "invalid_request_error")
        if isinstance(outputs, dict):
            outputs = [outputs]

        prompt_tokens = sum(item["meta_info"]["prompt_tokens"] for item in outputs)
        return {
            "object": "list",
            "model": served_model_name,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": _encode_vector(item["embedding"], encoding_format),
                }
                for index, item in enumerate(outputs)
            ],
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }

    return app


def serve(server_args, *, host: str, port: int, api_key: str | None) -> None:
    """Start an embedding ``Engine`` and serve ``/v1/embeddings`` on it.

    Args:
        server_args: Engine ``ServerArgs`` for a pooling checkpoint.
        host: HTTP bind host.
        port: HTTP bind port.
        api_key: Bearer key required on ``/v1/*``; None disables the check.
    """
    import uvicorn

    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(server_args=server_args)
    if not engine.server_args.is_embedding:
        engine.shutdown()
        raise RuntimeError(
            f"{server_args.model} did not start as an embedding model; serve it "
            "through the generative `ts serve` path instead."
        )
    served = server_args.served_model_name or server_args.model
    logger.info(
        f"embedding server: model={served} "
        f"pooling_type={engine.server_args.pooling_type} "
        f"normalize={engine.server_args.pooling_normalize} "
        f"listening on http://{host}:{port}"
    )
    if api_key is None:
        logger.warning(
            "embedding server started without --api-key: /v1/* accepts "
            "unauthenticated requests"
        )
    app = create_app(engine, served_model_name=served, api_key=api_key)
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        engine.shutdown()

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

"""Ref2VA / FL2VA multipart reference intake for the diffusion HTTP surface.

HTTP contract (OpenAI-shaped ``/v1/videos`` form, TokenSpeed-native):

* Repeat ``input_references`` once per media file; MIME (or filename suffix)
  selects image / video / audio. Form order is semantic and preserved.
* Legacy single ``input_reference`` + optional ``last_image`` still work for
  FL2VA first/last keyframes.
* Optional ``audio_reference`` JSON ``{"audio_url": "..."}`` appends one audio
  after the multipart list (Ref2VA).

Limits match MiniMax-H3: images ≤9, videos ≤3, audios ≤3, total ≤12; audio
requires at least one visual reference.
"""

from __future__ import annotations

import mimetypes
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

__all__ = [
    "MAX_REF_IMAGES",
    "MAX_REF_VIDEOS",
    "MAX_REF_AUDIOS",
    "MAX_REF_TOTAL",
    "ReferenceUpload",
    "classify_media",
    "validate_ref2va_counts",
    "stage_uploads",
    "cleanup_staged",
]

MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
MAX_REF_TOTAL = 12

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".heic", ".heif"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass(frozen=True)
class ReferenceUpload:
    """One staged reference file with a modality tag."""

    path: str
    kind: str  # image | video | audio
    filename: str | None = None
    content_type: str | None = None


def _suffix(name: str | None) -> str:
    if not name:
        return ""
    return os.path.splitext(name)[1].lower()


def classify_media(*, filename: str | None = None, content_type: str | None = None) -> str:
    """Return ``image`` / ``video`` / ``audio`` from MIME or filename suffix."""
    ctype = (content_type or "").split(";")[0].strip().lower()
    if ctype.startswith("image/"):
        return "image"
    if ctype.startswith("video/"):
        return "video"
    if ctype.startswith("audio/"):
        return "audio"

    suffix = _suffix(filename)
    if suffix in _IMAGE_SUFFIXES:
        return "image"
    if suffix in _VIDEO_SUFFIXES:
        return "video"
    if suffix in _AUDIO_SUFFIXES:
        return "audio"

    if filename and not ctype:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return classify_media(filename=filename, content_type=guessed)

    raise ValueError(
        f"cannot classify reference media (filename={filename!r}, content_type={content_type!r}); "
        "use image/*, video/*, or audio/* MIME or a known suffix"
    )


def validate_ref2va_counts(kinds: Sequence[str]) -> None:
    """Enforce MiniMax-H3 Ref2VA cardinality rules."""
    n_img = sum(1 for k in kinds if k == "image")
    n_vid = sum(1 for k in kinds if k == "video")
    n_aud = sum(1 for k in kinds if k == "audio")
    total = len(kinds)
    if total == 0:
        raise ValueError("ref2va requires at least one image or video reference")
    if total > MAX_REF_TOTAL:
        raise ValueError(f"ref2va allows at most {MAX_REF_TOTAL} references (got {total})")
    if n_img > MAX_REF_IMAGES:
        raise ValueError(f"ref2va allows at most {MAX_REF_IMAGES} images (got {n_img})")
    if n_vid > MAX_REF_VIDEOS:
        raise ValueError(f"ref2va allows at most {MAX_REF_VIDEOS} videos (got {n_vid})")
    if n_aud > MAX_REF_AUDIOS:
        raise ValueError(f"ref2va allows at most {MAX_REF_AUDIOS} audios (got {n_aud})")
    if n_aud and (n_img + n_vid) == 0:
        raise ValueError("ref2va audio references require at least one image or video")


def stage_uploads(
    items: Iterable[tuple[bytes, str | None, str | None]],
    *,
    directory: str | None = None,
) -> list[ReferenceUpload]:
    """Write upload bytes to temp files; caller must ``cleanup_staged`` afterward."""
    staged: list[ReferenceUpload] = []
    base = directory or tempfile.mkdtemp(prefix="ts-h3-refs-")
    os.makedirs(base, exist_ok=True)
    for index, (data, filename, content_type) in enumerate(items):
        if not data:
            raise ValueError(f"empty reference upload at index {index}")
        kind = classify_media(filename=filename, content_type=content_type)
        suffix = _suffix(filename) or {
            "image": ".png",
            "video": ".mp4",
            "audio": ".wav",
        }[kind]
        path = os.path.join(base, f"ref_{index:02d}{suffix}")
        with open(path, "wb") as handle:
            handle.write(data)
        staged.append(
            ReferenceUpload(
                path=path,
                kind=kind,
                filename=filename,
                content_type=content_type,
            )
        )
    return staged


def cleanup_staged(staged: Sequence[ReferenceUpload]) -> None:
    """Best-effort delete of staged files and their parent temp dir."""
    parents: set[str] = set()
    for item in staged:
        parents.add(os.path.dirname(item.path))
        try:
            os.remove(item.path)
        except OSError:
            pass
    for parent in parents:
        try:
            os.rmdir(parent)
        except OSError:
            pass


def audio_url_from_field(raw: str | None) -> str | None:
    """Parse optional ``audio_reference`` JSON ``{"audio_url": "..."}`` or bare URL."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        import json

        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"audio_reference must be JSON or a URL: {exc}") from exc
        if not isinstance(payload, dict) or "audio_url" not in payload:
            raise ValueError('audio_reference JSON must contain "audio_url"')
        url = str(payload["audio_url"]).strip()
    else:
        url = text
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "file", "data"} and not os.path.isfile(url):
        raise ValueError(f"unsupported audio_reference URL: {url!r}")
    return url

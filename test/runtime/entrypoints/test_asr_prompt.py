"""The Whisper decoder prompt, which is where `language` and `task` live."""

import pytest

from tokenspeed.runtime.entrypoints.asr_http import _prompt_tokens


class _Tokenizer:
    """A tokenizer with Whisper's habit: unknown tokens map to unk, not None."""

    def __init__(self, known):
        self._known = {t: i for i, t in enumerate(known)}
        self._unk = len(known)

    def convert_tokens_to_ids(self, tokens):
        return [self._known.get(t, self._unk) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        back = {i: t for t, i in self._known.items()}
        return [back.get(i, "<|unk|>") for i in ids]


KNOWN = [
    "<|startoftranscript|>",
    "<|en|>",
    "<|zh|>",
    "<|transcribe|>",
    "<|translate|>",
    "<|notimestamps|>",
]


def test_prompt_is_start_language_task_notimestamps():
    tok = _Tokenizer(KNOWN)
    ids = _prompt_tokens(tok, "en", "transcribe")
    assert tok.convert_ids_to_tokens(ids) == [
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    ]


def test_task_token_is_what_separates_translate_from_transcribe():
    tok = _Tokenizer(KNOWN)
    transcribe = _prompt_tokens(tok, "zh", "transcribe")
    translate = _prompt_tokens(tok, "zh", "translate")
    assert transcribe != translate
    assert tok.convert_ids_to_tokens(translate)[2] == "<|translate|>"


def test_unknown_language_is_refused_not_silently_unked():
    # The reason this is a round-trip and not a None check: an unknown token
    # becomes unk_token_id, so a None check returns 200 and Whisper decodes
    # from a prompt it cannot mean. Measured against a running server before
    # the fix.
    tok = _Tokenizer(KNOWN)
    with pytest.raises(ValueError, match="zzz"):
        _prompt_tokens(tok, "zzz", "transcribe")


def test_unknown_task_is_refused_too():
    tok = _Tokenizer(KNOWN)
    with pytest.raises(ValueError, match="summarise"):
        _prompt_tokens(tok, "en", "summarise")

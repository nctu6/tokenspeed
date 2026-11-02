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

"""CPU-only: the split (conditioner | rest) call keeps the requested canvas."""

from __future__ import annotations

from tokenspeed.runtime.diffusion.pipeline import _split_call_kwargs


def _t2va_call() -> dict:
    return {
        "prompt": "rain at night",
        "num_frames": 124,
        "num_inference_steps": 50,
        "output": ["videos", "audio", "sampling_rate"],
        "width": 672,
        "height": 384,
        "generator": object(),
    }


def test_rest_pipeline_receives_requested_geometry():
    # Regression: the denoise side used to lose width/height/num_frames and
    # render the default 1344x768 canvas for a 672x384 t2va request.
    _, rest = _split_call_kwargs(_t2va_call())
    assert (rest["width"], rest["height"], rest["num_frames"]) == (672, 384, 124)


def test_conditioner_receives_prompt_refs_and_geometry():
    call = _t2va_call()
    call["image"] = "first-frame"
    cond, _ = _split_call_kwargs(call)
    assert cond == {
        "prompt": "rain at night",
        "image": "first-frame",
        "width": 672,
        "height": 384,
        "num_frames": 124,
    }


def test_conditioner_only_inputs_stay_off_the_rest_pipeline():
    call = _t2va_call()
    call["references"] = ["ref"]
    _, rest = _split_call_kwargs(call)
    assert not {"prompt", "image", "last_image", "references"} & rest.keys()
    assert rest["num_inference_steps"] == 50
    assert rest["generator"] is call["generator"]

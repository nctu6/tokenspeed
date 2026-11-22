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
"""CPU-only: resident H3 components are placed on the device only when offload is off."""

from __future__ import annotations

import torch

from tokenspeed.runtime.diffusion.pipeline import place_resident


def test_no_offload_moves_weights_to_the_device():
    # "meta" stands in for a GPU: the weights must leave the CPU.
    module = torch.nn.Linear(4, 4)
    placed = place_resident(module, "meta", enable_cpu_offload=False)
    assert all(p.device.type == "meta" for p in placed.parameters())


def test_offload_leaves_placement_to_the_components_manager():
    module = torch.nn.Linear(4, 4)
    placed = place_resident(module, "meta", enable_cpu_offload=True)
    assert all(p.device.type == "cpu" for p in placed.parameters())

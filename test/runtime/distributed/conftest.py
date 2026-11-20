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

"""Make this directory's shared test helpers (``_tp_layout_fakes``) importable
by bare module name under every pytest import mode; ``mp.spawn`` workers
inherit the path."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


import pytest


@pytest.fixture(autouse=True)
def _host_independent_nvls_probe(monkeypatch):
    """Keep routing unit tests independent of the host's NVLS fabric.

    ``nvls_multicast_mapping_usable`` probes the real driver (Fabric
    ClusterUUID) and returns ``False`` on hosts whose multicast bind is
    broken, which vetoes every multimem route before the tests' own
    topology/fabric fakes are consulted. Report "inconclusive" (``None``),
    the value that leaves routing to the faked topology; a test that wants
    the broken-host verdict patches it back explicitly.
    """
    from tokenspeed.runtime.distributed import nvls_multicast

    monkeypatch.setattr(nvls_multicast, "nvls_multicast_mapping_usable", lambda: None)


@pytest.fixture(autouse=True)
def _host_independent_amd_fabric(monkeypatch):
    """Keep routing tests independent of the host's AMD GPU fabric.

    On PCIe-only AMD hosts the auto backend skips every symmetric-memory
    route; these tests exercise the XGMI defaults. The environment variable
    (unlike a monkeypatched function) also reaches ``mp.spawn`` workers.
    """
    monkeypatch.setenv("TOKENSPEED_AMD_GPU_FABRIC", "xgmi")

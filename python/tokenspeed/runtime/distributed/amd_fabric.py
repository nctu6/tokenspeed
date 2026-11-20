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

"""Whether this host's AMD GPUs reach each other over XGMI or only PCIe.

Iris all-reduce loads and stores through peer GPU memory. On PCIe-only hosts
(for example MI350P cards) it beats RCCL at every all-reduce size measured,
prefill activations included (4x MI350P, BF16: 42 MB 1.33 vs 1.87 ms,
168 MB 5.51 vs 6.19 ms), so the Iris backend lifts its size caps there.

``TOKENSPEED_AMD_GPU_FABRIC`` overrides the detection: ``auto`` (default)
reads the amdkfd topology, ``xgmi`` or ``pcie`` force the answer.
"""

import functools
from pathlib import Path

from tokenspeed.runtime.utils.env import envs

# amdkfd io-link types (include/uapi/linux/kfd_sysfs.h).
_KFD_LINK_TYPE_XGMI = 11
_KFD_NODES = Path("/sys/class/kfd/kfd/topology/nodes")


def _link_types(links_dir: Path) -> dict[int, int]:
    """Map peer KFD node id -> link type for one node's link directory."""
    links: dict[int, int] = {}
    if not links_dir.is_dir():
        return links
    for link in links_dir.iterdir():
        fields = dict(
            line.split(maxsplit=1)
            for line in (link / "properties").read_text().splitlines()
            if " " in line
        )
        links[int(fields["node_to"])] = int(fields["type"])
    return links


def local_gpus_pcie_only() -> bool:
    """True when this host's GPUs should be treated as PCIe-only peers."""
    fabric = envs.TOKENSPEED_AMD_GPU_FABRIC.get()
    if fabric == "auto":
        return _detected_pcie_only()
    if fabric not in ("xgmi", "pcie"):
        raise ValueError(
            f"TOKENSPEED_AMD_GPU_FABRIC must be auto, xgmi or pcie, got {fabric!r}"
        )
    return fabric == "pcie"


@functools.lru_cache(maxsize=1)
def _detected_pcie_only() -> bool:
    return kfd_gpus_pcie_only(_KFD_NODES)


def kfd_gpus_pcie_only(nodes_dir: Path) -> bool:
    """True when some pair of GPUs in an amdkfd topology has no XGMI link.

    Reads the amdkfd topology: GPU nodes are those with a nonzero
    ``gfx_target_version``; peers appear under ``io_links`` and ``p2p_links``.
    Returns False when the topology is unreadable or lists fewer than two
    GPUs, which keeps the XGMI defaults.
    """
    if not nodes_dir.is_dir():
        return False
    gpus: list[int] = []
    for node in nodes_dir.iterdir():
        props = node / "properties"
        if not node.name.isdigit() or not props.is_file():
            continue
        for line in props.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key == "gfx_target_version" and int(value) != 0:
                gpus.append(int(node.name))
    if len(gpus) < 2:
        return False
    for gpu in gpus:
        links = _link_types(nodes_dir / str(gpu) / "io_links")
        links.update(_link_types(nodes_dir / str(gpu) / "p2p_links"))
        for peer in gpus:
            if peer != gpu and links.get(peer) != _KFD_LINK_TYPE_XGMI:
                return True
    return False

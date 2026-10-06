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

"""Detect hosts where NVLS multicast bind is advertised but broken.

NCCL and Triton RSAG multimem both need a working ``cuMulticastBindMem``. On
some NVSwitch hosts (notably misconfigured Fabric Manager clusters) the driver
reports multicast support while bind returns ``CUDA_ERROR_ILLEGAL_STATE``
(401). NCCL then hard-fails at ``init_process_group``; RSAG multimem later
hits illegal memory access on a null multicast pointer.

The signal we use without a multi-process rendezvous: Fabric Manager reports
``ClusterUUID`` all zeros while the CUDA driver still advertises multicast
support. That pattern matches the bind-401 hosts we have measured; a real
cluster UUID leaves NVLS alone. Hosts without fabric (``ClusterUUID: N/A``)
are left alone too -- NCCL will not enable NVLS there.
"""

from __future__ import annotations

import logging
import re
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

_ZERO_CLUSTER_UUID = "00000000-0000-0000-0000-000000000000"
_UUID_RE = re.compile(
    r"ClusterUUID\s*:\s*([0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12})",
    re.MULTILINE,
)


def _driver_reports_multicast_support() -> bool | None:
    """Return whether the CUDA driver advertises NVLS multicast, or None."""
    try:
        import torch
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return None
        return bool(
            _SymmetricMemory.has_multicast_support(
                DeviceType.CUDA, torch.cuda.current_device()
            )
        )
    except Exception:  # noqa: BLE001 - capability probe must not raise
        return None


def _read_fabric_cluster_uuids() -> list[str] | None:
    """Parse ``nvidia-smi -q`` ClusterUUID values; None if unreadable."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return [m.group(1).lower() for m in _UUID_RE.finditer(proc.stdout)]


@lru_cache(maxsize=1)
def nvls_multicast_mapping_usable() -> bool | None:
    """Whether NVLS multicast bind is expected to work on this host.

    Returns:
        ``True`` when a non-zero Fabric ClusterUUID is present (leave NVLS
        defaults alone), ``False`` when the driver advertises multicast but
        every reported ClusterUUID is all zeros (auto-disable NCCL NVLS and
        skip Triton RSAG multimem), or ``None`` when the probe is
        inconclusive.
    """
    uuids = _read_fabric_cluster_uuids()
    if not uuids:
        return None
    if any(u != _ZERO_CLUSTER_UUID for u in uuids):
        return True
    # All reported UUIDs are zero. Only treat as broken when the driver also
    # claims multicast support -- otherwise NCCL will not enable NVLS anyway.
    advertised = _driver_reports_multicast_support()
    if advertised is False:
        return None
    if advertised is True:
        logger.warning(
            "NVLS multicast appears broken on this host: CUDA advertises "
            "multicast support but Fabric ClusterUUID is all zeros. NCCL "
            "NVLS init would hard-fail with cuMulticastBindMem "
            "CUDA_ERROR_ILLEGAL_STATE; Triton RSAG multimem would illegal-"
            "memory-access on a null multicast pointer. Auto-disabling "
            "NCCL NVLS and skipping multimem (same effect as "
            "--disable-nccl-nvls for the NVLS half; RSAG fusion/IPC paths "
            "stay available)."
        )
        return False
    # Could not query the driver (e.g. before CUDA init). Still treat a
    # zero-only ClusterUUID as broken: that is the measured mewtwo failure
    # mode, and a false disable only costs NVLS throughput -- not
    # correctness.
    logger.warning(
        "Fabric ClusterUUID is all zeros; treating NVLS multicast as "
        "unavailable (cannot confirm CUDA multicast advertisement yet)."
    )
    return False


def should_auto_disable_nccl_nvls() -> bool:
    """True when resolve_communication should force ``disable_nccl_nvls``."""
    return nvls_multicast_mapping_usable() is False

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

"""Collision-free local control-plane addresses for concurrent servers.

Two servers started at the same moment must not pick the same internal port.
Probing a port with ``bind`` and closing it (check-then-bind) cannot guarantee
that: both servers probe before either one binds. This module avoids it two
ways:

* ``create_ipc_dir`` gives a server a private directory for ``ipc://`` sockets,
  so same-host ZMQ endpoints need no port at all.
* ``claim_port`` claims a TCP port by binding an abstract Unix socket named
  after it. The bind is atomic and visible to every process on the host, and
  the kernel drops it when the claiming process (and any child that inherited
  it) exits. Every TokenSpeed server claims before it uses a port, so two of
  them can never both hold the same one. A port taken by some other program is
  rejected by a real bind probe at claim time.
"""

from __future__ import annotations

import atexit
import os
import random
import shutil
import socket
import sys
import tempfile

from tokenspeed.runtime.utils.network import is_port_available

# Longest ``sun_path`` (bytes) that fits ``sockaddr_un`` on Linux, minus the NUL.
_MAX_UNIX_SOCKET_PATH = 107
# Longest endpoint file name created inside an IPC directory.
_LONGEST_IPC_FILE_NAME = len("scheduler_input")

# Ports sampled by ``reserve_port``: below the kernel's ephemeral range, so an
# unrelated outbound connection cannot be handed the same port as its source.
_RESERVE_PORT_LOW = 10000
_RESERVE_PORT_HIGH_DEFAULT = 32767
_EPHEMERAL_RANGE_PATH = "/proc/sys/net/ipv4/ip_local_port_range"
_RESERVE_MAX_ATTEMPTS = 200

# Sockets that hold claims, kept alive for the life of this process.
_held_claims: dict[int, socket.socket] = {}
# IPC directories created by this process, removed when it exits.
_owned_ipc_dirs: list[str] = []
_owner_pid = os.getpid()


def _claim_name(port: int) -> bytes:
    return f"\0tokenspeed-port-{port}".encode()


def claim_port(port: int) -> bool:
    """Atomically claim ``port`` for this process.

    Args:
        port: TCP port the caller is about to use.

    Returns:
        True if this process now holds the claim and a bind probe of the port
        succeeds; False if another TokenSpeed server holds the claim or another
        program is using the port. Claiming a port this process already holds
        returns True.
    """
    if not sys.platform.startswith("linux"):
        raise NotImplementedError(
            f"port claims use the Linux abstract socket namespace; got {sys.platform}"
        )
    if port in _held_claims:
        return True
    claim = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        claim.bind(_claim_name(port))
    except OSError:
        claim.close()
        return False
    if not is_port_available(port):
        claim.close()
        return False
    _held_claims[port] = claim
    return True


def release_port(port: int) -> None:
    """Drop this process's claim on ``port``, if it holds one."""
    claim = _held_claims.pop(port, None)
    if claim is not None:
        claim.close()


def _reserve_range() -> tuple[int, int]:
    try:
        with open(_EPHEMERAL_RANGE_PATH) as handle:
            ephemeral_low = int(handle.read().split()[0])
    except (OSError, ValueError, IndexError):
        return _RESERVE_PORT_LOW, _RESERVE_PORT_HIGH_DEFAULT
    return _RESERVE_PORT_LOW, min(_RESERVE_PORT_HIGH_DEFAULT, ephemeral_low - 1)


def reserve_port() -> int:
    """Pick a free TCP port and claim it for this process.

    Returns:
        A port outside the ephemeral range that no other TokenSpeed server holds
        and that no program was listening on when it was claimed.

    Raises:
        RuntimeError: If no port could be claimed after many random attempts.
    """
    low, high = _reserve_range()
    for _ in range(_RESERVE_MAX_ATTEMPTS):
        port = random.randint(low, high)
        if claim_port(port):
            return port
    raise RuntimeError(
        f"could not claim a free port in {low}-{high} after "
        f"{_RESERVE_MAX_ATTEMPTS} attempts"
    )


def _remove_owned_ipc_dirs() -> None:
    # Forked children inherit this hook; only the creating process cleans up.
    if os.getpid() != _owner_pid:
        return
    while _owned_ipc_dirs:
        shutil.rmtree(_owned_ipc_dirs.pop(), ignore_errors=True)


atexit.register(_remove_owned_ipc_dirs)


def create_ipc_dir() -> str:
    """Create a directory private to this server for its ``ipc://`` sockets.

    The directory is removed when this process exits.

    Returns:
        The absolute directory path.

    Raises:
        ValueError: If the temp directory path is too long for a Unix socket.
    """
    path = tempfile.mkdtemp(prefix="ts-")
    longest = len(os.path.join(path, "x" * _LONGEST_IPC_FILE_NAME).encode())
    if longest > _MAX_UNIX_SOCKET_PATH:
        shutil.rmtree(path, ignore_errors=True)
        raise ValueError(
            f"the temp directory {tempfile.gettempdir()!r} is too long for "
            f"ipc:// sockets ({longest} > {_MAX_UNIX_SOCKET_PATH} bytes); "
            "set TMPDIR to a shorter path"
        )
    _owned_ipc_dirs.append(path)
    return path

#!/usr/bin/env bash
#
# install.sh -- venv install of the TokenSpeed stack
# ==========================================================
#
# Line-for-line port of the NVIDIA TokenSpeed Docker install to a local Python
# virtualenv, for installing on a remote NVIDIA box (e.g. RTX PRO 6000 / sm_120)
# that already has the CUDA toolkit and a matching PyTorch.
#
# What this reproduces from the Dockerfile, and why the order matters:
#
#   1. CUDA_ARCH_LIST = "9.0a 10.0a 12.0a" -- one fat binary spanning the
#      datacenter tier (sm_90 Hopper, sm_100 Blackwell) and the workstation tier
#      (sm_120). tcgen05 groups build for sm_90/sm_100 and are auto-skipped for
#      sm_120 by the kernel's setup.py arch gate, so this list compiles cleanly.
#
#   2. THE KERNEL IS (RE)INSTALLED FROM SOURCE LAST. The engine (python/)
#      declares `tokenspeed-kernel>=0.1.3.dev0`, so `pip install ./python`
#      re-resolves it and OVERWRITES the source kernel built in step 1 with an
#      older published wheel (0.1.3) that is missing symbols the engine imports
#      (e.g. allgather_dual_rmsnorm). Installing the engine first, THEN
#      reinstalling the local kernel + scheduler with --no-deps
#      --force-reinstall, guarantees the source builds win in the final env.
#
# EDITABLE INSTALLS (`pip install -e`) for the ENGINE and SCHEDULER only.
# A synced source edit to the engine -- e.g. a new model in
# python/tokenspeed/runtime/models/ -- then takes effect on the next server
# start with NO reinstall. This is what shortens the Gemma-3 iteration loop.
#
# `tokenspeed-kernel` is installed NON-editable on purpose. Its editable
# ("0.editable") wheel uses a strict import finder that does not expose deeply
# nested subpackages, so the engine's
#   from tokenspeed_kernel.ops.attention.triton.linear.chunk_delta_h import ...
# fails at startup with ModuleNotFoundError under `-e`. A normal install copies
# the whole ops/ tree into site-packages and resolves fine. The kernel is a
# compiled extension anyway, so a source change needs a rebuild regardless --
# editable buys it nothing.
#
# Unlike the Docker build, this DOES import-free verify only; a GPU is assumed
# present on the target, but we still avoid importing the kernel during install
# so the script works the same whether or not a GPU is visible at install time.
#
# IDEMPOTENT BY DEFAULT. This is a from-source builder -- the kernel step
# compiles CUDA, which is slow -- so a plain re-run SKIPS anything already
# installed. The expensive kernel compile happens ONCE (first install) and is
# skipped thereafter. Because the engine and scheduler are editable, a synced
# Python edit (e.g. a model under python/tokenspeed/runtime/models/) needs NO
# reinstall at all -- just restart the server. So the normal loop is: sync +
# restart, and you only re-run this script when a package is genuinely missing.
#
# Force a full from-source rebuild/reinstall of everything:
#   REINSTALL=1 ./scripts/install.sh      (or: ./scripts/install.sh --force)
# Force just the kernel to recompile (after a C++/CUDA change):
#   REINSTALL_KERNEL=1 ./scripts/install.sh
# Force just the smg gateway to rebuild (after editing smg/, Rust):
#   REINSTALL_SMG=1 ./scripts/install.sh  (or: ./scripts/install.sh --force-smg)
#
# The smg gateway is a git submodule. On a git checkout this script runs
# `git submodule update --init smg` when smg/ is empty, so a fresh clone builds
# the gateway too. When present (submodule initialized, or an rsync'd tree), the
# OpenAI gateway is built from THAT source so it carries local patches (e.g. the
# best_of+stream validator fix), replacing the published tokenspeed-smg* wheels.
# Building the gateway needs a Rust toolchain, protoc, and maturin -- install.sh
# installs them if missing.
#
# VENV SELECTION. If a venv is ALREADY active, this script installs INTO it
# instead of creating a second one. "Active" is detected from the interpreter
# (sys.prefix != sys.base_prefix), so it works both when the venv was entered
# via `source .venv_docker/bin/activate` AND when a Docker image merely puts the
# venv on PATH (ENV PATH=/opt/venv/bin:$PATH) without exporting $VIRTUAL_ENV.
# Otherwise it falls back to ./.venv. An explicit VENV_DIR=... overrides both.
#
# Usage:
#   ./scripts/install.sh                 # adopt active $VIRTUAL_ENV, else create/use ./.venv
#   source .venv_docker/bin/activate && ./scripts/install.sh   # install into .venv_docker
#   REINSTALL=1 ./scripts/install.sh     # force full from-source rebuild of all packages
#   REINSTALL_KERNEL=1 ./scripts/install.sh  # force only the kernel to recompile
#   VENV_DIR=/path/to/venv ./scripts/install.sh   # force a specific venv (overrides active)
#   MAX_JOBS=8 ./scripts/install.sh
#   CUDA_ARCH_LIST="12.0a" ./scripts/install.sh   # build only for sm_120 (faster)
#
# Run this from the repository root (the directory that contains python/,
# tokenspeed-kernel/, tokenspeed-scheduler/).
set -euo pipefail

# --- Resolve repo root (this script lives at the repo root) ------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# scripts/ lives one level below the repo root; keep .venv and package paths at REPO_ROOT.
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

# --- Tunables (mirror the Dockerfile ARG/ENV values) -------------------------
MAX_JOBS="${MAX_JOBS:-16}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-9.0a 10.0a 12.0a}"
# VENV_DIR intentionally has NO default here. We resolve it below with this
# precedence: an explicit VENV_DIR wins; otherwise an already-active venv
# ($VIRTUAL_ENV, e.g. a .venv_docker the user sourced inside a container) is
# adopted as-is; otherwise we fall back to $REPO_ROOT/.venv. Defaulting it here
# would erase the distinction between "user asked for this path" and "we picked
# the fallback", which is exactly what lets us honor an active venv.
VENV_DIR="${VENV_DIR:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Force flags. REINSTALL forces every package to rebuild/reinstall from source;
# REINSTALL_KERNEL forces only the (expensive) kernel recompile. `--force` is a
# synonym for REINSTALL=1.
REINSTALL="${REINSTALL:-0}"
REINSTALL_KERNEL="${REINSTALL_KERNEL:-0}"
# REINSTALL_SMG forces just the smg gateway (Rust) to rebuild from local source
# -- use after editing smg/ (e.g. the best_of+stream validator fix).
REINSTALL_SMG="${REINSTALL_SMG:-0}"
# REINSTALL_FA2 forces a rebuild of the (opt-in) vllm_flash_attn FA2 path.
# BUILD_FA2 opts into building it at all -- it is OFF by default because it is
# an optional prefill speedup whose upstream package can shadow the FA4
# `flash_attn.cute` namespace the kernel needs. Both are equivalent triggers.
REINSTALL_FA2="${REINSTALL_FA2:-0}"
BUILD_FA2="${BUILD_FA2:-0}"
for arg in "$@"; do
    case "$arg" in
        --force) REINSTALL=1 ;;
        --force-kernel) REINSTALL_KERNEL=1 ;;
        --force-smg) REINSTALL_SMG=1 ;;
        --force-fa2) REINSTALL_FA2=1 ;;
        --with-fa2) BUILD_FA2=1 ;;
    esac
done

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
skip() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
err() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; }

# True when a distribution is already importable-installed in the venv.
have_dist() { "$PY" -m pip show "$1" >/dev/null 2>&1; }

# True when a top-level module can be located WITHOUT importing it (so no GPU
# init, no CUDA JIT). Used for vllm_flash_attn, whose distribution name has
# varied across the fork's history but whose import name is stable.
have_module() {
    "$PY" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec(sys.argv[1]) is not None else 1)
PY
}

# --- Sanity: are we in the right place? --------------------------------------
for d in python tokenspeed-kernel/python tokenspeed-scheduler; do
    if [[ ! -d "$d" ]]; then
        err "expected directory '$d' not found. Run this script from the repository root."
        exit 1
    fi
done

# The smg gateway is tracked as a git submodule (see .gitmodules). If this is a
# git checkout and the submodule has not been populated yet (fresh `git clone`
# without --recurse-submodules leaves smg/ empty), initialize it so the gateway
# build below has source to build. No-op when smg/ is already populated (e.g. an
# rsync'd tree or an already-initialized submodule) or when this is not a git
# checkout -- the subsequent HAVE_SMG_SRC check still governs whether we build.
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if grep -q '\[submodule "smg"\]' .gitmodules 2>/dev/null \
        && [[ ! -f smg/bindings/python/pyproject.toml ]]; then
        log "Initializing the smg submodule (git submodule update --init smg)"
        git submodule update --init --recursive smg || \
            err "git submodule update --init smg failed; the smg gateway will not be built from source."
    fi
fi

# The smg gateway source is optional: present it, we build the local gateway
# (with the best_of+stream fix) from source; absent, we leave whatever
# tokenspeed-smg* wheels the environment already has.
HAVE_SMG_SRC=0
if [[ -d smg/bindings/python && -d smg/grpc_servicer && -d smg/crates/grpc_client/python ]]; then
    HAVE_SMG_SRC=1
fi

# --- Virtualenv: adopt an active one, else create/reuse, then ACTIVATE -------
# Done before anything else so the user never has to remember to activate it --
# every pip/python call below runs inside the venv, and the activated shell is
# what they inherit after the script finishes.
#
# Resolution precedence (first match wins):
#   1. Explicit VENV_DIR=... from the caller -- always honored, created if
#      missing. The caller asked for a specific path; respect it verbatim.
#   2. An ALREADY-ACTIVE venv -- adopt it in place and do NOT create or switch
#      to .venv. This is the Docker case: a container whose venv is already live
#      should install INTO that venv, not have install.sh silently build a
#      second .venv beside it and leave the active one empty.
#   3. Neither -- fall back to $REPO_ROOT/.venv (the original default),
#      creating it on first run.
#
# DETECTING AN ACTIVE VENV. $VIRTUAL_ENV alone is NOT reliable: it is only
# exported when the venv was entered via `source bin/activate`. A Docker image
# that just puts the venv on PATH (`ENV PATH=/opt/venv/bin:$PATH`) runs entirely
# inside that venv yet leaves $VIRTUAL_ENV empty. The authoritative signal is
# the interpreter itself: inside a venv, Python's sys.prefix differs from
# sys.base_prefix. So we ask the active python3 directly, and treat $VIRTUAL_ENV
# only as the preferred source for the venv's PATH when it is set.
#
# detect_active_venv: echo the path of the venv the active `python3` belongs to,
# or nothing if python3 is not running inside a venv. Prefer $VIRTUAL_ENV (the
# activate-script case) and fall back to sys.prefix (the PATH-only case).
detect_active_venv() {
    "$PYTHON_BIN" - <<'PY' 2>/dev/null
import os, sys
if sys.prefix != sys.base_prefix:        # running inside a venv/virtualenv
    print(os.environ.get("VIRTUAL_ENV") or sys.prefix)
PY
}

ACTIVE_VENV=""
if [[ -z "$VENV_DIR" ]]; then
    ACTIVE_VENV="$(detect_active_venv)"
fi

if [[ -n "$VENV_DIR" ]]; then
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtualenv at $VENV_DIR (VENV_DIR override)"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        log "Reusing virtualenv at $VENV_DIR (VENV_DIR override)"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
elif [[ -n "$ACTIVE_VENV" ]]; then
    # A venv is already active (via `source bin/activate` OR just on PATH, e.g.
    # a .venv_docker inside a container). Adopt it as-is. If it was activated
    # the standard way its bin/ is already on PATH; if it was only PATH-injected
    # we still source its activate script so VIRTUAL_ENV and PATH are set
    # consistently for the pip/python calls below.
    VENV_DIR="$ACTIVE_VENV"
    log "Adopting the already-active virtualenv at $VENV_DIR"
    if [[ -z "${VIRTUAL_ENV:-}" && -f "$VENV_DIR/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    fi
else
    VENV_DIR="$REPO_ROOT/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        log "Creating virtualenv at $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        log "Reusing existing virtualenv at $VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

PY="$VENV_DIR/bin/python"
if [[ ! -x "$PY" ]]; then
    err "no python interpreter at $PY; the resolved venv ($VENV_DIR) looks incomplete."
    exit 1
fi
log "Activated venv; using interpreter: $("$PY" -c 'import sys; print(sys.executable)')"

# Export the build-time environment once; every source build below inherits it.
# (In the Dockerfile these ride on each RUN line; in one shell we export once.)
export MAX_JOBS
export FLASHINFER_CUDA_ARCH_LIST="$CUDA_ARCH_LIST"
export TOKENSPEED_KERNEL_BACKEND=cuda

# --- System dependencies (best-effort; needs sudo/apt) -----------------------
# libnuma1 is required at runtime by torch_memory_saver. libssl/openmpi/pkg-config
# are needed by parts of the build. Skipped automatically if apt-get is absent
# (e.g. non-Debian host) or if we lack privileges -- install these yourself then.
install_system_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        log "apt-get not found; skipping system deps. Ensure these are present: libssl-dev libopenmpi-dev libnuma1 pkg-config"
        return 0
    fi
    local SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        else
            log "not root and no sudo; skipping system deps. Ensure these are present: libssl-dev libopenmpi-dev libnuma1 pkg-config"
            return 0
        fi
    fi
    # ADDON (hardening): if every required package is already installed, skip
    # apt entirely. On hosts with an unrelated broken dpkg/DKMS state (e.g. a
    # half-configured nvidia-dkms for a kernel that was purged) any apt-get
    # install can abort under `set -e` even though nothing here needs
    # installing. FORCE_SYSTEM_DEPS=1 keeps the old always-run-apt behaviour.
    local _sys_pkgs=(libssl-dev libopenmpi-dev libnuma1 pkg-config)
    _sys_deps_present() {
        local p st
        for p in "${_sys_pkgs[@]}"; do
            st="$(dpkg-query -W -f='${Status}' "$p" 2>/dev/null || true)"
            [[ "$st" == "install ok installed" || "$st" == "hold ok installed" ]] || return 1
        done
        return 0
    }
    if [[ "${FORCE_SYSTEM_DEPS:-0}" != "1" ]] && command -v dpkg-query >/dev/null 2>&1 && _sys_deps_present; then
        skip "System dependencies already installed (${_sys_pkgs[*]}); skipping apt (FORCE_SYSTEM_DEPS=1 to run apt anyway)"
        return 0
    fi
    log "Installing system dependencies (libssl-dev libopenmpi-dev libnuma1 pkg-config)"
    # ADDON: tolerate apt failures caused by unrelated packages, as long as the
    # packages WE need end up installed. The apt path itself is unchanged.
    if ! { $SUDO apt-get update && $SUDO apt-get install -y --no-install-recommends \
        libssl-dev \
        libopenmpi-dev \
        libnuma1 \
        pkg-config; }; then
        if command -v dpkg-query >/dev/null 2>&1 && _sys_deps_present; then
            err "apt-get reported an error (likely an unrelated broken dpkg/DKMS package), but ${_sys_pkgs[*]} are installed; continuing."
            err "Inspect with: sudo dpkg --audit; sudo apt-get check"
        else
            err "apt-get failed and required system packages are missing: ${_sys_pkgs[*]}"
            return 1
        fi
    fi
}

# --- CUDA toolkit / nvcc discovery + optional apt install (ADDON) ------------
# tokenspeed-kernel/python/setup.py uses CUDA_HOME (default /usr/local/cuda) and
# FLASHINFER_NVCC (default $CUDA_HOME/bin/nvcc). Hosts without /usr/local/cuda
# used to fail deep in the wheel build with "nvcc was not found:
# /usr/local/cuda/bin/nvcc". Resolve a toolkit up front and export CUDA_HOME.
# An explicit CUDA_HOME that contains bin/nvcc always wins, and a host with
# /usr/local/cuda behaves exactly as before.
#
# Previously we ONLY resolved an existing nvcc and printed apt hints on failure.
# We did not auto-apt-install the toolkit because that needs sudo and can
# surprise multi-tenant hosts (driver/DKMS/kernel pulls). The user now wants
# detection + a guarded userspace-only install when nvcc is missing.
#
# Discovery order (resolve_cuda_home):
#   1. $CUDA_HOME/bin/nvcc   2. $FLASHINFER_NVCC   3. /usr/local/cuda
#   4. `nvcc` on PATH        5. /usr/local/cuda-* (torch's CUDA major first,
#      then newest)  6. /opt/cuda   7. pip wheels in the venv (nvidia/cu*/bin/nvcc,
#      only if nvcc's cicc, cuda_runtime.h and libcudart are all present)
#
# Target version for apt install (prefer torch's CUDA, e.g. 13.0 / cu130):
#   1. nvcc --version from an already-resolved toolkit
#   2. else torch.version.cuda in the venv
#   3. else nvidia-smi "CUDA Version" as a hint (driver max supported, NOT toolkit)
_cuda_home_ok() { [[ -n "${1:-}" && -x "$1/bin/nvcc" ]]; }

# Normalize "13.0" / "13.0.88" / "cuda_13.0" -> "13.0" (major.minor).
_cuda_xy() {
    local v="${1:-}"
    v="${v#cuda_}"
    v="${v#CUDA }"
    [[ "$v" =~ ^([0-9]+)\.([0-9]+) ]] && { echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"; return 0; }
    [[ "$v" =~ ^([0-9]+)$ ]] && { echo "${BASH_REMATCH[1]}.0"; return 0; }
    return 1
}

detect_target_cuda_version() {
    local v=""
    # 1) Existing toolkit / nvcc
    if _cuda_home_ok "${CUDA_HOME:-}"; then
        v="$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"
    elif [[ -n "${FLASHINFER_NVCC:-}" && -x "${FLASHINFER_NVCC}" ]]; then
        v="$("$FLASHINFER_NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"
    elif command -v nvcc >/dev/null 2>&1; then
        v="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1 || true)"
    fi
    v="$(_cuda_xy "$v" 2>/dev/null || true)"
    if [[ -n "$v" ]]; then
        echo "$v"
        return 0
    fi
    # 2) Prefer torch's CUDA (build must match the venv's torch)
    v="$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    v="$(_cuda_xy "$v" 2>/dev/null || true)"
    if [[ -n "$v" ]]; then
        echo "$v"
        return 0
    fi
    # 3) nvidia-smi hint only (driver max; may be newer than any installed toolkit)
    if command -v nvidia-smi >/dev/null 2>&1; then
        v="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1 || true)"
        v="$(_cuda_xy "$v" 2>/dev/null || true)"
        if [[ -n "$v" ]]; then
            echo "$v"
            return 0
        fi
    fi
    return 1
}

# True if apt-get -s output would touch kernel/DKMS/nvidia-driver packages.
# cuda-driver-dev-* is userspace stubs and is allowed.
_apt_cuda_pulls_forbidden() {
    local dry="$1" pkg
    while read -r pkg; do
        [[ -z "$pkg" ]] && continue
        case "$pkg" in
            linux-image|linux-image-*|linux-headers|linux-headers-*|dkms|*-dkms|\
            nvidia-driver|nvidia-driver-*|nvidia-dkms|nvidia-dkms-*|\
            nvidia-kernel|nvidia-kernel-*|nvidia-kernel-common|nvidia-kernel-common-*)
                err "Refusing CUDA toolkit apt install: dry-run would touch '$pkg' (kernel/DKMS/driver)."
                return 0
                ;;
        esac
    done < <(printf '%s\n' "$dry" | awk '/^(Inst|Conf|Remv) / { print $2 }')
    return 1
}

# ADDON: when nvcc is missing, install a matching userspace CUDA toolkit via apt.
# Prefer the minimal set (compiler + libraries-dev + nvml-dev); fall back to
# cuda-toolkit-X-Y. Always dry-run first and refuse kernel/DKMS/driver pulls.
# SKIP_CUDA_TOOLKIT_INSTALL=1 disables this. FORCE_SYSTEM_DEPS remains for the
# separate libssl/openmpi/numa apt path only.
install_cuda_toolkit_apt() {
    if [[ "${SKIP_CUDA_TOOLKIT_INSTALL:-0}" == "1" ]]; then
        skip "SKIP_CUDA_TOOLKIT_INSTALL=1; not attempting CUDA toolkit apt install"
        return 1
    fi
    if ! command -v apt-get >/dev/null 2>&1; then
        err "apt-get not found; cannot auto-install CUDA toolkit."
        return 1
    fi

    local target="" ver_dash=""
    target="$(detect_target_cuda_version || true)"
    if [[ -z "$target" ]]; then
        err "Could not detect a target CUDA version (no nvcc, torch.version.cuda, or nvidia-smi hint)."
        return 1
    fi
    ver_dash="${target/./-}"
    log "Target CUDA toolkit version: $target (prefer matching torch; nvidia-smi is driver-max hint only)"

    local SUDO=""
    if [[ "$(id -u)" -ne 0 ]]; then
        # Non-interactive only: never hang on a sudo password prompt.
        if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            SUDO="sudo"
        else
            err "CUDA toolkit install needs root (sudo -n unavailable). Install manually, then re-run ./scripts/install.sh:"
            err "  sudo apt-get update"
            err "  sudo apt-get install -y --no-install-recommends cuda-compiler-${ver_dash} cuda-libraries-dev-${ver_dash} cuda-nvml-dev-${ver_dash}"
            err "  # or: sudo apt-get install -y --no-install-recommends cuda-toolkit-${ver_dash}"
            err "  # Dry-run first: sudo apt-get -s install --no-install-recommends ...  (must NOT pull linux-image/DKMS/nvidia-driver)"
            return 1
        fi
    fi

    # Refresh package lists so cuda-*-X-Y from the NVIDIA CUDA repo is visible.
    log "Refreshing apt package lists (for CUDA toolkit ${target})"
    if ! $SUDO apt-get update; then
        err "apt-get update failed; cannot install CUDA toolkit ${target}."
        return 1
    fi

    local -a minimal=( "cuda-compiler-${ver_dash}" "cuda-libraries-dev-${ver_dash}" "cuda-nvml-dev-${ver_dash}" )
    local -a full=( "cuda-toolkit-${ver_dash}" )
    local -a try_sets=("minimal" "full")
    local set_name pkg_list dry missing p

    for set_name in "${try_sets[@]}"; do
        if [[ "$set_name" == "minimal" ]]; then
            pkg_list=( "${minimal[@]}" )
        else
            pkg_list=( "${full[@]}" )
        fi

        missing=0
        for p in "${pkg_list[@]}"; do
            if ! apt-cache show "$p" >/dev/null 2>&1; then
                err "CUDA apt package '$p' not found in apt cache (need NVIDIA CUDA apt repo for ${target})."
                missing=1
                break
            fi
        done
        [[ "$missing" -eq 1 ]] && continue

        log "Dry-running apt install of ${pkg_list[*]} (refuse kernel/DKMS/driver pulls)"
        dry="$($SUDO apt-get -s install --no-install-recommends "${pkg_list[@]}" 2>&1 || true)"
        if _apt_cuda_pulls_forbidden "$dry"; then
            err "Skipping ${set_name} set for CUDA ${target} due to forbidden packages in dry-run."
            continue
        fi
        if ! printf '%s\n' "$dry" | grep -qE '^(Inst|Conf) '; then
            # Already installed or nothing to do -- treat as success if nvcc appears later.
            log "apt dry-run for ${pkg_list[*]} reported no packages to install"
        fi

        log "Installing CUDA userspace toolkit ${target} via apt (${set_name}: ${pkg_list[*]})"
        if $SUDO apt-get install -y --no-install-recommends "${pkg_list[@]}"; then
            log "CUDA toolkit apt install (${set_name}) finished"
            return 0
        fi
        err "apt-get install failed for ${pkg_list[*]}; trying next candidate set if any."
    done

    err "Could not install a safe CUDA ${target} userspace toolkit via apt."
    return 1
}

resolve_cuda_home() {
    local cand torch_cuda="" torch_major="" nvcc_path
    torch_cuda="$("$PY" -c 'import torch; print(torch.version.cuda or "")' 2>/dev/null || true)"
    torch_major="${torch_cuda%%.*}"

    if _cuda_home_ok "${CUDA_HOME:-}"; then
        :
    elif [[ -n "${FLASHINFER_NVCC:-}" && -x "${FLASHINFER_NVCC}" ]]; then
        CUDA_HOME="$(cd -- "$(dirname -- "$FLASHINFER_NVCC")/.." && pwd)"
    elif _cuda_home_ok /usr/local/cuda; then
        CUDA_HOME=/usr/local/cuda
    elif nvcc_path="$(command -v nvcc 2>/dev/null)" && [[ -n "$nvcc_path" ]]; then
        nvcc_path="$(readlink -f "$nvcc_path" 2>/dev/null || echo "$nvcc_path")"
        CUDA_HOME="$(cd -- "$(dirname -- "$nvcc_path")/.." && pwd)"
    else
        CUDA_HOME=""
        local -a cands=()
        [[ -n "$torch_major" ]] && cands+=( $(ls -d /usr/local/cuda-"$torch_major".* 2>/dev/null | sort -rV || true) )
        cands+=( $(ls -d /usr/local/cuda-* 2>/dev/null | sort -rV || true) /opt/cuda )
        for cand in "${cands[@]}"; do
            if _cuda_home_ok "$cand"; then CUDA_HOME="$cand"; break; fi
        done
        if [[ -z "$CUDA_HOME" ]]; then
            local sp
            sp="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null || true)"
            for cand in $(ls -d "$sp"/nvidia/cu"${torch_major}" "$sp"/nvidia/cu* 2>/dev/null); do
                if _cuda_home_ok "$cand" && [[ -x "$cand/nvvm/bin/cicc" && -f "$cand/include/cuda_runtime.h" ]] \
                    && ls "$cand"/lib/libcudart.so* >/dev/null 2>&1; then
                    CUDA_HOME="$cand"
                    err "Using pip-wheel CUDA toolkit at $CUDA_HOME (experimental; a full system toolkit is preferred)."
                    break
                fi
            done
        fi
    fi

    if _cuda_home_ok "${CUDA_HOME:-}"; then
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        log "CUDA toolkit: CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')); torch CUDA ${torch_cuda:-unknown}"
        return 0
    fi
    unset CUDA_HOME
    err "No CUDA toolkit (nvcc) found. Checked: \$CUDA_HOME, \$FLASHINFER_NVCC, /usr/local/cuda, PATH, /usr/local/cuda-*, /opt/cuda, venv nvidia/cu*/bin."
    err "torch in this venv was built for CUDA ${torch_cuda:-unknown}. The kernel build needs a matching toolkit."
    err "  or point at an existing toolkit: CUDA_HOME=/path/to/cuda ./scripts/install.sh"
    return 1
}

# Resolve existing nvcc first; if missing, optionally apt-install a matching
# userspace toolkit (dry-run guarded), then re-resolve CUDA_HOME.
ensure_cuda_home() {
    if resolve_cuda_home; then
        return 0
    fi
    log "nvcc not found; attempting guarded CUDA toolkit apt install (userspace only, no kernel/DKMS/driver)"
    if install_cuda_toolkit_apt; then
        # Clear a stale empty CUDA_HOME so re-resolve can pick /usr/local/cuda-X.Y.
        unset CUDA_HOME
        if resolve_cuda_home; then
            return 0
        fi
        err "CUDA toolkit apt install reported success, but nvcc still not found under /usr/local/cuda*."
        return 1
    fi
    local hint=""
    hint="$(detect_target_cuda_version 2>/dev/null || true)"
    if [[ -n "$hint" ]]; then
        err "  Manual install (after dry-run confirms no linux-image/DKMS/nvidia-driver):"
        err "    sudo apt-get install -y --no-install-recommends cuda-compiler-${hint/./-} cuda-libraries-dev-${hint/./-} cuda-nvml-dev-${hint/./-}"
        err "    # or: sudo apt-get install -y --no-install-recommends cuda-toolkit-${hint/./-}"
    fi
    return 1
}

# --- smg gateway build (Rust) ------------------------------------------------
# Build and install the smg OpenAI gateway from the local smg/ source so it
# carries the best_of+stream validator fix (smg/crates/protocols), plus the
# matching smg-grpc-proto and smg-grpc-servicer. Recipe follows
# smg/scripts/installation/install-smg.sh and ci_install_tokenspeed.sh.
build_and_install_smg() {
    # protoc: the gateway's build compiles .proto definitions. Install the
    # OFFICIAL protoc release zip, NOT apt's protobuf-compiler. The apt package
    # ships the binary but omits the well-known-type includes
    # (google/protobuf/timestamp.proto, struct.proto) the smg protos import, so
    # codegen fails with "File not found". The zip lays down both the binary
    # AND /usr/local/include/google/protobuf/*.proto. This mirrors
    # smg/scripts/installation/install-smg.sh.
    local SUDO=""
    [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    local PROTOC_BIN=""
    if [[ -x /usr/local/bin/protoc && -f /usr/local/include/google/protobuf/timestamp.proto ]]; then
        PROTOC_BIN=/usr/local/bin/protoc
    else
        log "Installing official protoc (with well-known-type includes) for the smg build"
        local protoc_url="https://github.com/protocolbuffers/protobuf/releases/download/v32.0/protoc-32.0-linux-x86_64.zip"
        if command -v apt-get >/dev/null 2>&1; then
            $SUDO apt-get install -y --no-install-recommends unzip curl >/dev/null 2>&1 || true
        fi
        if curl --proto '=https' --tlsv1.2 -fsSL -o /tmp/protoc.zip "$protoc_url" \
            && $SUDO unzip -o /tmp/protoc.zip -d /usr/local >/dev/null; then
            rm -f /tmp/protoc.zip
            PROTOC_BIN=/usr/local/bin/protoc
        fi
    fi
    if [[ -z "$PROTOC_BIN" ]] || ! "$PROTOC_BIN" --version >/dev/null 2>&1; then
        err "protoc (official zip) could not be installed; the smg gateway build needs it with its well-known-type includes. Install it and re-run with --force-smg."
        return 1
    fi
    export PROTOC="$PROTOC_BIN"

    # Rust toolchain (rustup): the gateway is a Rust crate.
    if ! command -v cargo >/dev/null 2>&1; then
        [[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        log "Installing Rust toolchain via rustup (needed to build the smg gateway)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v cargo >/dev/null 2>&1; then
        err "cargo still not on PATH after rustup install; cannot build the smg gateway."
        return 1
    fi

    # maturin: the gateway's PEP-517 build backend.
    "$PY" -m pip install "maturin>=1.0,<2.0"

    # Remove the published wheels that claim the same import paths (smg /
    # smg_grpc_proto / smg_grpc_servicer); otherwise they shadow the local
    # source builds and the fix never takes effect. Ignore if absent.
    "$PY" -m pip uninstall -y \
        tokenspeed-smg tokenspeed-smg-grpc-proto tokenspeed-smg-grpc-servicer \
        smg smg-grpc-proto smg-grpc-servicer >/dev/null 2>&1 || true

    log "Building smg gateway wheel from source (maturin, release, vendored-openssl)"
    # Clean dist/ first: `maturin build --out dist` APPENDS, and a stale wheel
    # from a prior build (or a different manylinux tag) left behind makes the
    # `dist/*.whl` glob match two files, which pip then refuses to install
    # ("conflicting dependencies"). A fresh dist/ guarantees exactly one wheel.
    rm -rf smg/bindings/python/dist
    ( cd smg/bindings/python && \
      ulimit -n 65536 2>/dev/null || true; \
      maturin build --release --features vendored-openssl --out dist )
    log "Installing smg gateway wheel"
    "$PY" -m pip install --force-reinstall smg/bindings/python/dist/*.whl

    log "Installing smg-grpc-proto + smg-grpc-servicer from source (editable)"
    "$PY" -m pip install -e smg/crates/grpc_client/python
    "$PY" -m pip install -e smg/grpc_servicer
}

# --- vLLM FlashAttention-2 build (sm_120 prefill/extend path) -----------------
# Optional FA2 MHA prefill/extend path: registers only if
# `import vllm_flash_attn` succeeds; without it the engine falls back to the
# Triton prefill kernel (~1.39x TTFT loss). No usable PyPI wheel exists for
# this torch/cu combo, so it is compiled
# from source against the venv's own torch -- guaranteeing the ABI matches --
# for the SAME CUDA_ARCH_LIST the rest of the stack uses. Mirrors the
# Dockerfile's vllm-flash-attention step.
VLLM_FLASH_ATTN_REPO="${VLLM_FLASH_ATTN_REPO:-https://github.com/vllm-project/flash-attention}"
VLLM_FLASH_ATTN_REF="${VLLM_FLASH_ATTN_REF:-main}"
build_and_install_vllm_flash_attn() {
    if ! command -v git >/dev/null 2>&1; then
        err "git not found; cannot build vllm_flash_attn from source. Install git and re-run, or the sm_120 FA2 prefill path stays on the Triton fallback."
        return 1
    fi
    local src
    src="$(mktemp -d)"
    log "Cloning vLLM flash-attention ($VLLM_FLASH_ATTN_REF) with submodules"
    if ! git clone --depth 1 --branch "$VLLM_FLASH_ATTN_REF" --recursive \
            "$VLLM_FLASH_ATTN_REPO" "$src"; then
        err "clone of $VLLM_FLASH_ATTN_REPO failed; skipping FA2 (selection falls back to Triton)."
        rm -rf "$src"
        return 1
    fi
    log "Building vllm_flash_attn from source (arches: $CUDA_ARCH_LIST) -- this compiles CUDA and is slow"
    # TORCH_CUDA_ARCH_LIST is the arch list the torch/CMake extension build
    # reads; use the same value as the rest of the image so FA2 covers every
    # tier, not just sm_120.
    #
    # --no-deps is REQUIRED, not tidiness. This fork's wheel declares
    # `torch==2.4.0` as a runtime dependency; without --no-deps pip honours it
    # and DOWNGRADES the venv's torch 2.13.0/cu13 to 2.4.0/cu12, pulling a whole
    # cu12 stack. That silently breaks every already-compiled cu13 extension --
    # the trtllm kernel then dies at import with "libcublas.so.13: cannot open
    # shared object file" and the server never starts. FA2 only needs torch at
    # BUILD time, which --no-build-isolation already supplies from the installed
    # torch; it must not be allowed to change the installed torch.
    if ! TORCH_CUDA_ARCH_LIST="$CUDA_ARCH_LIST" CUDA_ARCH_LIST="$CUDA_ARCH_LIST" \
            "$PY" -m pip install "$src" --no-build-isolation --no-deps; then
        rm -rf "$src"
        err "vllm_flash_attn build failed; the sm_120 FA2 prefill path will fall back to Triton. Re-run with REINSTALL_FA2=1 after fixing the toolchain."
        return 1
    fi
    rm -rf "$src"

    # COLLISION GUARD. The vLLM fork installs a top-level `flash_attn` package.
    # If it shadows the base image's FA4 `flash_attn.cute` -- which
    # tokenspeed_kernel imports at ops/attention/flash_attn/__init__.py -- the
    # whole engine dies at import ("No module named 'flash_attn_2_cuda'"). FA2
    # is optional; the kernel is not. So verify BOTH import, and if FA2 broke
    # `flash_attn.cute`, back FA2 out again and keep the working kernel path.
    if ! "$PY" -c "import flash_attn.cute" >/dev/null 2>&1; then
        err "vllm_flash_attn shadowed the FA4 'flash_attn.cute' namespace the kernel needs; uninstalling it to keep the engine importable (prefill falls back to Triton). The fork's package must import as 'vllm_flash_attn', not 'flash_attn'."
        "$PY" -m pip uninstall -y vllm-flash-attn vllm_flash_attn >/dev/null 2>&1 || true
        # DESTRUCTIVE-UNINSTALL REPAIR. The fork installs its package under the
        # import name `flash_attn`, the SAME top-level directory `tokenspeed-fa4`
        # owns (it ships `flash_attn/cute/`). pip merges the two on install, so
        # the uninstall above deletes the shared `flash_attn/` directory -- and
        # with it tokenspeed-fa4's `flash_attn/cute/` files -- while pip still
        # records tokenspeed-fa4 as installed. Left as-is the engine dies at
        # import with "No module named 'flash_attn'" (tokenspeed_kernel imports
        # flash_attn.cute). Force-reinstall tokenspeed-fa4 to lay its files back
        # down. Pin to the version pip currently records so this restores the
        # exact build the kernel resolved; fall back to unpinned if unknown.
        local fa4_ver
        fa4_ver="$("$PY" -m pip show tokenspeed-fa4 2>/dev/null \
            | awk -F': ' '/^Version:/{print $2}')"
        if [[ -n "$fa4_ver" ]]; then
            "$PY" -m pip install --no-deps --force-reinstall \
                "tokenspeed-fa4==$fa4_ver" >/dev/null 2>&1 || true
        else
            "$PY" -m pip install --no-deps --force-reinstall \
                tokenspeed-fa4 >/dev/null 2>&1 || true
        fi
        if "$PY" -c "import flash_attn.cute" >/dev/null 2>&1; then
            log "Restored the FA4 'flash_attn.cute' provider (tokenspeed-fa4) after backing out the FA2 fork; engine import path is intact."
        else
            err "FA4 'flash_attn.cute' is STILL missing after reinstalling tokenspeed-fa4; the engine will not import. Recover with: '$PY' -m pip install --no-deps --force-reinstall tokenspeed-fa4"
        fi
        return 1
    fi
    if ! "$PY" -c "import vllm_flash_attn" >/dev/null 2>&1; then
        err "vllm_flash_attn installed but not importable under that name; FA2 selection will fall back to Triton. Left in place; it did not break the kernel."
        return 1
    fi
    log "vllm_flash_attn installed and both it and flash_attn.cute import cleanly"
}

# --- Decide how much work to do ----------------------------------------------
# If nothing is installed yet, this is a first install: do the full from-source
# build. If everything is already present and no force flag is set, there is
# nothing to compile -- the editable engine/scheduler pick up synced edits on
# their own, so the script becomes a no-op and returns fast.
FRESH=0
if [[ "$REINSTALL" == "1" ]]; then
    FRESH=1
    log "REINSTALL=1: forcing a full from-source rebuild of every package."
elif ! have_dist tokenspeed || ! have_dist tokenspeed-kernel \
        || ! have_dist tokenspeed-scheduler; then
    FRESH=1
    log "One or more packages are missing; performing a full from-source install."
else
    skip "All packages already installed. Nothing to build."
    skip "Editable engine/scheduler pick up synced Python edits with no reinstall."
    skip "Force a rebuild with:  REINSTALL=1 ./scripts/install.sh   (kernel only: REINSTALL_KERNEL=1)"
fi

# The kernel compile is the expensive step. Build it only on a fresh/forced
# full install, or when REINSTALL_KERNEL is set (a C++/CUDA change). Otherwise
# leave the already-compiled kernel in place.
KERNEL_WORK=0
if [[ "$FRESH" == "1" || "$REINSTALL_KERNEL" == "1" ]]; then
    KERNEL_WORK=1
fi

# The smg gateway is a Rust build (also slow). Build it from local source when:
# a full/forced install is happening, OR REINSTALL_SMG is set (after editing
# the gateway, e.g. the best_of fix), OR the local `smg` gateway dist is not
# installed yet. Only meaningful when the smg/ source tree is present.
SMG_WORK=0
if [[ "$HAVE_SMG_SRC" == "1" ]]; then
    if [[ "$FRESH" == "1" || "${REINSTALL_SMG:-0}" == "1" ]] || ! have_dist smg; then
        SMG_WORK=1
    fi
fi

# vllm_flash_attn (sm_120 FA2): OPT-IN ONLY. This is an optional prefill
# speedup, not required to serve -- prefill falls back to the Triton
# prefill kernel when it is absent. It is deliberately NOT built on a plain or
# even a full (--force) install, because the vLLM fork's package collides with
# and can shadow the base image's FA4 `flash_attn.cute` namespace that
# tokenspeed_kernel imports, taking the whole engine down. Build it only when
# explicitly asked (BUILD_FA2=1 or --with-fa2 / REINSTALL_FA2=1 / --force-fa2),
# and the guard after the build removes it again if it broke `flash_attn.cute`.
FA2_WORK=0
if [[ "$BUILD_FA2" == "1" || "$REINSTALL_FA2" == "1" ]]; then
    FA2_WORK=1
fi

if [[ "$FRESH" == "1" || "$KERNEL_WORK" == "1" || "$SMG_WORK" == "1" || "$FA2_WORK" == "1" ]]; then
    # System build deps and the pip build front-ends are only needed when we
    # actually compile. Skipping them on a no-op run keeps it fast and offline.
    install_system_deps
    # ADDON: resolve CUDA_HOME/nvcc (and optionally apt-install userspace toolkit)
    # before any source build. Fatal only when the CUDA kernel is going to be
    # compiled; smg/FA2-only runs just warn.
    if ! ensure_cuda_home; then
        if [[ "$FRESH" == "1" || "$KERNEL_WORK" == "1" ]]; then
            exit 1
        fi
    fi
    # setuptools + wheel are REQUIRED here because the kernel and engine are
    # built with --no-build-isolation: pip runs the build backend
    # (setuptools.build_meta) from THIS venv. A fresh `python -m venv` ships pip
    # but not setuptools/wheel on modern Pythons, so without this the source
    # build dies with: BackendUnavailable: Cannot import 'setuptools.build_meta'.
    "$PY" -m pip install --upgrade pip
    log "Installing build front-ends (setuptools, wheel, cmake, ninja)"
    "$PY" -m pip install setuptools wheel cmake ninja
fi

# --- Build/install, in the Dockerfile's exact order --------------------------
if [[ "$FRESH" == "1" ]]; then
    log "[1/5] Building tokenspeed-kernel from source, non-editable (arches: $CUDA_ARCH_LIST)"
    "$PY" -m pip install tokenspeed-kernel/python/ --no-build-isolation

    log "[2/5] Installing tokenspeed-scheduler (editable)"
    "$PY" -m pip install -e tokenspeed-scheduler/

    log "[3/5] Installing the engine (python/, editable) -- pulls third-party deps"
    "$PY" -m pip install -e ./python --no-build-isolation

    log "[4/5] Reinstalling tokenspeed-kernel from source (non-editable, force, no-deps) so it wins"
    "$PY" -m pip install tokenspeed-kernel/python/ \
        --no-build-isolation --no-deps --force-reinstall

    log "[5/5] Reinstalling tokenspeed-scheduler from source (editable, force, no-deps)"
    "$PY" -m pip install -e tokenspeed-scheduler/ --no-deps --force-reinstall

    # torchao: required to load torchao-quantized checkpoints (e.g. the
    # gemma-3-27b-it-FP8 build whose config.json declares
    # quant_method="torchao"). The engine's TorchAOConfig delegates the actual
    # dequant/compute to this library, so it must be importable at runtime.
    # Pure Python, so a synced engine edit still needs no reinstall.
    log "Installing torchao (runtime dep for torchao-quantized checkpoints)"
    "$PY" -m pip install "torchao>=0.10.0"
    # NOTE: vllm_flash_attn (FA2) is intentionally NOT built here. It is an
    # optional prefill speedup and its package can shadow the kernel's FA4
    # namespace, so it is opt-in via --with-fa2 (handled by the FA2 block below).
elif [[ "$KERNEL_WORK" == "1" ]]; then
    # Kernel-only rebuild: recompile and reinstall just the kernel, leaving
    # the editable engine/scheduler and every third-party dep untouched.
    log "REINSTALL_KERNEL=1: recompiling tokenspeed-kernel from source only"
    "$PY" -m pip install tokenspeed-kernel/python/ \
        --no-build-isolation --no-deps --force-reinstall
fi

# --- smg gateway (Rust) from local source ------------------------------------
# Independent of the kernel branches: rebuild the OpenAI gateway (+ proto +
# servicer) from smg/ so the best_of+stream fix is in effect. Skipped when the
# smg/ source is absent or the gateway is already built and not forced.
if [[ "$SMG_WORK" == "1" ]]; then
    log "Building + installing smg gateway/proto/servicer from local source (smg/)"
    build_and_install_smg
elif [[ "$HAVE_SMG_SRC" == "1" ]]; then
    skip "smg gateway already installed. Rebuild after editing smg/ with: REINSTALL_SMG=1 ./scripts/install.sh"
fi

# --- vllm_flash_attn (sm_120 FA2), OPT-IN, independent of kernel/smg branches -
# Only runs when explicitly requested (--with-fa2 / BUILD_FA2=1 / --force-fa2 /
# REINSTALL_FA2=1). The build function backs FA2 out automatically if it shadows
# the kernel's flash_attn.cute, so a failed FA2 never leaves the engine broken.
if [[ "$FA2_WORK" == "1" ]]; then
    log "Building vllm_flash_attn from source (sm_120 FA2 prefill/extend path; opt-in)"
    build_and_install_vllm_flash_attn || \
        err "continuing without vllm_flash_attn; FA2 prefill falls back to Triton (engine unaffected)."
else
    skip "Skipping FA2 (vllm_flash_attn). It is optional; prefill uses the Triton fallback. Build it with: ./scripts/install.sh --with-fa2"
fi

# --- Verify WITHOUT importing the kernel (mirrors the Dockerfile check) -------
# Importing tokenspeed_kernel runs GPU detection; we avoid it so the check is
# identical whether or not a GPU is visible. We assert the installed dist is the
# source/dev build (not the 0.1.3 wheel) and that it exports the symbol the wheel
# lacked -- both via metadata/file inspection only.
#
# Editable-install note: `md.files()` may return None (or omit source paths) for
# an `-e` install, so fall back to locating the package via its top-level import
# path (metadata origin / __init__) without importing it.
if [[ "$FRESH" != "1" && "$KERNEL_WORK" != "1" && "$SMG_WORK" != "1" && "$FA2_WORK" != "1" ]]; then
    log "Done. No build was needed. Activate with:  source \"$VENV_DIR/bin/activate\""
    exit 0
fi

# The kernel verification below only makes sense when a kernel build happened;
# an smg-only or FA2-only run has nothing to verify there.
if [[ "$FRESH" != "1" && "$KERNEL_WORK" != "1" ]]; then
    log "Done (gateway/FA2 rebuilt). Activate with:  source \"$VENV_DIR/bin/activate\""
    exit 0
fi

log "Verifying the source kernel won (metadata inspection, no import)"
"$PY" - <<'PY'
import importlib.metadata as md
import importlib.util
import pathlib
import sys

ver = md.version("tokenspeed-kernel")
print("installed tokenspeed-kernel:", ver)
if ".dev" not in ver and "+git" not in ver:
    sys.exit(f"ERROR: installed a release wheel ({ver}), not the source build; "
             "a dependency reinstall shadowed the local kernel.")


def _find_comm_init() -> pathlib.Path | None:
    rel = "ops/communication/__init__.py"
    # 1) Recorded files (normal wheel install).
    try:
        files = md.files("tokenspeed-kernel") or []
    except Exception:
        files = []
    for f in files:
        if f.as_posix().endswith(rel):
            return pathlib.Path(f.locate())
    # 2) Package location via import machinery (no import executed).
    try:
        spec = importlib.util.find_spec("tokenspeed_kernel")
    except Exception:
        spec = None
    if spec and spec.submodule_search_locations:
        for root in spec.submodule_search_locations:
            cand = pathlib.Path(root) / "ops" / "communication" / "__init__.py"
            if cand.is_file():
                return cand
    # 3) Fall back to the source checkout this script installs from.
    cand = pathlib.Path("tokenspeed-kernel/python/tokenspeed_kernel") / rel
    if cand.is_file():
        return cand.resolve()
    return None


init = _find_comm_init()
if init is None:
    # The version check above already proved the source build won; this file
    # probe is a belt-and-braces symbol check, so a location miss is a warning
    # rather than a hard failure that would fail an otherwise-good install.
    print("WARNING: could not locate ops/communication/__init__.py to verify "
          "the allgather_dual_rmsnorm symbol; version check passed, continuing.")
elif "allgather_dual_rmsnorm" not in init.read_text():
    sys.exit("ERROR: installed kernel lacks allgather_dual_rmsnorm.")
else:
    print("kernel source + symbol OK:", init)
PY

log "Done. Activate the environment with:  source \"$VENV_DIR/bin/activate\""

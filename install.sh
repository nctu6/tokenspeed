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
#   REINSTALL=1 ./install.sh      (or: ./install.sh --force)
# Force just the kernel to recompile (after a C++/CUDA change):
#   REINSTALL_KERNEL=1 ./install.sh
# Force just the smg gateway to rebuild (after editing smg/, Rust):
#   REINSTALL_SMG=1 ./install.sh  (or: ./install.sh --force-smg)
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
#   ./install.sh                 # adopt active $VIRTUAL_ENV, else create/use ./.venv
#   source .venv_docker/bin/activate && ./install.sh   # install into .venv_docker
#   REINSTALL=1 ./install.sh     # force full from-source rebuild of all packages
#   REINSTALL_KERNEL=1 ./install.sh  # force only the kernel to recompile
#   VENV_DIR=/path/to/venv ./install.sh   # force a specific venv (overrides active)
#   MAX_JOBS=8 ./install.sh
#   CUDA_ARCH_LIST="12.0a" ./install.sh   # build only for sm_120 (faster)
#
# Run this from the repository root (the directory that contains python/,
# tokenspeed-kernel/, tokenspeed-scheduler/).
set -euo pipefail

# --- Resolve repo root (this script lives at the repo root) ------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

# --- Tunables (mirror the Dockerfile ARG/ENV values) -------------------------
MAX_JOBS="${MAX_JOBS:-16}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-9.0a 10.0a 12.0a}"
# VENV_DIR intentionally has NO default here. We resolve it below with this
# precedence: an explicit VENV_DIR wins; otherwise an already-active venv
# ($VIRTUAL_ENV, e.g. a .venv_docker the user sourced inside a container) is
# adopted as-is; otherwise we fall back to $SCRIPT_DIR/.venv. Defaulting it here
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
#   3. Neither -- fall back to $SCRIPT_DIR/.venv (the original default),
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
    VENV_DIR="$SCRIPT_DIR/.venv"
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
    log "Installing system dependencies (libssl-dev libopenmpi-dev libnuma1 pkg-config)"
    $SUDO apt-get update
    $SUDO apt-get install -y --no-install-recommends \
        libssl-dev \
        libopenmpi-dev \
        libnuma1 \
        pkg-config
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
    skip "Force a rebuild with:  REINSTALL=1 ./install.sh   (kernel only: REINSTALL_KERNEL=1)"
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
    skip "smg gateway already installed. Rebuild after editing smg/ with: REINSTALL_SMG=1 ./install.sh"
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
    skip "Skipping FA2 (vllm_flash_attn). It is optional; prefill uses the Triton fallback. Build it with: ./install.sh --with-fa2"
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

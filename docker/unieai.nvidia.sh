#!/usr/bin/env bash
# Build the UnieAI NVIDIA image (Dockerfile.nvidia.unieai) from the repo root.
#
# Tag convention:
#   nctu6/tokenspeed:<short-commit>-<YYYYMMDD>
# Example:
#   nctu6/tokenspeed:d9e6238b-20261007
#
# DATE is the build day in BUILD_TZ (default Asia/Taipei), independent of the
# build host clock zone. Override with:  BUILD_DATE=20261007 ./docker/unieai.nvidia.sh
#                                 or:  BUILD_TZ=UTC ./docker/unieai.nvidia.sh
# Pass-through docker build args:
#   MAX_JOBS=32 ./docker/unieai.nvidia.sh
#   CUDA_ARCH_LIST="9.0a 12.0a" ./docker/unieai.nvidia.sh
#   WITH_FA2=1 ./docker/unieai.nvidia.sh   # opt-in vLLM FA2 (default off)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# The Dockerfile builds smg/ and DeepGEMM/ from source. Populate the submodules
# (DeepGEMM recursively, for its cutlass + deep_jit) so a fresh clone does not
# silently fall back to the published wheels. This is automatic: the user runs
# the script with no flags. Only empty checkouts are initialized, so local
# submodule edits are kept.
#
# The UnieAI NVIDIA image targets SM120 (it compiles 12.0a) alongside Hopper
# and datacenter Blackwell, so the DeepGEMM source build (its kernels JIT at
# runtime and cover arch 9/10/12) is always the right choice here -- no
# hardware detection is needed at build time.
if [[ ! -f smg/bindings/python/pyproject.toml ]]; then
  git submodule update --init --recursive smg
fi
if [[ ! -f DeepGEMM/setup.py || ! -d DeepGEMM/third-party/cutlass/include \
      || ! -d DeepGEMM/third-party/deep_jit/include ]]; then
  git submodule update --init --recursive DeepGEMM || true
fi
if [[ ! -f DeepGEMM/setup.py || ! -d DeepGEMM/third-party/cutlass/include \
      || ! -d DeepGEMM/third-party/deep_jit/include ]]; then
  # Could not populate DeepGEMM (e.g. offline, or not a git checkout). The
  # image build then keeps the published tokenspeed-deepgemm wheel (no SM120
  # kernels) rather than failing the whole build.
  echo "WARNING: DeepGEMM/ is not populated; the image will keep the published tokenspeed-deepgemm wheel (no SM120 kernels)." >&2
  DEEPGEMM_COMMIT="(not built; wheel kept)"
else
  DEEPGEMM_COMMIT="$(git -C DeepGEMM rev-parse --short=8 HEAD)"
fi

COMMIT="$(git rev-parse --short=8 HEAD)"
BUILD_TZ="${BUILD_TZ:-Asia/Taipei}"
DATE="${BUILD_DATE:-$(TZ="${BUILD_TZ}" date +%Y%m%d)}"
IMAGE="nctu6/tokenspeed:${COMMIT}-${DATE}"

BUILD_ARGS=()
if [[ -n "${MAX_JOBS:-}" ]]; then
  BUILD_ARGS+=(--build-arg "MAX_JOBS=${MAX_JOBS}")
fi
if [[ -n "${CUDA_ARCH_LIST:-}" ]]; then
  BUILD_ARGS+=(--build-arg "CUDA_ARCH_LIST=${CUDA_ARCH_LIST}")
fi
if [[ -n "${WITH_FA2:-}" ]]; then
  BUILD_ARGS+=(--build-arg "WITH_FA2=${WITH_FA2}")
fi
if [[ -n "${VLLM_FLASH_ATTN_REPO:-}" ]]; then
  BUILD_ARGS+=(--build-arg "VLLM_FLASH_ATTN_REPO=${VLLM_FLASH_ATTN_REPO}")
fi
if [[ -n "${VLLM_FLASH_ATTN_REF:-}" ]]; then
  BUILD_ARGS+=(--build-arg "VLLM_FLASH_ATTN_REF=${VLLM_FLASH_ATTN_REF}")
fi

echo "Building ${IMAGE}"
echo "  Dockerfile: docker/Dockerfile.nvidia.unieai"
echo "  context:    ${REPO_ROOT}"
echo "  commit:     ${COMMIT}"
echo "  DeepGEMM:   ${DEEPGEMM_COMMIT}"
echo "  date:       ${DATE} (${BUILD_TZ})"

docker build \
  -f docker/Dockerfile.nvidia.unieai \
  -t "${IMAGE}" \
  "${BUILD_ARGS[@]}" \
  .

echo "IMAGE=${IMAGE}"
docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}' \
  | awk -v img="${IMAGE}" '$1 == img { print; found=1 } END { if (!found) exit 1 }'

#!/bin/bash
# Build a torch overlay whose bundled ROCm user-space libraries come from another
# ROCm release, so one Python environment can A/B two runtimes with no reinstall.
#
# Usage: rocm_overlay.sh <rocm_root> <overlay_dir> [<triton_pkg_dir>]
#   rocm_root      a TheRock-style tree with lib/ (and lib/llvm/lib, lib/rocm_sysdeps/lib)
#                  and .kpack/ next to lib/, e.g. /opt/rocm/core-7.14 or a copy of an
#                  image's _rocm_sdk_devel
#   overlay_dir    output; put <overlay_dir>/site first on PYTHONPATH
#   triton_pkg_dir an upstream triton package built for the same LLVM as rocm_root's
#                  comgr (required when rocm_root's comgr links a shared libLLVM)
# Then run with:
#   PYTHONPATH=<overlay_dir>/site:$PYTHONPATH \
#   LD_LIBRARY_PATH=<rocm_root>/lib:<rocm_root>/lib/rocm_sysdeps/lib:<rocm_root>/lib/llvm/lib:$LD_LIBRARY_PATH
# See references/10-rocm-versions.md for why each step is needed.
set -euo pipefail
ROOT=${1:?rocm_root}; OUT=${2:?overlay_dir}; TRITON=${3:-}
R=$ROOT/lib
[ -d "$ROOT/.kpack" ] || echo "warning: $ROOT/.kpack missing; RCCL/BLAS kernels will fail with 'invalid device function'" >&2
SRC=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
DST=$OUT/site/torch
rm -rf "$OUT/site"; mkdir -p "$OUT/site"
cp -a "$SRC" "$DST"
n=0
for f in "$DST"/lib/*.so*; do
  b=$(basename "$f")
  # Swap every runtime library torch bundles together: a partial swap loads two
  # HIP runtimes into one process and segfaults.
  case "$b" in
    libamdhip64.so*|libhsa-runtime64.so*|librccl.so*|libhipblaslt.so*|libhipblas.so*|librocblas.so*|\
    libamd_comgr.so*|libhiprtc.so*|librocprofiler-register.so*|librocprofiler-sdk*|libroctracer64.so*|\
    libroctx64.so*|libhsa-amd-aqlprofile64.so*|librocm_smi64.so*|librocm-core.so*) ;;
    *) continue ;;
  esac
  if [ -e "$R/$b" ]; then rm -f "$f"; ln -s "$(readlink -f "$R/$b")" "$f"; n=$((n + 1)); fi
done
for d in hipblaslt rocblas; do
  if [ -d "$R/$d" ]; then rm -rf "${DST:?}/lib/$d"; ln -s "$R/$d" "$DST/lib/$d"; fi
done
if [ -n "$TRITON" ]; then ln -s "$TRITON" "$OUT/site/triton"; fi
echo "swapped $n libraries into $DST"

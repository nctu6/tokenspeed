"""掃 Triton/Gluon 編譯快取裡的 AMDGPU 組語，列出有 register spill（使用 scratch）的 kernel。
用法：python3 scan_spills.py [cache_dir ...]   （預設 ~/.triton/cache 與 $TRITON_CACHE_DIR）
為什麼：spill 的 kernel 慢；在 AMD 上 ROCr 會依「每 lane 私有大小 × 常駐 wave 數」為每個硬體佇列配置 scratch，
KV 吃滿記憶體後首次在新佇列執行大 spill kernel 會 HSA_STATUS_ERROR_OUT_OF_RESOURCES。
NVIDIA：改用 cuobjdump/ptxas -v 看 "bytes spill stores"。
"""

import collections
import glob
import os
import re
import sys

dirs = sys.argv[1:] or [
    d
    for d in (os.environ.get("TRITON_CACHE_DIR"), os.path.expanduser("~/.triton/cache"))
    if d
]
pat_name = re.compile(r"^\s*\.amdhsa_kernel\s+(\S+)", re.M)
pat_priv = re.compile(r"\.amdhsa_private_segment_fixed_size\s+(\d+)")
pat_vgpr = re.compile(r"\.amdhsa_next_free_vgpr\s+(\d+)")
best = collections.defaultdict(lambda: (0, 0, ""))
n = 0
for d in dirs:
    for f in glob.glob(os.path.join(d, "**", "*.amdgcn"), recursive=True):
        n += 1
        s = open(f, errors="ignore").read()
        names, privs, vgprs = (
            pat_name.findall(s),
            pat_priv.findall(s),
            pat_vgpr.findall(s),
        )
        for name, priv, vg in zip(names, privs, vgprs or ["?"] * len(names)):
            priv = int(priv)
            if priv > best[name][0]:
                best[name] = (priv, vg, f)
spill = sorted(
    ((v[0], k, v[1], v[2]) for k, v in best.items() if v[0] > 0), reverse=True
)
print(
    f"scanned {n} .amdgcn files; {len(spill)} kernels with scratch (private segment > 0)"
)
print(f"{'bytes/lane':>10s}  {'vgpr':>5s}  kernel")
for priv, name, vg, f in spill[:60]:
    print(f"{priv:10d}  {vg:>5s}  {name}")

"""GPU-only torch trace -> 每類 kernel 的總時間與「暴露時間」（同時沒有其他 stream 的 kernel 在跑）。
用法：python3 trace_exposed.py <trace_dir> [--rank TP0] [--top-other 30]
換模型/硬體時：先看 --top-other 印出的未分類 kernel，把它們加進 CATS。
"""

import argparse
import bisect
import collections
import glob
import gzip
import json
import re

# (類別, kernel 名稱正則)。順序重要：先匹配者勝。【MI350P/DSv4.1】的命名；換模型要改。
CATS = [
    ("AR-iris", r"iris_"),
    ("RCCL/NCCL", r"nccl|rccl"),
    ("MoE", r"moe|_gather_package|_mxfp4_quantize|sqrt_softplus_topk|silu_and_mul"),
    ("attn", r"attention|dsv4_prefill|rope|swa_|flash|mla"),
    ("indexer", r"index_topk"),
    ("mHC", r"mhc"),
    ("GEMM", r"mm_mxfp8|Cijk|gemm|quantize_fp8|rmsnorm_fp8"),
    ("copy/misc", r"copy|elementwise|Fill|pack|gather|index|SoftMax|memcpy|memset"),
]
PREFILL_AR_US = 500  # AR kernel 超過這個長度視為 prefill 尺寸


def cat(n):
    for c, p in CATS:
        if re.search(p, n):
            return c
    return "other"


def merged(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rank", default="TP0")
    ap.add_argument("--top-other", type=int, default=0)
    a = ap.parse_args()
    f = glob.glob(f"{a.dir}/*{a.rank}*")[0]
    K = [
        e
        for e in json.load(gzip.open(f) if f.endswith("gz") else open(f))["traceEvents"]
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    K.sort(key=lambda k: k["ts"])
    t0 = K[0]["ts"]
    t1 = max(k["ts"] + k["dur"] for k in K)
    busy = sum(b - a for a, b in merged([(k["ts"], k["ts"] + k["dur"]) for k in K]))
    print(
        f"window {(t1-t0)/1e3:.0f} ms  union busy {busy/1e3:.0f} ms ({busy/(t1-t0)*100:.0f}%)  idle {(t1-t0-busy)/1e3:.0f} ms"
    )
    by_s = collections.defaultdict(list)
    for k in K:
        by_s[k["args"].get("stream")].append((k["ts"], k["ts"] + k["dur"]))
    other = {s: merged([x for t, v in by_s.items() if t != s for x in v]) for s in by_s}
    starts = {s: [p for p, _ in other[s]] for s in other}

    def overlap(s, a_, b_):
        iv = other[s]
        i = max(bisect.bisect_right(starts[s], a_) - 1, 0)
        o = 0
        while i < len(iv) and iv[i][0] < b_:
            o += max(0, min(b_, iv[i][1]) - max(a_, iv[i][0]))
            i += 1
        return o

    tot, exp, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    oth = collections.Counter()
    for k in K:
        c = cat(k["name"])
        s = k["args"].get("stream")
        if c == "AR-iris":
            c += " prefill" if k["dur"] > PREFILL_AR_US else " decode"
        if c == "other":
            oth[k["name"][:90]] += k["dur"]
        key = (c, s)
        tot[key] += k["dur"]
        exp[key] += k["dur"] - overlap(s, k["ts"], k["ts"] + k["dur"])
        cnt[key] += 1
    W = (t1 - t0) / 100
    print(
        f"{'category':22s} {'stream':>6s} {'n':>7s} {'total ms':>9s} {'exposed ms':>10s} {'% wall':>7s}"
    )
    for key in sorted(tot, key=lambda x: -exp[x]):
        print(
            f"{key[0]:22s} {str(key[1]):>6s} {cnt[key]:7d} {tot[key]/1e3:9.1f} {exp[key]/1e3:10.1f} {exp[key]/W:7.1f}"
        )
    if a.top_other:
        print("\nunclassified kernels (add to CATS):")
        for n, d in oth.most_common(a.top_other):
            print(f"  {d/1e3:8.1f} ms  {n}")


if __name__ == "__main__":
    main()

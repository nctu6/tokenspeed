"""解析 AgentX 結果目錄，印每格指標與每臂平均、Δ。
用法：
  python3 summarize_cells.py <tag-prefix> [<tag-prefix> ...]        # 每個 prefix 是一臂（例 ABA ABB）
  ROOT=/path python3 summarize_cells.py base                          # 單臂：列出所有格
結果目錄命名：results/agentx/<engine>-<TAG>-c<conc>-d<dur>-<stamp>（run_agentx.sh 產生）
ab_abba.sh 的 TAG 形如 <prefix><arm><conc><rep>，所以 prefix=ABA 會匹配 ABA321、ABA322、ABA641…
無效格（無 export）印 INVALID 並略過，不會中止。
"""

import collections
import glob
import json
import os
import re
import statistics
import sys

ROOT = os.environ.get("ROOT", ".")


def g(x, k, s="avg"):
    try:
        return x[k][s]
    except Exception:
        return float("nan")


arms = sys.argv[1:] or sys.exit(__doc__)
table = collections.defaultdict(dict)  # conc -> arm -> [rpm...]
for arm in arms:
    for d in sorted(glob.glob(f"{ROOT}/results/agentx/*-{arm}*-c*-d*")):
        name = os.path.basename(d)
        m = re.search(r"-c(\d+)-d", name)
        conc = int(m.group(1)) if m else -1
        f = f"{d}/aiperf_artifacts/profile_export_aiperf.json"
        if not os.path.exists(f):
            print(f"{name[:48]:48s} INVALID (no export)")
            continue
        x = json.load(open(f))
        rpm = g(x, "request_throughput") * 60
        err = g(x, "request_error_rate")
        print(
            f"{name[:48]:48s} RPM {rpm:6.1f}  out {g(x,'output_token_throughput'):6.0f}  "
            f"ITL {g(x,'inter_token_latency'):6.2f}/{g(x,'inter_token_latency','p90'):6.2f} ms  "
            f"TTFT {g(x,'time_to_first_token')/1e3:5.2f}/{g(x,'time_to_first_token','p90')/1e3:5.2f} s  "
            f"n {int(g(x,'request_count','avg') or 0)}  err {err:.1f}%"
        )
        table[conc].setdefault(arm, []).append(rpm)
if len(arms) >= 2:
    print("\nmean RPM per arm (Δ vs first arm):")
    for conc in sorted(table):
        row = table[conc]
        base = row.get(arms[0])
        parts = []
        for arm in arms:
            v = row.get(arm)
            if not v:
                parts.append(f"{arm} -")
                continue
            mu = statistics.mean(v)
            d = (
                f" ({(mu/statistics.mean(base)-1)*100:+.1f}%)"
                if base and arm != arms[0]
                else ""
            )
            parts.append(f"{arm} {mu:.1f} [{', '.join(f'{r:.1f}' for r in v)}]{d}")
        sep = ""
        if base and len(arms) >= 2 and row.get(arms[1]):
            sep = (
                "  separated"
                if min(row[arms[1]]) > max(base) or max(row[arms[1]]) < min(base)
                else "  overlapping"
            )
        print(f"  c{conc}: " + " | ".join(parts) + sep)
    print(
        "\n判定：Δ≥+5% 且 separated → 採用候選（換一個 concurrency 再驗）；|Δ|<5% → 中性；有 INVALID → 先補跑"
    )

"""安全地新增 / 更新 experiments.json（檔案鎖 + 寫前重讀，多個 session 共用也不會互相覆蓋）。
新增：python3 log_experiment.py --area Kernels --experiment "..." [--result 待跑] [--decision -] [--commit abc123]
更新：python3 log_experiment.py --update 245 --result "..." --interpretation "..." --decision "否決"
顯示：python3 log_experiment.py --tail 5
環境變數 EXPERIMENTS 指定檔案（預設 $ROOT/experiments.json）。
"""

import argparse
import datetime
import fcntl
import json
import os
import tempfile

p = argparse.ArgumentParser()
p.add_argument("--update", type=int)
p.add_argument("--tail", type=int)
for k in ("area", "experiment", "result", "interpretation", "decision", "commit"):
    p.add_argument(f"--{k}")
a = p.parse_args()
path = os.environ.get(
    "EXPERIMENTS", os.path.join(os.environ.get("ROOT", "."), "experiments.json")
)
if not os.path.exists(path):
    open(path, "w").write("[]")
with open(path + ".lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    data = json.load(open(path))  # 寫前重讀
    if a.tail:
        for e in data[-a.tail :]:
            print(json.dumps(e, ensure_ascii=False, indent=1))
        raise SystemExit
    fields = {
        k: getattr(a, k)
        for k in (
            "area",
            "experiment",
            "result",
            "interpretation",
            "decision",
            "commit",
        )
        if getattr(a, k) is not None
    }
    if a.update is not None:
        hit = [e for e in data if e.get("id") == a.update]
        if not hit:
            raise SystemExit(f"id {a.update} not found")
        hit[0].update(fields)
        eid = a.update
    else:
        if not a.area or not a.experiment:
            raise SystemExit("--area and --experiment are required")
        eid = max([e.get("id", 0) for e in data] + [0]) + 1
        e = {
            "id": eid,
            "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
            "area": a.area,
            "experiment": a.experiment,
            "result": fields.get("result", "待跑"),
            "interpretation": fields.get("interpretation"),
            "decision": fields.get("decision", "-"),
            "commit": fields.get("commit", "-"),
        }
        data.append(e)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)))
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)  # 原子替換
print(
    f"experiments.json: {'updated' if a.update is not None else 'added'} id {eid}. 記得更新 dashboard。"
)

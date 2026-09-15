#!/usr/bin/env python3
"""实验注册表：每次任务判定后追加一行到 registry/runs.jsonl。
用法：python3 scripts/log_run.py --task M1-data-audit --status PASS [--note "..."] [--result outputs/M1/data_verdict.md]
"""
import argparse
import datetime
import json
import os
import subprocess

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rp(*a):
    return os.path.join(root, *a)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "no-git"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--status", required=True, choices=["PASS", "FAIL"])
    ap.add_argument("--note", default="")
    ap.add_argument("--result", default="")
    a = ap.parse_args()

    active = {}
    ap_path = rp("registry", "active_task.json")
    if os.path.exists(ap_path):
        with open(ap_path, encoding="utf-8") as f:
            active = json.load(f)

    entry = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": a.task,
        "status": a.status,
        "git_commit": git_commit(),
        "contract": active.get("contract", ""),
        "result": a.result,
        "note": a.note,
    }
    with open(rp("registry", "runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + chr(10))
    print(f"[log_run] 已登记: {a.task} -> {a.status}")


if __name__ == "__main__":
    main()

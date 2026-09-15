#!/usr/bin/env python3
"""机械验收：判定权在脚本，不在模型。用法：python3 scripts/accept_check.py plan/tasks/<task>.yaml

支持的 acceptance 规则类型：
  file_exists       { path }
  csv_min_rows      { path, min_rows }
  content_contains  { path, needle }
  metric_min        { path, key(点分路径), min }   # 支持 JSON / YAML 指标文件
退出码：全 PASS=0，任一 FAIL=1。
"""
import csv
import json
import os
import sys

try:
    import yaml
except ImportError:
    print("缺少 pyyaml：pip install pyyaml")
    sys.exit(1)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rp(*a):
    return os.path.join(root, *a)


def get_dotted(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)] if int(part) < len(cur) else None
        else:
            return None
    return cur


def check(rule):
    t = rule.get("type")
    path = rp(rule.get("path", ""))
    if t == "file_exists":
        return os.path.isfile(path), f"exists={os.path.isfile(path)}"
    if t == "csv_min_rows":
        if not os.path.isfile(path):
            return False, "file missing"
        with open(path, encoding="utf-8-sig", newline="") as f:
            n = sum(1 for row in csv.reader(f) if any(c.strip() for c in row))
        return (n - 1) >= int(rule.get("min_rows", 1)), f"data_rows={max(n - 1, 0)} (need >= {rule.get('min_rows')})"
    if t == "content_contains":
        if not os.path.isfile(path):
            return False, "file missing"
        text = open(path, encoding="utf-8", errors="replace").read()
        ok = rule.get("needle", "") in text
        return ok, f"needle '{rule.get('needle')}' found={ok}"
    if t == "metric_min":
        if not os.path.isfile(path):
            return False, "file missing"
        text = open(path, encoding="utf-8", errors="replace").read()
        try:
            obj = json.loads(text)
        except Exception:
            obj = yaml.safe_load(text)
        val = get_dotted(obj or {}, rule.get("key", ""))
        try:
            ok = val is not None and float(val) >= float(rule.get("min", 0))
        except (TypeError, ValueError):
            ok = False
        return ok, f"{rule.get('key')}={val} (need >= {rule.get('min')})"
    return False, f"unknown rule type: {t}"


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/accept_check.py plan/tasks/<task>.yaml")
        sys.exit(1)
    with open(rp(sys.argv[1]), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rules = cfg.get("acceptance") or []
    if not rules:
        print("[accept_check] 合同无 acceptance 规则 —— 视为 FAIL（合同必须可机械验收）")
        sys.exit(1)
    fails = 0
    for i, rule in enumerate(rules, 1):
        ok, detail = check(rule)
        print(f"[{'PASS' if ok else 'FAIL'}] #{i} {rule.get('type')}: {rule.get('path', '')} :: {detail}")
        fails += 0 if ok else 1
    verdict = "PASS" if fails == 0 else "FAIL"
    print()
    print(f"[accept_check] 任务 {cfg.get('task')}: {verdict}（{len(rules) - fails}/{len(rules)} 条通过）")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()

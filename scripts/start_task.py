#!/usr/bin/env python3
"""激活一个任务合同。用法：python3 scripts/start_task.py plan/tasks/<task>.yaml

检查：人工门标志、registry/STOP。通过后写 registry/active_task.json。
只有这个文件能把训练类命令"解锁"（见 .claude/hooks/gate_check.py）。
"""
import json
import os
import sys
import time

try:
    import yaml
except ImportError:
    print("缺少 pyyaml：pip install pyyaml")
    sys.exit(1)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rp(*a):
    return os.path.join(root, *a)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 scripts/start_task.py plan/tasks/<task>.yaml")
        sys.exit(1)
    contract = sys.argv[1]
    with open(rp(contract), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if os.path.exists(rp("registry", "STOP")):
        print("[start_task] registry/STOP 存在：人工暂停中。")
        sys.exit(1)

    gate = cfg.get("gate_required")
    if gate and not os.path.exists(rp("registry", "gates", gate)):
        print(f"[start_task] 人工门 {gate} 未通过（registry/gates/{gate} 不存在）。"
              f"停止，向人类报告。")
        sys.exit(1)

    active = {
        "task": cfg["task"],
        "contract": contract,
        "gate_required": gate,
        "started_at": time.time(),
        "budget": cfg.get("budget", {}),
    }
    with open(rp("registry", "active_task.json"), "w", encoding="utf-8") as f:
        json.dump(active, f, ensure_ascii=False, indent=2)
    print(f"[start_task] 任务已激活: {cfg['task']}（预算: {active['budget']}）")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PreToolUse hook — 长训练任务启动前的 人工门+预算+暂停 检查（exit 2 = 拦截）。

规则：
  1. 训练类 Bash 命令必须先有 registry/active_task.json（由 scripts/start_task.py 创建）
  2. 活动任务声明的 gate_required 标志文件必须存在（人工门）
  3. registry/STOP 存在 = 人工暂停，一切训练拦截
  4. 墙钟预算超限 = 拦截，等人工解锁
"""
import json
import os
import sys
import time

TRAIN_PATTERNS = ("scripts/train", "anomalib fit", "-m anomalib", "yolo ", "torchrun", "accelerate launch")


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if data.get("tool_name") != "Bash":
        return
    cmd = str((data.get("tool_input") or {}).get("command", "")).lower()
    if not any(p in cmd for p in TRAIN_PATTERNS):
        return

    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

    def rp(*a):
        return os.path.join(root, *a)

    if os.path.exists(rp("registry", "STOP")):
        print("[gate_check] registry/STOP 存在：人工暂停中，禁止启动训练。", file=sys.stderr)
        sys.exit(2)

    active_path = rp("registry", "active_task.json")
    if not os.path.exists(active_path):
        print("[gate_check] 未激活任务：训练前必须先运行 scripts/start_task.py <合同路径>。", file=sys.stderr)
        sys.exit(2)

    with open(active_path, encoding="utf-8") as f:
        active = json.load(f)

    gate = active.get("gate_required")
    if gate and not os.path.exists(rp("registry", "gates", gate)):
        print(f"[gate_check] 人工门 {gate} 未通过（标志文件不存在）。"
              f"停止并输出'等待人类门裁决'。禁止自行过门。", file=sys.stderr)
        sys.exit(2)

    budget = active.get("budget") or {}
    wall_h = float(budget.get("wall_clock_h", 24))
    started = float(active.get("started_at", time.time()))
    if time.time() - started > wall_h * 3600:
        print(f"[gate_check] 墙钟预算超限（{wall_h}h）。已拦截；请人类评估后删除 "
              f"registry/active_task.json 解锁。", file=sys.stderr)
        sys.exit(2)


main()

#!/usr/bin/env python3
"""PreToolUse hook — 冻结区写保护（Claude Code 约定：exit 2 = 拦截并把 stderr 反馈给模型）。

保护对象：
  plan/**                 冻结的研究方案与任务合同
  registry/gates/**       人工门标志（只有人类能创建）
  data/splits/**          数据划分（协议冻结的一部分）
  .claude/hooks/**, .claude/settings.json, CLAUDE.md   约束机制本身
"""
import json
import os
import sys

FROZEN_TOKENS = ("plan/", "registry/gates", "data/splits", ".claude/hooks")
FROZEN_BASENAMES = {"plan_v1.1.md", "claude.md", "settings.json"}
BASH_FROZEN_TOKENS = ("plan/", "registry/gates", "data/splits", ".claude/hooks", "claude.md")
BASH_VERBS = (">", ">>", "mv ", "cp ", "rm ", "sed -i", "git checkout", "git restore", "tee ")


def norm(p):
    return str(p).replace("\\", "/").lower()


def blocked_path(path):
    p = norm(path)
    base = os.path.basename(p)
    return any(t in p for t in FROZEN_TOKENS) or base in FROZEN_BASENAMES


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return  # 非 JSON 输入：放行（交由其他机制处理）
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}

    if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = ti.get("file_path") or ti.get("notebook_path") or ""
        if path and blocked_path(path):
            print(f"[guard_freeze] 已拦截：'{path}' 位于冻结区（计划/人工门/划分/约束机制只读）。"
                  f"如确需变更，停止并请人类直接修改。", file=sys.stderr)
            sys.exit(2)

    elif tool == "Bash":
        cmd = str(ti.get("command", "")).lower()
        hits_frozen = any(t in cmd for t in BASH_FROZEN_TOKENS)
        hits_verb = any(v in cmd for v in BASH_VERBS)
        if hits_frozen and hits_verb:
            print("[guard_freeze] 已拦截：命令疑似修改冻结区。读取类命令不受影响；"
                  "如确需变更，停止并请人类处理。", file=sys.stderr)
            sys.exit(2)


main()

#!/usr/bin/env python3
"""一键自检：验证 冻结写保护 / 门与预算检查 / 任务激活 / 机械验收 / 注册表 全链路。
用法：python3 scripts/selftest.py   （部署后先跑这个，全 PASS 再交给 Agent）
"""
import json
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(args, stdin_text=None):
    p = subprocess.run(args, input=stdin_text, capture_output=True, text=True, cwd=root)
    return p.returncode, ((p.stderr or "") + (p.stdout or "")).strip().replace(chr(10), " | ")


results = []


def check(name, cond, detail=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} :: {detail}")


# 1. 冻结写保护：Edit 计划文件 -> exit 2
code, out = run([PY, ".claude/hooks/guard_freeze.py"],
                json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "plan/PLAN_v1.1.md"}}))
check("guard_freeze 拦截 Edit 冻结计划", code == 2, f"exit={code}")

# 2. 冻结写保护：Bash 拷贝冻结文件 -> exit 2
code, out = run([PY, ".claude/hooks/guard_freeze.py"],
                json.dumps({"tool_name": "Bash", "tool_input": {"command": "cp plan/PLAN_v1.1.md backup.md"}}))
check("guard_freeze 拦截 Bash 拷贝冻结文件", code == 2, f"exit={code}")

# 3. 冻结写保护：正常文件放行 -> exit 0
code, out = run([PY, ".claude/hooks/guard_freeze.py"],
                json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "outputs/M1/notes.md"}}))
check("guard_freeze 放行正常写入", code == 0, f"exit={code}")

# 4. 门检查：无活动任务 -> 拦截 exit 2
ap = os.path.join(root, "registry", "active_task.json")
existed = os.path.exists(ap)
if existed:
    os.remove(ap)
code, out = run([PY, ".claude/hooks/gate_check.py"],
                json.dumps({"tool_name": "Bash", "tool_input": {"command": "anomalib fit --model PatchCore"}}))
check("gate_check 拦截未激活任务的训练", code == 2, f"exit={code}")

# 5. start_task 激活合同
code, out = run([PY, "scripts/start_task.py", "plan/tasks/M1-data-audit.yaml"])
check("start_task 激活 M1 合同", code == 0, f"exit={code}")

# 6. 门检查：活动任务内的训练放行 -> exit 0
code, out = run([PY, ".claude/hooks/gate_check.py"],
                json.dumps({"tool_name": "Bash", "tool_input": {"command": "python scripts/train.py --cfg x"}}))
check("gate_check 放行活动任务内训练", code == 0, f"exit={code}")

# 7. 机械验收：产物缺失 -> FAIL (exit 1)
code, out = run([PY, "scripts/accept_check.py", "plan/tasks/M1-data-audit.yaml"])
check("accept_check 对缺失产物判 FAIL", code == 1, f"exit={code}")

# 8. 注册表写入 + 清理测试行
code, out = run([PY, "scripts/log_run.py", "--task", "selftest", "--status", "FAIL", "--note", "smoke"])
check("log_run 登记注册表", code == 0, f"exit={code}")

# 清理：恢复"无活动任务"状态，移除自测行
if not existed and os.path.exists(ap):
    os.remove(ap)
runs_path = os.path.join(root, "registry", "runs.jsonl")
with open(runs_path, encoding="utf-8") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]
lines = [l for l in lines if json.loads(l).get("task") != "selftest"]
with open(runs_path, "w", encoding="utf-8") as f:
    f.write(chr(10).join(lines) + (chr(10) if lines else ""))

passed = sum(results)
print()
print(f"[selftest] {passed}/{len(results)} 通过" + ("" if passed == len(results) else " —— 有 FAIL，先修复再交给 Agent"))
sys.exit(0 if passed == len(results) else 1)

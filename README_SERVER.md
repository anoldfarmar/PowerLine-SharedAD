# xuanti-exp 服务器部署说明

把计划（PLAN_v1.1）变成"合同 + 门 + 预算"三层约束的实验执行工作区。
软约束说服模型，硬约束拦截模型，验收脚本替模型做判定。

## 目录
```
CLAUDE.md            宪法（常驻上下文，≤50 行）
plan/PLAN_v1.1.md    冻结的研究方案（只读，建议 git tag plan-v1.1）
plan/tasks/*.yaml    任务合同（Agent 行动的唯一许可范围）
registry/gates/      人工门标志（只有人类能 touch）
registry/runs.jsonl  实验注册表（commit/合同/结果，ARS provenance 回流的输入）
registry/STOP        人工暂停开关（创建即全局停训练）
.claude/settings.json 权限 allow/deny + hooks 注册
.claude/hooks/       guard_freeze.py（冻结区写保护）+ gate_check.py（门+预算检查）
scripts/             start_task / accept_check / log_run
```

## 服务器安装
```bash
# python3 + 依赖
pip install pyyaml          # anomalib 环境通常已含
# Claude Code 按官方文档安装；本目录作为项目根
```
注意：本包按 Linux 服务器写的是 `python3`；Windows 本地试跑请把 settings.json 中的
`python3` 临时改成 `python`。

## 人工操作（只有三件事）
```bash
touch registry/gates/M1_PASSED     # 通过某个人工门
touch registry/STOP                # 全局暂停（删除即恢复）
rm registry/active_task.json       # 解除超限任务
```

## 无值守运行
- 单次：`claude -p "执行 plan/tasks/M1-data-audit.yaml"`（配合 tmux）
- 定时/远程触发：见 Claude Code 官方 Scheduled Tasks 文档
- 权限提示：长任务不要用 bypassPermissions 裸奔；靠 settings.json 的 allow/deny + hooks

## 部署后自检（先跑这个，全 PASS 再交给 Agent）
```bash
python3 scripts/selftest.py     # 8 项全链路自检：写保护/门/预算/激活/验收/注册表
```

## 与 ARS 的回流
runs.jsonl 每行含 git commit + 合同引用 —— 写作阶段喂给 academic-pipeline 的
experiment provenance intake（Material Passport），/ars-full 有据可查。
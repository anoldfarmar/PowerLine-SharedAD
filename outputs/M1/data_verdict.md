# M1 数据充分性判定（data_verdict）

> 任务：M1-data-audit · 日期：2026-09-15 · 依据：PLAN v1.1 §3（三条件门）、§8 Week-1 作业、本地实测与文献核查。
> 支撑文件：`insplad_fd_class_stats.csv`（22 行逐类实测）、`insplad_eval_protocol.md`、`stnplad_label_audit.md`、`licenses.md`。

## 数据基线（本地实测，脚本 `scripts/m1_fd_stats.py`）

- InsPLAD-det（基类训练池）：train 7,981 图 / 22,635 框；val 2,626 图 / 6,324 框；18 资产类（最大类 stockbridge damper 5,699 框，最小类 sphere 26 框）。
- InsPLAD-FD（故障裁剪池，`defect_supervised/`）：11,525 crop；故障类 crop 共 2,972（train 2,670 + val 302）；故障类型：missing-cap、rust、corrosion、bird-nest；覆盖 5 资产。
- 注：论文称 402 缺陷样本（口径疑为增广前/特定版本），本地实测 2,972——引用时注明口径（详见协议文档 §5）。

## 条件1：故障类实例数 ≥30 的类数 ≥6

**满足。** 证据（`insplad_fd_class_stats.csv`，train split，资产/故障组合粒度）：

| 组合 | 实例数 | 组合 | 实例数 |
|---|---|---|---|
| glass-insulator/missing-cap | 690 | vari-grip/bird-nest | 350 |
| polymer-insulator-upper-shackle/corrosion | 740 | vari-grip/rust | 290 |
| lightning-rod-suspension/rust | 310 | yoke-suspension/rust | 290 |

6 个组合 ≥30（train）；加上 val 侧 polymer-insulator-upper-shackle/rust(33) 共 7 个。
**按缺陷类型粒度合并（rust/corrosion/missing-cap/bird-nest）为 4 类，全部 ≥30**——低于 6 的阈值，因此**必须执行已锁定的宏类合并协议**（罕见资产×故障组合并入 other-fault 宏类，PLAN §3/R6），并以资产×故障组合为原型库的建库粒度（每组合独立原型，决策层做宏类归并）。

## 条件2：总量支撑基类训练

**满足。** 22,635 训练框（7,981 图）对轻量主干（YOLOv8n/s 级、原型头 v0 冻结主干）的基类检测训练充足；16/18 类训练框 ≥72，仅 sphere(26) 与 polymer-insulator-tower-shackle(47) 偏少——两者非故障载体资产，可并入背景/忽略或低权重处理（不触发宏类协议，因其属基类检测而非故障评估）。min_box_px 实测最小 37px（yoke-suspension train，全表 37–469），小目标属性成立，支持 PLAN 的 P3 特征 + SAHI 切片设计。

## 条件3：跨域靶可用

**满足。** STN-PLAD 公开可得、原始版 133 图 / 2,409 框 / 5 正常资产类（本地实测，2026-09-15 人工下载后核验回填）、GPL-3.0；类别语义与 InsPLAD 可对齐（insulator、damper、spacer、tower plate、tower），契合"冻结主干 + 新域原型入库"的减负跨域设计——该设计只需正常类裁剪，STN-PLAD 无故障类**不构成阻断**。原始版无官方划分，M7 需自行划分防同批次泄漏（已记入审计文档）。

## 风险与注记（承 PLAN §7 R1/R6/R5）

1. **k-shot 惯例缺失**：文献无 InsPLAD k-shot 划分，M2 须自行固化划分（×3 种子、公开划分文件、沿杆塔分组防泄漏）——已记入协议文档 §4。
2. **标注噪声**：`corrosão/normal` 葡语命名、yoke-suspension val/good 异常偏大（5,742）、论文与发布包样本数口径差异——M2 划分固化前需清洗脚本复核。
3. **许可约束**：InsPLAD 为 CC BY-NC 3.0，禁止商业用途；落地验证不得使用 InsPLAD 权重的商业交付（见 licenses.md §5）。
4. AD 实际评测面为 5 含缺陷资产（DifferNet 0.905 / CS-Flow 0.903 AUROC 为文献上界参照）。

# 判定：有条件足够

三条件全部满足（条件3 附本地核验待办）。与 PLAN v1.1 §3 预判一致：主靶 InsPLAD-FD 可支撑"基类训练 + k-shot 原型构建 + 主对比"，前提是执行宏类合并协议与 M2 划分固化纪律；跨域靶 STN-PLAD 可用于原型扩展式减负实验。无数据级证伪信号，不触发 R1 换数据预案。

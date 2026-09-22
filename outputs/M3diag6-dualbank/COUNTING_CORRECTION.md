# 计数更正说明: "118 组" vs "glass 83 + yoke 37"

> 任务: M3DIAG6-dualbank-score-v1 前置审计 · 日期: 2026-09-21 · 依据工件: outputs/M3diag5-testaudit/matching_full.json
> 历史文件 SHOT_AUDIT.md 保留原文不改, 本文件为唯一更正记录。

## 1. 矛盾内容

- SHOT_AUDIT.md §4 标题: "全量结果 (118 组, 阈值 0.8)"
- 同节表格: glass missing-cap 83 组 + yoke rust 37 组 = **120**
- 同节读法: "确认同帧 glass ≥3 组; 其余 **112** 组" (120-3=117, 亦不符)

## 2. 统计单位与真实分母 (依 matching_full.json 权威字段)

- **统计单位**: 故障裁剪源帧的 (会话日期前缀, DJI) 组, 仅限本任务两配置 (glass-insulator/missing-cap, yoke-suspension/rust) 的故障侧;
- **真实分母: 120 组 = glass 83 + yoke 37** (matching_full.json groups 字段逐字计数);
- 档位分解: 确认同帧 in det **3 组** (glass confirmed_train=2 + confirmed_val_only=1; yoke 0); 0.6-0.8 边界 10 组 (glass 8 + yoke 2, 见 SHOT_AUDIT §4 表格); 其余 107 组低于 0.6 未确证;
- "其余" = **117 组** (120-3) 同源关系未确证, 非 112。

## 3. 定性

matching_full.json 中不存在 118 或 112 的任何分组计数, 两组数字无法从工件溯源, 判定为笔误。
不影响结论: 无论 117/118/112, "同源关系未知不等价于确认未见" 的三档表述与 Branch C 判定均不变。

## 4. 口径区分

- (会话,DJI) 精化口径: glass 故障 83 组 / yoke 37 组 (本文件与 SHOT_AUDIT §2.3);
- 裸 DJI 号口径 (M2 冻结, DATA_AUDIT): glass 故障 81 组 / yoke 37 组, 跨航次重号被并组, 分母更小;
- 两口径均非 118。

## 5. 对本任务 (M3DIAG6) 的约束

- 报告中 k 写作 "历史 k-crop 支持预算" 并附实际源组数 (k10 支持 = 10 裁剪来自 1-3 个源组, 见 SHOT_AUDIT §2.1);
- 源组身份未知不得升级为确认未见; 本任务整体为探索性诊断。
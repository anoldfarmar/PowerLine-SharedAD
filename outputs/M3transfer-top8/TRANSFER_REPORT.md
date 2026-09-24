# TRANSFER_REPORT — M3TRANSFER-top8-v1（探索性）

- 2 资产 (yoke-suspension/rust, glass-insulator/missing-cap) × 2 特征 (YOLO-P3 64d L2 5N, WRN50 layer2+3 1536d 10N) × 2 评分 (官方 PC nn=9, top8 欧氏) × 5 参照组 = **40 配置**, 集合断言 set_match=1
- 声明: shot 为裁剪预算; 查询跨臂复用; 源帧身份限制未核实; glass 为跨资产迁移验证, 非真正未见数据的确认性实验; 无显著性检验声明
- 同组 PC 与 T8 共享特征张量/库/最近邻距离/查询路径, 仅最终聚合不同 (same_bank_identity=1)

## 1. 结果矩阵 (AUROC, 5 参照组)

**yoke-suspension/rust**:

| 臂 | 均值 | 范围 | std(ddof=0) |
|---|---|---|---|
| Y-PC | 0.8119 | [0.7237, 0.8549] | 0.0477 |
| Y-T8 | 0.8734 | [0.7788, 0.9207] | 0.0511 |
| W-PC | 0.8882 | [0.8780, 0.8971] | 0.0061 |
| W-T8 | 0.9062 | [0.8978, 0.9227] | 0.0088 |

**glass-insulator/missing-cap**:

| 臂 | 均值 | 范围 | std(ddof=0) |
|---|---|---|---|
| Y-PC | 0.7689 | [0.7407, 0.8041] | 0.0234 |
| Y-T8 | 0.8063 | [0.7642, 0.8422] | 0.0278 |
| W-PC | 0.7903 | [0.7828, 0.8040] | 0.0073 |
| W-T8 | 0.8209 | [0.8143, 0.8364] | 0.0081 |

## 2. yoke YOLO 锚点 (复现门)

以 M3SCORE 的 E1 缓存/固定 N*/数值路径为锚点: Y-PC 逐路径复现 S4 **max|diff|=0.0** (门限 1e-4), Y-T8 逐路径复现 S2 **max|diff|=0.0**; 正常库张量哈希与容量逐组断言一致。yoke YOLO 10 配置直接复用为矩阵单元。

## 3. 库内聚合收益 (同库同距离, 仅聚合不同)

| 比较 | yoke 均值 | glass 均值 | yoke 正/负 | glass 正/负 |
|---|---|---|---|---|
| YOLO: T8−PC | **0.0615** | **0.0373** | 5/5, 0/5 | 5/5, 0/5 |
| WRN: T8−PC | **0.0180** | **0.0306** | 5/5, 0/5 | 5/5, 0/5 |

- top8 收益在 4 组特征×资产组合全部 5/5 组为正 (合计 20/20): 聚合改进同时惠及 YOLO 与 WRN, 且从 yoke 迁移到 glass 保持 (YOLO glass +0.0373, 5/5; 量级小于 yoke 的 +0.0615)。
- 这是**同库内**的聚合效应; 跨库/跨特征的比较见 §4, 不能与聚合收益混读。

## 4. 外部比较 (不同支持预算的系统对比)

| 比较 | yoke 均值 | glass 均值 | 正/负 (yoke, glass) |
|---|---|---|---|
| Y-T8 − W-PC (vs 标准 WRN) | -0.0147 | 0.0159 | 2/5, 3/5 |
| Y-T8 − W-T8 (vs 加强 WRN) | -0.0327 | -0.0147 | 2/5, 2/5 |

- W-PC(0.888/0.790) 与 W-T8(0.906/0.821) 在两资产均为最高; YOLO 5N 系统整体低于 WRN 10N 系统, 但 glass 上 Y-T8 与 W-PC 基本持平 (+0.0159, 3/5 正)。
- YOLO5N 与 WRN10N 是不同支持预算的系统对比, 不能归结为纯主干能力差异。

## 5. WRN 历史库回退披露 (合同 §七)

M3BENCH 未留存库/特征/patch 距离 → 按原标准配置确定性重建 (种子 pc_seed=7001..7005, 双建库核验 ref3001 torch.equal)。历史 W-PC 仅作参考:

| 组 | 历史 W-PC | 新 W-PC | 差 |
|---|---|---|---|
| yoke_ref3001 | 0.7971 | 0.8780 | 0.0809 |
| yoke_ref3002 | 0.8056 | 0.8895 | 0.0840 |
| yoke_ref3003 | 0.8882 | 0.8971 | 0.0089 |
| yoke_ref3004 | 0.8264 | 0.8881 | 0.0618 |
| yoke_ref3005 | 0.8570 | 0.8882 | 0.0312 |
| glass_ref3001 | 0.7926 | 0.7905 | -0.0020 |
| glass_ref3002 | 0.7179 | 0.7877 | 0.0698 |
| glass_ref3003 | 0.8075 | 0.8040 | -0.0035 |
| glass_ref3004 | 0.8285 | 0.7828 | -0.0457 |
| glass_ref3005 | 0.8042 | 0.7867 | -0.0175 |

- 新旧 W-PC 逐组差 yoke 0.0089..0.0840, glass -0.0457..0.0698 — **库随机性量级可达 ±0.08**, 大于 top8 收益 (0.018-0.037) 与历史 W-PC−YOLO 差距; 未挑种子追平历史均值, 未把新旧 W-PC 差异解释为聚合收益。

## 6. 正确性门 (正式评分前立即 assert)

| 门 | 结果 |
|---|---|
| top8 轴 | patch 轴正确, 错误轴(列)结果确实不同: {"axis_ok": true, "wrong_axis_differs": true}
| reshape | (batch,h,w) 行主序: {"shape_ok": true, "position_ok": true, "order": "patch \u884c\u5e8f = (batch, h, w) \u884c\u4e3b\u5e8f, \u884c\u5185\u4e3a\u8be5\u4f4d\u7f6e C \u7ef4\u5411\u91cf"}
| WRN 距离 | 官方分块(2500 行跨块+尾块) vs float64 直接: {"max_abs_err": 3.6780136767333715e-06, "nn_index_ok": true, "chunks": 3, "tail": true, "pass": true}
| WRN 独立组合 | 官方 compute_anomaly_score vs float64 公式: {"max_abs_diff": 1.1920928955078125e-07, "pass": true}
| WRN batch | 同库 batch=1 vs 16: {"max_abs": 0.000213623046875, "pass": true}
| WRN 库重建 | {"ref3001": true, "n_bank": 1600} (同种子)
| glass 提取 | ext/cache diff: {"ext_diff": 0.0, "cache_diff": 0.0}
| YOLO 缓存身份 | yoke 缓存 SHA256+确定性非TF32+路径集合断言; glass 新缓存同规则
| AUROC 重算 | 40/40 配置从逐图分数重算, 与存档差 <1e-12

## 7. 资源

- 总 wall=104.2s (GPU 预算 2h 上限); 阶段: {"gates_s": 16.2, "anchor_s": 8.4, "glass_yolo_s": 30.2, "wrn_s": 48.7}
- VRAM 峰值 allocated=1173 MiB / reserved=1876 MiB; WRN 距离 npz {"yoke": "20 MiB", "glass": "31 MiB"} (数据盘, 不提交)
- 评分耗时为缓存特征计算, 非端到端推理延迟

## 8. 五个最终回答

1. **top8 收益迁移到 glass: 是** — Y-T8−Y-PC = +0.0373 (5/5 组; 10/10 YOLO 配置为正), 量级小于 yoke (+0.0615)。支持冻结解释 A: 收益跨这两个资产保持, 仍需更广数据验证。
2. **WRN 同样聚合提升**: yoke +0.0180 (5/5), glass +0.0306 (5/5); W-T8 成为两资产最高 (0.9062 / 0.8209) — 聚合改动惠及双方, 此前领先不能归功于 YOLO 主干 (解释 C 成立一半: WRN 也提升并领先, 但 YOLO-T8 收益幅度更大)。
3. **YOLO-T8 相对基线**: 对标准 W-PC: yoke −0.0147 / glass +0.0159 (互有胜负); 对加强 W-T8: yoke −0.0327 / glass −0.0147 (落后)。注: 5N vs 10N 支持预算不同, 是系统级对比。
4. **最有竞争力: W-T8** (重建确定性库), 两资产最高且库内方差最小 (std 0.006-0.009)。代价: WRN 特征路径 24.85M 参数 (YOLO 冻结全模型 3.01M, P3 子图 80.7k)、1536 维 patch 特征 vs 64 维、10N vs 5N 支持、1600 vs 800 原型。预算受限场景下 Y-T8 (64 维/5N) 在 glass 距 W-T8 仅 0.015, 是轻量备选。
5. **下一步**: 先**统一支持预算并固定建库随机性**再比较系统——本轮证据显示库随机性 (重建 vs 历史差 ±0.08) 可吞掉聚合收益 (0.018-0.062), 不固定将淹没结论; 统一预算后再扩大资产范围, 隔离/部署验证随后。探索性判断, 不自动触发。

## 9. 产物

FROZEN_INPUTS.json / CACHE_MANIFEST.json (yoke 复用+glass 新建) / BANK_MANIFEST.json (合并于 transfer_eval bank_meta: 库哈希/容量/coreset 索引) / REPRO_AND_CORRECTNESS.json / WRN_REUSE_OR_REBUILD.md / raw_scores/ (40, JSON 全精度) / transfer_eval.json / paired_comparisons.csv / RESOURCE_COST.json / transfer_matrix.png / run.log
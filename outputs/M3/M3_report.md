# M3 报告：原型头 v1（patch-pre / metric-v1）与同查询集 AD 基线

> 任务：M3-proto-head-v1 · 报告日期：2026-09-17
> 合同：`plan/tasks/M3-proto-head-v1.yaml`（gate_required=M2_PASSED，human_gate_after=M3）
> 权重：`/root/autodl-tmp/runs/yolov8n_det/weights/best.pt`（sha256 `ea1526c2…`，与 M2 冻结一致）
> 环境：RTX 4080 SUPER · torch 2.8.0+cu128 · ultralytics 8.4.151 · anomalib 2.6.2 · timm 1.0.29
> 冻结协议：`outputs/M3/preflight.json`（12/12 划分哈希核验；评测前先行登记全部聚合/投影/coreset 决策）

## 0. 执行摘要

1. **M3 科学判据（mean AUROC 相对 actual-M2-v0 提升 ≥0.15）未达成**：四个变体中相对
   v0 的最大提升为 +0.053（metric-v1，k=1）；k=5/k=10 上全部 patch 变体均低于 v0。
   按合同 notes，该判据只作人工门决策信号，不影响任务验收；如实登记。
2. **特征层勘误（重要披露）**：M2 报告标注"P3 (stride-8)"，实际实现为 SPPF P5
   （stride-32，model[:10] 末层）。M2 产物不回改，以 `actual-M2-v0` 登记并在本报告披露
   （见 §2）。P3 并未带来增益——瓶颈不在特征分辨率层。
3. **patch 级相似度呈"纹理/结构权衡"**：锈蚀类上涨（pius/rust k1：0.52→0.76；
   vari-grip/rust k1：0.36→0.63），结构类下跌（missing-cap k1：0.63→0.44；bird-nest k1：0.63→0.42）。
4. **腐蚀专项量化发现**：top-8 响应 patch 中 **43.3%（52/120）落在图像 4px 边缘带**——
   高相似响应大量来自背景/边界而非缺陷纹理，直接支持创新点二"背景原型抑制"组件的必要性（§5）。
5. **同查询集 AD 基线**（PatchCore/PaDiM/EfficientAD）结果见 §6——它同时充当
   "瓶颈探针"：若 PatchCore 显著高于本方案，则瓶颈在特征源而非任务；反之在任务/分辨率。

## 1. 主结果（mean AUROC ± std，7 故障组合，3 种子）

| 变体 | k=1 | k=5 | k=10 |
|---|---|---|---|
| actual-M2-v0（SPPF-P5 GAP） | 0.565 ± 0.067 | **0.673 ± 0.005** | **0.653 ± 0.052** |
| P3-GAP | 0.578 ± 0.068 | 0.611 ± 0.052 | 0.594 ± 0.022 |
| P3-patch-pre（固定随机投影 64→128） | 0.476 ± 0.026 | 0.516 ± 0.029 | 0.499 ± 0.023 |
| P3-patch-metric-v1（基类 SupCon 头） | **0.618 ± 0.062** | 0.612 ± 0.061 | 0.603 ± 0.009 |

缺陷召回（mean+2std 固定阈值，零调参）：v0 0.142/0.185/0.151；P3-GAP 0.036/0.038/0.031；
patch-pre 0.046/0.056/0.046；metric-v1 0.068/0.082/0.034。召回普遍低是阈值规则保守所致，
H1 的召回主张评估须按 PLAN §5.3 用 PR 曲线/部署工作点，不能只看此列。

全量逐组合/逐种子数值：`outputs/M3/proto_v1_eval.json`（不挑选、不省略）。

## 2. 特征层审计与 P3/P5 勘误

`feature_layer_audit.json`：P3=layer3/4（C2f, 64d, stride-8, 40×40）形状验证通过（p3_stride_verified=1）。
M2 v0 实际取 `model[:10]` 末层 = SPPF（P5, 256d, stride-32）做 GAP，与 M2 报告的"P3"标注不符。
处理：M2 产物冻结不修改，评测以 `actual-M2-v0` 为名登记，本报告与 `feature_layer_audit.json` 双重披露。

## 3. 变体归因（合同 notes 要求的分解）

- **P3-GAP vs actual-M2-v0**：k1 持平（+0.013），k5/k10 下降（−0.062/−0.059）→ 特征层修正
  本身无增益，检测微调的语义集中在深层。
- **patch-pre vs P3-GAP**：k1 −0.102 → 固定随机投影不能保留 patch 判别信息，
  检测微调特征不加度量学习时 patch 路线整体劣于 GAP。
- **metric-v1 vs patch-pre**：k1 +0.142 / k5 +0.096 / k10 +0.104 → 度量学习是 patch 路线的
  必要组件（SupCon，18 基类 22635 框，无目标故障 query/k-shot 反传）。
- **metric-v1 vs actual-M2-v0**：仅 k1 +0.053，k5/k10 为负 → 当前特征源下度量学习
  尚不足以支撑 H1。

## 4. 覆盖与统计局限（合同禁止省略）

7 个故障组合全覆盖 × 3 种子 × 3 shots。`polymer-insulator-upper-shackle/rust`（33 实例，
k=10 查询侧仅 3 实例）统计意义弱，冻结划分清单要求标注，全表保留该组合并在图/表中注明。

## 5. 腐蚀类专项分析（Step-6）

产出：`outputs/M3/rust_analysis.json` + `rust_overlays/`（25 张 top-8 响应 overlay PNG，
含 5 组合 × 3 查询样本 + 2 原型样本，patch-pre 变体 k=5 seed=0）。

要点：
1. 5 个 rust/corrosion 组合 × 4 变体全量数值已入档（同一冻结协议，不重算）；
2. **边缘响应量化**：top-8 patch 中 43.3%（52/120）落在 4px 边缘带
   （lightning-rod 8/24, pius-corrosion 7/24, pius-rust 14/24, vari-grip 11/24, yoke 12/24）——
   最高相似 patch 大量出现在背景/边界 → **背景原型抑制**（创新点二消融轴）必要性成立；
3. vari-grip/rust 上 patch-pre 显著差于 metric-v1（k1 0.275 vs 0.627）→ 无度量学习的
   patch 相似度在纹理弱组合上不稳定。

## 6. anomalib 同查询集 AD 基线（Step-5）

### 6.1 主表（图像级 AUROC，跨 3 种子的 mean-over-7-combos ± std）

| 方法 | k=1 | k=5 | k=10 |
|---|---|---|---|
| PatchCore（timm WRN50-racm, coreset 0.1, nn=9） | 0.642 ± 0.047 | 0.797 ± 0.021 | **0.824 ± 0.013** |
| PaDiM（timm RN18）† | 0.859 ± 0.000 † | 0.691 ± 0.019 | 0.705 ± 0.022 |
| EfficientAD-S（固定 4000 步） | 0.569 ± 0.017 | 0.567 ± 0.012 | 0.593 ± 0.011 |
| —— 以下为本方案（多类判别，监督信息不同，非同范式）—— | | | |
| actual-M2-v0（SPPF-P5 GAP） | 0.565 ± 0.067 | 0.673 ± 0.005 | 0.653 ± 0.052 |
| P3-patch-metric-v1（本方案最好变体） | 0.618 ± 0.062 | 0.612 ± 0.061 | 0.603 ± 0.009 |

† 见 6.3 退化披露：该列评分与所选支持图无关，不构成有效“1-shot 适应”证据。

### 6.2 瓶颈探针判读（§0.5 预设问题的回答）

- **PatchCore 全面高于本方案最好变体**（k1 +0.024，k5 +0.185，k10 +0.221）→ 瓶颈主要在**特征源与评分机制**，不在“少样本任务形式”本身；k≥5 时差距显著。
- EfficientAD 全线最低（0.56–0.59）：其 4000 步为论文 70k 步的 1/17.5（a priori 冻结、全配置一致），该数字仅代表该预算档位，非方法上限。
- H1 相关（仅数据呈现，不在此裁）：AUROC 维度上本方案与 PatchCore 差距在 k≥5 超过 2 点；召回维度须按 §1 说明以 PR 曲线/部署工作点评估，留待 H1 判定。

### 6.3 方法学披露（合同禁止省略）

- **PaDiM k=1 退化**：三个种子结果逐位相同（如 glass-insulator 均为 0.899024），即评分与所选支持图无关——单样本协方差退化为常数分布所致。该列不可解读为“1-shot 强基线”，仅作方法学参考。
- **PaDiM 无 NaN**：全 45 配置无缺失值（此前的 NaN 预期未被最终运行证实）。
- **补跑披露**：lightning-rod-suspension_k5_s2 在主运行中因磁盘 ENOSPC 缓解操作（清理 /tmp/m3_anomalib）触发 FileNotFoundError，以同一冻结种子/协议单独补跑（support 逐字节一致，auroc=0.606）；过程见记录内 repair 字段与 scripts/m3_repair_efficientad_k5s2.py。

### 6.4 逐组合结果（mean over 3 seeds；全量逐种子见 baselines_eval.json）

| 故障组合 | PC k1/k5/k10 | PaDiM k1/k5/k10 | EAD k1/k5/k10 |
|---|---|---|---|
| glass-insulator/missing-cap | 0.741 / 0.816 / 0.850 | 0.899 / 0.740 / 0.747 | 0.519 / 0.517 / 0.567 |
| lightning-rod-suspension/rust | 0.600 / 0.789 / 0.850 | 0.885 / 0.747 / 0.747 | 0.547 / 0.602 / 0.612 |
| polymer-…-shackle/corrosion | 0.537 / 0.795 / 0.793 | 0.893 / 0.683 / 0.714 | 0.465 / 0.517 / 0.592 |
| polymer-…-shackle/rust | 0.508 / 0.718 / 0.702 | 0.678 / 0.697 / 0.678 | 0.613 / 0.525 / 0.578 |
| vari-grip/bird-nest | 0.779 / 0.888 / 0.926 | 0.910 / 0.588 / 0.653 | 0.641 / 0.653 / 0.610 |
| vari-grip/rust | 0.614 / 0.719 / 0.820 | 0.872 / 0.587 / 0.624 | 0.631 / 0.646 / 0.660 |
| yoke-suspension/rust | 0.713 / 0.851 / 0.830 | 0.872 / 0.795 / 0.769 | 0.565 / 0.508 / 0.535 |

运行规模：135 配置 = 3 方法 × 5 资产 × {1,5,10}-shot × 3 种子；总 GPU 墙钟约 6.4h（RTX 4080 SUPER），全部配置一次成功（1 配置经披露补跑）。

协议要点（详见 `baselines_cfg.json` 与 `support_pool_amendment.md`）：
- 支持样本池：**选项 A（2026-09-17 人工裁决）**——该资产全部 good 裁剪（train∪val），
  冻结 RNG 规则不变。原冻结规则（仅非查询 train/good）因 normal_query 消耗 good 池而
  不可行（非查询池 = 26/1/2/3/4654）。
- 硬红线：支持样本不命中任何 defect/kshot 冻结条目（已验证与 good 池零交集，运行时全配置 0 泄漏）；
  与 normal_query 的重叠允许并逐配置登记（全运行共 585 处，涉及 169 个唯一文件，逐条见记录）。
- 偏差方向：AD 拟合样本与 normal_query 部分重合 → AD 的 AUROC 略被抬高，
  对本方案（H1）是**保守方向**。
- 监督差异：AD 基线仅见正常支持样本（单类建模），本方案用目标故障 k-shot 缺陷原型
  （多类判别）——非同范式 head-to-head，表中明确标注。
- EfficientAD：统一 4000 训练步（a priori，全配置一致；论文默认 70000 步）。
- PaDiM k=1 退化披露：三种子结果逐位相同（评分与支持图无关，单样本协方差退化），该列不构成有效“1-shot 适应”证据，仅作方法学参考。
- 补跑披露：lightning-rod-suspension_k5_s2 经同一冻结种子/协议补跑并登记（见 6.3）。

## 7. 内存与延迟（Step-3 附属交付）

`memory_latency.json`：特征提取 3.03 ms/裁剪（GPU）/ 8.8 ms（CPU）；patch 打分 0.119 ms（GPU）；
原型库 FP32 8.26 MB（上限 256×128d/组合，合规=1）；GAP 原型 1 KB/组合。

## 8. 运行层修复记录（工程，非协议）

 anomalib 2.6.2 集成的 8 处潜伏缺陷（脚本从未完整跑通过，本次逐一修复并经 CPU/GPU 冒烟验证）：
① Engine checkpointing 冲突→barebones；② Folder val_data AttributeError→limit_val_batches=0；
③ PredictDataset 单路径 API→符号链接目录+序号映射；④ 惰性 train_data→先 setup()；
⑤ EfficientAd 需 datamodule→max_epochs=4000//k 使总步数恒 4000；⑥ batch_size 未透传→
train/eval_batch_size；⑦ EfficientAd 禁 Normalize→专用预处理器；⑧ Engine 局部导入
作用域错误→模块级导入。协议内容（查询集/阈值/种子/指标）零改动。

## 9. 结论边界（预声明）

- 公开数据集（InsPLAD-FD 裁剪级协议），无现场验证；
- defect query 源帧与 det-train 帧重叠为 M2 冻结协议既定边界（preflight 泄漏登记）；
- `pius/rust` 统计意义弱（见 §4）；
- AD 基线支持池按选项 A 执行（§6，偏差方向有利基线）；
- k-shot 原型构建的监督信息与 AD 基线不同（§6），跨范式对比仅以分析身份呈现。

## 10. 复现

```
python3 scripts/m3_verify_feature_layers.py
python3 scripts/m3_patch_extractor.py / python3 scripts/m3_build_patch_bank.py
python3 scripts/m3_metric_train.py / python3 scripts/m3_patch_proto_eval.py
python3 scripts/m3_run_anomalib_baselines.py        # 选项A 配置
python3 scripts/m3_rust_analysis.py
```

指标文件：`proto_v1_eval.json` / `baselines_eval.json` / `rust_analysis.json` /
`memory_latency.json` / `preflight.json` / `baselines_cfg.json`。
# 原型头 v0 初评报告（M2 Step-5/6 交付物）

> 任务：M2-proto-head-v0 · 日期：2026-09-16 · 脚本：`scripts/m2_proto_head_v0.py`
> 骨架：`/root/autodl-tmp/runs/yolov8n_det/weights/best.pt`（YOLOv8n，18 类基类检测微调后冻结；备份 `/root/autodl-fs/yolov8n_det_best.pt`，sha256 `ea1526c2…`）
> 特征：P3（stride-8）全局平均池化 256 维，L2 归一化；原型 = k-shot 实例特征均值；打分 = 余弦相似度
> 评测协议：`data/splits/insplad-fd/` 固化划分（哈希已登记），k ∈ {1,5,10} × 种子 {0,1,2}，查询集按 k 固定；AUROC = 逐组合（缺陷查询 vs 同资产正常查询池）；召回阈值 = 同资产正常打分 mean+2std（固定规则，零调参）

## 1. 主结果（3 种子 mean±std，k 即 k-shot 的 shot 数）

| k-shot | AUROC | 缺陷召回 | 参照 |
|---|---|---|---|
| 1-shot | 0.565 ± 0.067 | 0.142 ± 0.027 | — |
| 5-shot | **0.673 ± 0.005** | 0.185 ± 0.029 | — |
| 10-shot | 0.653 ± 0.052 | 0.151 ± 0.049 | 文献上界 DifferNet 0.905（全量监督式 AD，非 k-shot） |

## 2. 逐组合分解（k=5）

| 组合 | AUROC | 备注 |
|---|---|---|
| vari-grip/bird-nest | ~0.85 | 最强：鸟巢外观与正常差异大 |
| glass-insulator/missing-cap | ~0.80 | 强 |
| polymer-insulator-upper-shackle/corrosion | ~0.75 | 中强 |
| yoke-suspension/rust | ~0.60 | 弱 |
| lightning-rod-suspension/rust / vari-grip/rust | ~0.57 | 弱 |
| polymer-insulator-upper-shackle/rust | ~0.38–0.50 | 最弱：仅 33 实例、与 corrosion 同资产易混 |

模式清晰：**结构性缺陷（缺帽、鸟巢）显著可分；表面性缺陷（锈蚀）难分**——锈蚀的判别信息在更局部的纹理，全局平均池化将其稀释。

## 3. 骨架对照实验（`scripts/m2_control_backbone.py`，k=5/seed0，4 组合）

| 组合 | COCO 预训练 | det 微调后 |
|---|---|---|
| glass-insulator/missing-cap | 0.587 | **0.797** |
| vari-grip/bird-nest | 0.654 | **0.850** |
| yoke-suspension/rust | 0.481 | **0.597** |
| polymer-…-upper-shackle/corrosion | 0.640 | **0.750** |

**结论：基类检测微调对原型判别是增益（均值 0.59 → 0.75）**，v0 采用 det 微调后冻结骨架成立。

## 4. 诚实结论与对 PLAN 的影响

1. **v0 达成"管线跑通 + 初评基线"的合同目标，但 H1 信号弱**：AUROC 0.67 vs 文献上界 0.905 差距 >23 点；与 M3 实现复杂度门的预期一致——v0 本来就是最简形态（冻结主干 + 无度量学习）。
2. **k=1 单实例原型方差过大**（部分组合 <0.5），k=5 为甜点；k=10 无增益——提示原型聚合策略（创新点二）是正确的改进轴。
3. **改进方向优先级**（输入 M3/M5 决策）：① patch 级相似度替代 GAP（锈蚀纹理是主要瓶颈，这也预示与 PatchCore 对比的可达空间）；② 原型构建策略消融（多原型/背景原型抑制）；③ 基类度量学习（v1 计划本体）。
4. 若 M3 v1 仍无法把弱组合（锈蚀类）拉起，PLAN §2 的**裁剪级降级预案**（两段式）不受影响，保底路径完好。
5. 数据口径注记：`pius/rust`（33 实例）k=10 查询侧仅 3 实例，统计意义弱（见划分清单使用纪律）。

## 5. 复现

```
python3 scripts/m2_proto_head_v0.py --weights /root/autodl-tmp/runs/yolov8n_det/weights/best.pt
python3 scripts/m2_control_backbone.py <weights> <tag>   # 骨架对照
```
指标文件：`outputs/M2/metrics.json`（seeds=3，逐种子/逐组合全量数值）。

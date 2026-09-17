# InsPLAD 评测协议核查（M1 Step-2）

> 证据来源：InsPLAD 论文全文（arXiv 2311.01619，*International Journal of Remote Sensing* 2024，ar5iv HTML 全文抓取）+ 本地数据包结构核验（`/root/autodl-tmp/data/insplad`）。
> 核查日期：2026-09-15。

## 1. 数据集组成（论文口径 vs 本地核验）

| 子集 | 论文口径 | 本地核验 |
|---|---|---|
| InsPLAD-det | 10,607 张 Full HD UAV 图，17 资产类，28,933 实例框；按输电杆塔分组 80/20 划分 | COCO 标注：train 7,981 图 / 22,635 框；val 2,626 图 / 6,324 框；18 类（含 `sphere`，见 §5 差异） |
| InsPLAD-fault（监督故障分类，本地 `defect_supervised/`） | 5 资产 × 正常/缺陷，6 种缺陷共 402 个缺陷样本（裁剪图） | 5 资产 × train/val × {good, 故障}，故障裁剪图共 2,972 张（详见 stats CSV）；**显著多于论文的 402**（差异分析见 §5） |
| InsPLAD-anomaly（无监督 AD，本地 `unsupervised_anomaly_detection.zip`） | 训练集仅正常样本；缺陷只出现在测试集 | 11 资产目录 × `train/good` + `test/good`(+部分资产 `test/<fault>`)；实测含 rust / missingcap / torned-up / nest / peeling-paint |

论文明确的 5 个故障资产与缺陷：Yoke Suspension(rusty)、Glass Insulator(missing cap)、Polymer Insulator Upper Shackle(rusty/corrosion)、Lightning Rod Suspension(rusty)、Vari-grip(bird's nest, 也可 rusty)。六种缺陷 = 4×腐蚀类、1×部件破损、1×鸟巢。

## 2. 部件级 or 图像级？（协议定位）

**两者皆有，但任务不同**：
- 所有标注粒度：检测框（det）+ 图像级标签（fault/anomaly）；
- **故障分类与 AD 均在部件裁剪图（component crops）上进行**——即评测对象是裁剪后的部件图，标签是图像级（该 crop 正常/某缺陷）；
- 论文原文："three image-level computer vision tasks ... object detection (AP); defect classification (Balanced Accuracy); anomaly detection (AUROC)"，其中 FC/AD 的输入均为 cropped power line component images。

**对本课题的含义**：PLAN v1.1 §5.1 的"统一部件裁剪管线 → 裁剪级故障分类/定位"与 InsPLAD-FD 协议**天然对齐**；裁剪级 AUROC/召回可与文献直接对齐。

## 3. 指标定义

| 任务 | 指标 | 说明 |
|---|---|---|
| 目标检测 | AP（COCO 风格） | 17 类资产检测 |
| 监督故障分类 | **Balanced Accuracy** | 训练集缺陷样本经 Albumentations ×10 增广以平衡 |
| 无监督 AD | **AUROC**（逐资产 + 平均） | 训练仅正常样本；报告模型权重大小与吞吐 |

AD 基线结果（论文 Table 10）：DifferNet 平均 AUROC 0.905（最佳），CS-Flow 0.903（5 资产中 3 项最佳）；基于 CNN 特征 + 归一化流的方法显著优于其他范式。

## 4. k-shot 划分惯例

**论文与已发表基准均未定义 k-shot 划分**——InsPLAD-fault/anomaly 的官方协议是全量监督/全量无监督，不存在 1/5/10-shot 惯例。FiLo/AnomalyCLIP 等少样本 AD 工作也未发布 InsPLAD k-shot 切分。

**结论**：k-shot 划分是本课题**必须自行定义并固化的协议组件**（PLAN M2 "k-shot 划分固化 ×3 种子"），无现成惯例可循。设计时需：缺陷 crop 池中按类随机抽 k 个实例构建原型（种子固定）、其余作查询；因无惯例，需在论文中明确声明划分生成方式并公开划分文件。构建 k-shot 池时建议沿用官方 det 划分的杆塔分组思想，避免同塔近重复图同时出现在原型与查询集。

## 5. 数据差异与审计注记

1. **缺陷样本数差异**：论文称 402 缺陷样本；本地 `defect_supervised` 实测故障类 crop 共 2,972 张（train 2,670 + val 302，含增广可能性待查，×10 增广应对应 4,020）。402 ≈ 增广前原始缺陷裁剪数（论文口径），发布包可能包含部分增广产物或不同版本。**使用时以本地实测为准，引用论文数字时注明口径。**
2. **类名不一致**：`polymer-insulator-upper-shackle` 的 train 目录为葡语 `corrosão`/`normal`，val 为 `good`/`rust`；论文称该资产缺陷为 rusty。统计表已归一化（corrosão→corrosion, normal→good）。
3. **类别数**：本地 COCO 标注 18 类（含 `sphere`，train 26 框），论文口径 17 类。
4. **yoke-suspension val/good = 5,742** 张，与 train/good = 299 严重不对称（其余资产 train≫val），使用 val 时需注意该资产正常的验证池过大。
5. AD 子集中 `damper-preformed` 与 `spacer` 的 test 目录仅含 `good`（无缺陷测试样本），AD 评测实际覆盖 5 个含缺陷资产（与论文 Table 10 一致）。

## 6. 对 PLAN v1.1 §3.2 的协议影响

- 主对比协议（少样本 AD 族互比）按**裁剪级**对齐，指标 AUROC + 缺陷类召回（PLAN §5.3 召回优先），Balanced Accuracy 可作辅助；
- PatchCore/PaDiM/EfficientAD（anomalib 复现）在裁剪级与本方案可比，图像级 AUROC 仅作文献对齐参考；
- 端到端框输出能力单列（AD 基线无此能力，PLAN §5.1 已定）。

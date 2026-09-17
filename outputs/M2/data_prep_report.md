# M2 数据准备报告（Step-2 交付物）

> 任务：M2-proto-head-v0 · 日期：2026-09-15 · 脚本：`scripts/m2_coco2yolo.py`
> 输入：`/root/autodl-tmp/data/insplad/annotations/instances_{train,val}.json` + 数据包图像目录
> 输出：`/root/autodl-tmp/data/yolo_det/`（ultralytics 格式，图像用符号链接不复制）

## 1. COCO → YOLO 转换结果

| split | COCO 图像条目 | 唯一文件 | 符号链接 | 有效框 | 空标注(背景图) |
|---|---|---|---|---|---|
| train | 7,981 | 7,935 | 7,935 | 22,635 | 2 |
| val | 2,626 | 2,626 | 2,626 | 6,324 | 0 |

**框数对账**：转换后 YOLO label 总框数 = COCO 标注总数（train 22,635 ✓ / val 6,324 ✓），零丢失、零跳过。

## 2. 数据包布局问题与清洗决策（差异清单）

### 2.1 COCO train 标注含 46 组重复图像条目
- 现象：7,981 条图像条目 → 仅 7,935 个唯一文件名；重复条目使用不同 image_id，且**部分标注挂在重复 id 上**（例：`331-2_DJI_0190.jpg` 两个 id，合计 14 框分挂两侧）。
- 风险：按 image_id 写 label 会导致同名文件两次写盘、后写覆盖先写 → 丢框。已实测发生过并在修复后对账清零（§1）。
- **决策**：按文件名聚合全部框（等价于把重复条目合并）。训练语义不受影响（同一图的所有框都在）。未去重是**有意为之**：删除条目会改变有效框的统计口径，留给评测协议文档说明。
- 影响：基类训练的有效图像数为 7,935（非 7,981）；引用训练集规模时用"7,935 图 / 22,635 框"。

### 2.2 46 张训练图像不在 `train/` 目录（散落数据包其他位置）
- 现象：`train/` 目录仅 7,935 张，与唯一文件名数一致；标注里的图像若不在对应 split 目录，脚本按文件名全库搜索补齐。
- 含义：散落图像均在数据包内被找回（linked=7,981 无 missing），未损失。此现象与 §2.1 重复条目同源——发布包打包不一致，记入论文数据集描述的脚注候选。

### 2.3 bbox 越界处理
- 规则：负坐标裁剪至图像边界（COCO bbox 允许越界；YOLO 归一化坐标要求 [0,1]）。本批次 0 条越界框需要裁剪（skipped=0），规则作为防御性处理保留。

### 2.4 背景图
- train 有 2 张无标注图（生成空 label 文件，YOLO 训练惯例保留作负样本）；val 无。

## 3. FD 类名归一化（承 M1 审计）

| 原始目录名 | 归一化 | 位置 |
|---|---|---|
| `corrosão`（葡语"腐蚀"） | corrosion | polymer-insulator-upper-shackle/train |
| `normal` | good | polymer-insulator-upper-shackle/train |
| 其余（good/rust/missing-cap/bird-nest） | 不变 | 全部资产 |

归一化映射实现于 `scripts/m1_fd_stats.py`（LABEL_NORMALIZE）；M2 后续所有 FD 相关脚本必须复用同一映射（单一事实来源），不另写映射表。

## 4. yoke-suspension val/good=5,742 异常池决策
- M1 发现该池与 train(299) 严重不对称。**M2 决策：基类检测训练不受影响**（它属 FD 裁剪池，不进 YOLO 训练集）；k-shot 划分（Step-4）不使用 val/good 作原型池（原型只从故障类抽），查询侧如需正常池将按资产分层下采样至与故障查询量同量级——该决策随划分清单固化。

## 5. 类别集说明
- 18 类全部保留（含 train 仅 26 框的 `sphere`、47 框的 `polymer insulator tower shackle`）——基类检测器训练不做类删减， rare 类的可用性在 M2 报告中以逐类 AP 呈现，为 M3+ 的"基类/故障类划分"提供依据（PLAN §8 作业1：基类/故障类划分由 M1+M2 数据共同决定）。

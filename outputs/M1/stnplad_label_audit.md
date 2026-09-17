# STN-PLAD 标注类别审计（M1 Step-3）

> 核查日期：2026-09-15（同日回填本地核验）。
> 证据来源：datasetninja.com/stn-plad 页面元数据、GitHub PLAD 仓库 README 与 API，**以及本地文件实测**（`/root/autodl-tmp/data/stn-plad/`，原始 plad.zip 解包：`annotations.json` + `20181127-A1/` 108 图 + `20181129-R/` 25 图）。

## 1. 数据集概况（已核实：官方页面与仓库一致）

- 论文：*STN PLAD: A Dataset for Multi-Size Power Line Assets Detection in High-Resolution UAV Images*
- 规模：**847 张高分辨率 UAV 图像，2,409 个标注框**
- 任务：目标检测（无异常检测/故障检测基准）
- 许可：GPL-3.0（GitHub license 字段 + datasetninja 元数据一致）
- 下载源（供人工下载参考，优先级序）：
  1. datasetninja 直链（supervisely assets tar，DatasetNinja/Supervisely 格式，含 meta.json 类别定义）
  2. GitHub Release：`https://github.com/andreluizbvs/PLAD/releases/download/1.0/plad.zip`
  3. Google Drive：`https://drive.google.com/file/d/1KsNziErZ5ZRuWBpwUS5nlTnb8CcB2uQp/view`（可用 gdown）
  4. 附加标签：`https://github.com/andreluizbvs/PLAD/files/8952243/labels.zip`

## 2. 有无故障/异常类？

**无。** 5 个类别全部为正常资产类别（目标检测标注，无缺陷/异常/状态标签）：

1. transmission tower（输电杆塔）
2. insulator（绝缘子）
3. spacer（间隔棒）
4. tower plate（杆塔标识牌）
5. Stockbridge damper（防振锤）

**对本课题的含义**（PLAN v1.1 §2 跨域章"原型扩展式"）：
- 跨域实验设计 = 冻结主干 + 用 STN-PLAD 的部件裁剪构建**新域正常类原型**入库测试——**只需要正常类样本，不需要故障标注**，故"无故障类"**不构成阻断**；
- 但跨域只能测"原型扩展后正常类检出/域适应"，无法测"新域故障召回"——论文写作时须明确该结论边界（与 PLAN §5.3 结论边界预声明一致）；
- STN-PLAD 类别与 InsPLAD 资产类别语义重叠（insulator ↔ glass/polymer insulator、damper ↔ stockbridge damper、spacer ↔ spacer、tower plate ↔ tower id plate、tower ↔ 无对应），可做原型库的类级对齐映射。

## 3. 资产类别数与图像属性（全部已本地核验）

| 项目 | 值 | 状态 |
|---|---|---|
| 资产类别数 | 5（COCO category id 0,1,2,3,5；**id 4 缺位**——定义了 5 类但编号跳过 4） | 已核实 |
| 图像数 / 标注框数 | 133 / 2,409（本地实测；两个采集批次 108+25，**全部图像均有标注**） | 已核实 |
| 故障/异常类 | 无（tower/insulator/spacer/damper/plate 全为正常资产） | 已核实 |
| 图像尺寸 | 全部 5472px 宽：108 张 5472×3078，25 张 5472×3648（高分辨率，切片推理必要性成立） | 已核实 |
| 每类框数 | damper 1,505 · insulator 312 · tower 253 · spacer 253 · plate 86 | 已核实 |
| 官方划分 | 原始版无 train/val 划分（datasetninja 版才有）——M7 需自行划分并防同批次泄漏 | 已核实 |
| 许可（标注文件内声明） | GPLv3，Copyright 2021 UFPE, In Forma and STN. | 已核实 |

注：与 datasetninja 页面的"847 images"不符——847 为 DatasetNinja 重打包版（含增强切片/多版本）口径；原始版 133 图与论文一致。跨域实验以**原始版**为准（133 图 / 2,409 框）。

## 4. 可用性判定（条件3 证据）

- ✅ 公开可获取（本地文件已就位，133 图全部有标注，标注与图像一一对应核验通过）
- ✅ 类别结构与"原型扩展式跨域"实验需求匹配（只需正常类部件裁剪；damper 1,505 框样本充足，plate 86 框偏少需在 M7 注意）
- ✅ 许可 GPL-3.0 允许研究使用（见 licenses.md）
- ✅ 本地核验待办已全部回填，无遗留阻塞项
- 提醒：原始版无官方划分，M7 跨域实验需自行划分（按采集批次/杆塔分组防泄漏）

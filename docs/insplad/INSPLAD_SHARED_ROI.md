# InsPLAD 小图共享 P3：A/B/C

只准备脚本和 CPU 预检；GPU 实验由用户启动。不训练，不用 CPLID。

固定 10N / ref3001，沿用完整 COCO 基线的五资产、七个资产—故障组合和全部查询。
不是五组参照，不增加 P2/P3 融合。每资产只建三份库，两个故障共用正常库。

| 臂 | 输入/取特征 | 正常库/分数 |
|---|---|---|
| A | 原小图 RGB Resize320，COCO 原生 P3；用已验证冻结前缀重算，并逐图对齐历史分数 | 官方 KCenter .1 seed7001；L2特征、欧氏1NN、top8 |
| B | 冻结检测模型同次前向 P3，全图框 RoIAlign | 自己同流程正常库；全图采样为恒等对照 |
| C | 同次前向 P3，已知资产类别的最高置信度检测框 RoIAlign | 支持/查询均同流程；无框回退全图，不删图 |

输入三臂均为 PIL RGB `Resize((320,320))→ToTensor`，不走 predict 的隐式 LetterBox/fuse/FP16。
检测阈值 conf=.25、NMS IoU=.7、class-aware、max_det=300，不根据测试结果改阈值。
RoIAlign：40×40、spatial_scale=1/8、aligned=True、sampling_ratio=1，之后逐位置通道 L2。
每批 hook 实际验证 P3 只执行一次。B 每批验证全图采样与原特征精确相等，因此缓存别名 A，避免重复占盘。

类别来自数据集资产，是已知类别条件诊断；不能声称对未知类别全自动分流。
没有框 GT，因此不报这批小图的检测 mAP/定位准确率，也不把 B 叫理想 GT 框。
320 输入和原检测训练1280不同；预裁剪图也与原航拍大图不同。这些限制需要保留。
C 的逐图预测框、原图坐标、置信度、回退标记与支持框会保存。

## 启动

```bash
cd /root/auto-exp
nohup bash scripts/insplad_shared_roi_launch.sh > /root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1.log 2>&1 &
tail -f /root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1.log
```

同指令可恢复已校验特征/库缓存；评分会重算而不是信任旧分数。不要在中途改 batch（默认16）。
无需重训或下载权重；默认显式使用 `/root/miniconda3/bin/python`。
特征缓存约4.1GiB，准备阶段检查空间并留1GiB余量；不自动删除现有文件。

## CPU 验证（不启动 GPU）

```bash
OMP_NUM_THREADS=4 /root/miniconda3/bin/python tests/test_insplad_shared_roi.py
OMP_NUM_THREADS=4 /root/miniconda3/bin/python scripts/insplad_shared_roi.py --mode prepare --cpu-check
```

正式运行首先实际执行 GPU batch1/批量特征与框一致性、重复推理、原始COCO前缀检查。
所有库检查评分批量一致性、float64独立距离；代表 C 库重复建库；A逐图匹配历史分数。
门限 atol=rtol=1e-4，失败即停止，不自动放宽。

结果在 `/root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1/`：
`PERFORMANCE_TABLE.md`、`performance_comparison.csv`、`raw_scores/`、`metrics.json`、
`DETECTION_SUMMARY.json`（支持/正常/各故障回退率）、`detections/`、
`FROZEN_INPUTS.json`、`REPRO_AND_CORRECTNESS.json`、`BANK_MANIFEST.json`、
`CACHE_MANIFEST.json`、`RESOURCE_COST.json`。
单参照结果不声称统计显著或参照组稳定性；无框回退很高时不得把接近 A 归功于成功定位。

# InsPLAD 共享骨干 P3：A/B/C 正式评测报告

状态：正式评测已完成，原任务验收 **6/6 PASS**；20张可视化验收 **7/7 PASS**。
本次同步独立从21份逐图原始分数重算 AUROC，核对全部查询路径、标签和分数绑定；未重跑模型或改变实验结果。

## 范围与方法

数据为 InsPLAD `defect_supervised` 原有小图，5资产、7个资产—故障组合。
支持固定为 **10N、ref3001**，每资产10张正常支持，共50张；不解释为50个独立拍摄源帧。
共5215张唯一查询，连同支持共5265张输入。相同资产的两个故障共用正常查询和正常库，不能当作独立数据集。

| 臂 | 特征提取与定位 |
|---|---|
| A | 原 COCO P3＋top8；使用已验证完全相同的冻结前缀重算，逐图匹配历史分数 |
| B | 完整检测前向的同次 P3，全图边界作为 ROI；RoIAlign 恒等控制，不是 GT 框上限 |
| C | 同次 P3，按已知资产类别的最高置信度预测框取 ROI；没有有效框则回退全图 |

全部使用 PIL RGB `Resize((320,320))→ToTensor`，不做 LetterBox。
P3 为第4层、64通道；RoIAlign 输出40×40，`spatial_scale=1/8, aligned=True, sampling_ratio=1`，之后通道L2归一化。
检测固定 `conf=.25, NMS IoU=.7, max_det=300`；预测框选择不接触异常分数或故障标签。
每臂支持图和查询均走对应的同一特征流程，重新建立正常库。
正常库使用官方 KCenterGreedy、ratio=.1、seed7001，每库16000候选patch选1600；欧氏精确1NN距离，top8均值作为图像分数。
使用FP32并禁用TF32，无训练、无Adapter、无P2融合。

## AUROC 结果

| 资产／故障 | A：原COCO＋top8 | B：全图ROI | C：预测框ROI | C−A（百分点） |
|---|---:|---:|---:|---:|
| glass-insulator／missing-cap | 0.9026 | 0.9026 | 0.9129 | +1.03 |
| lightning-rod-suspension／rust | 0.9755 | 0.9755 | 0.9717 | −0.37 |
| polymer-insulator-upper-shackle／corrosion | 0.8923 | 0.8923 | 0.8973 | +0.50 |
| polymer-insulator-upper-shackle／rust | 0.8663 | 0.8663 | 0.8635 | −0.27 |
| vari-grip／bird-nest | 0.9256 | 0.9256 | 0.9139 | −1.17 |
| vari-grip／rust | 0.8738 | 0.8738 | 0.8911 | +1.73 |
| yoke-suspension／rust | 0.9541 | 0.9541 | 0.9388 | −1.52 |
| **七组合宏平均** | **0.912885** | **0.912885** | **0.912764** | **−0.0121** |

C相对A：3个组合上升、4个下降。B与A的逐图分数完全一致，恒等控制通过。
本组小图上平均AUROC基本保持，但不能据此声称所有故障无损，也没有统计显著性或跨参照组稳定性结论。

## 无框回退是结果的一部分

50张正常支持均有目标类别预测框，支持回退为0。
正常查询的回退率按五资产依次为1.0%、0.4%、2.0%、0.2%、0.6%。

| 异常查询组合 | 全图回退／总数 | 回退率 |
|---|---:|---:|
| glass-insulator／missing-cap | 223/705 | 31.63% |
| lightning-rod-suspension／rust | 64/315 | 20.32% |
| polymer-insulator-upper-shackle／corrosion | 92/725 | 12.69% |
| polymer-insulator-upper-shackle／rust | 0/22 | 0.00% |
| vari-grip／bird-nest | 31/358 | 8.66% |
| vari-grip／rust | 44/295 | 14.92% |
| yoke-suspension／rust | 3/295 | 1.02% |

这些是“没有有效指定类别框”的比例，不是基于框GT的漏检率。没有删除任何无框支持或查询。
高回退率意味着C是预测ROI与全图回退的混合系统，不能把接近A的平均精度全部归因于成功定位。

## 正确性与资源

- COCO与冻结检测模型的前五层参数及BN状态逐项相同；代表输入归一化P3差异0。
- 每批检查同次完整检测前向只计算一次P3；B全图ROI等于原生P3。
- GPU单张／批量C特征最大差异低于4.2e-6；代表C库重复建库差异0。
- A重算分数与历史逐图分数最大差异约2.4e-7；独立float64距离检查通过。
- 全部模型状态前后不变；全部21份逐图分数的AUROC独立复算一致。
- 实际评测调用墙钟时间97.74秒，含正确性门、CPU/I/O及建库评分，不是纯GPU内核时间或部署端到端延时。
- 分资产记录的最大GPU已分配显存198785024字节，约0.185GiB；主机峰值RSS约2.17GiB。
- 缓存4324669062字节，约4.03GiB。成本不构成与原方法的严格速度对照。

原始数值见 [metrics.json](metrics.json)、[正确性门](REPRO_AND_CORRECTNESS.json)、[资源成本](RESOURCE_COST.json) 和 [回退统计](DETECTION_SUMMARY.json)。

## 可视化与结论边界

20张可视化：5正常、15异常，其中5张特意选择无框回退；按固定路径哈希分层选样，不按异常分数挑图。
全部20张距离图的top8分数与正式结果最大差异1.1921e-7；只在CPU上复用C特征和正常库，没有重新前向骨干。
详见 [可视化说明](../InsPLAD-sharedROI-C-vis20-v1/README.md) 和 [样例清单](../InsPLAD-sharedROI-C-vis20-v1/samples.csv)。

限制：

- 资产类别由数据集提供用于候选框和库路由，是已知类别条件诊断，不是未知类别全自动识别。
- 小图只有图像级故障标签，无框或像素GT；不报告本批小图的检测mAP、定位准确率或像素精度。
- 预裁剪小图不等于原始1920×1080航拍场景；检测输入320也不同于训练1280，存在输入域与尺度差异。
- 单参照组不能证明稳健性；七组合宏平均不能掩盖个别故障下降或异常图较高回退。
- 热力图是到正常库的距离响应，不是注意力、缺陷概率或人工标注。

本轮支持的结论是：**在固定小图与10N/ref3001条件下，共享同次P3并使用预测ROI＋全图回退，保留了接近原COCO＋top8的宏平均AUROC；真实大图与纯检测框条件下的效果尚未得到证明。**

## 代码与结果存放

- 运行：`bash scripts/insplad_shared_roi_launch.sh`；完整配置见 `scripts/INSPLAD_SHARED_ROI.md`。
- 可视化：`python scripts/insplad_shared_roi_visualize.py`。
- 原始结果：`/root/autodl-tmp/version10-12/InsPLAD-sharedROI-ABC-ref3001-v1/`。
- PDF、20张PNG、ZIP和距离矩阵位于上述结果目录的 `vis_C_20/`，未提交Git。
- 本仓库保留代码、报告、汇总指标、正确性与来源哈希；原始图片、模型权重和大缓存不入库。
- [PROVENANCE.json](PROVENANCE.json) 记录权重/代码/源结果/21份逐图分数哈希及支持路径；[SYNC_VERIFICATION.json](SYNC_VERIFICATION.json) 记录上传前独立核验结果。

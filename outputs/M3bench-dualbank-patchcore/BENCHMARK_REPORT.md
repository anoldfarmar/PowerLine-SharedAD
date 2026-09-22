# M3BENCH-dualbank-patchcore-v1 基准报告

1N1A/5N5A 双库方法 (Ours) 对标准 PatchCore 的探索性直接比较。全部结果**探索性**: 查询集复用、历史 k-crop 为支持预算而非独立源帧数、骨干训练源帧身份未知 (记未知, 不记零重叠)。本轮非独立确认性测试, 不单独宣布 H1 达标。

## 1. 冻结与执行完整性
- 支持排列: 每 10 张正常参照按固定规则排序后, NumPy PCG64(seed=组号) 一次排列; N1=第1张, N5=前5张, N10=全部10张; 全名单与 SHA256 见 SUPPORT_LISTS.json / FROZEN_INPUTS.json
- 组一: 两方法同 N1, 同 query_k1; 组二: Ours 用 N5, PatchCore 用含该 N5 的 N10, 同 query_k5; 两方法查询路径集合与标签逐配置断言一致
- 组一与组二查询集不同 (历史划分如此), 已显式披露; 组间差异不解释为纯支持数量效应
- 故障支持: 组一历史 k1、组二历史 k5, 种子 0/1/2; PatchCore 不依赖故障种子, 同查询复用不计独立实验
- Ours: M3DIAG6/7 冻结实现 (YOLO P3 + mapper_B + L2 + 库上限256 + P1 前50% + FPS seed0 + 同 patch 差值 top8 均值); 故障库由**当前预算**正常库筛选, 未复用 10 张参照库
- PatchCore: anomalib 2.6.2 官方 Engine 链路, ImageItem + ImageBatch.collate; effective config 与权重哈希 (wide_resnet50_2-95faca4d.pth, sha256前16=95faca4d11227ddd, 132MB) 见 bench_eval.json `patchcore_effective_config`; Resize(320) 在数据集内以满足 collate, Normalize 保留在 PreProcessor; 拟合仅访问支持集, 未使用查询标签
- 复现锚点 (原历史预算, 单独标注不混入比较): 每资产 k5/seed0/ref3001 对 M3DIAG6 原始分数 max|diff| = 5.0e-09 (A/B/C 三分数逐路径), 容差 1e-4 未放宽; 证据于 0e28d04 会话产生并登记
- 正确性: 分数全有限; AUROC 平均秩处理同分 (小型人工分数验证沿用历史实现); Ours mapper/BN 前后哈希一致; PatchCore 实际库大小逐配置记录 (g1: 160×1536, g2: 1600×1536, float32)

## 2. 主结果 (图像级 AUROC, Ours − PatchCore 配对差)
| 单元 | 五参照组差均值 | 范围 | 正收益组数 |
|---|---|---|---|
| glass g1 (1N1A vs 1N) | −0.1153 | [−0.5355, +0.0207] | 2/5 |
| yoke g1 (1N1A vs 1N) | −0.1314 | [−0.3228, +0.0410] | 1/5 |
| glass g2 (5N5A vs 10N) | −0.0373 | [−0.0635, −0.0140] | 0/5 |
| yoke g2 (5N5A vs 10N) | −0.1840 | [−0.3138, −0.0599] | 0/5 |

- 宏平均仅作补充; 逐参照组逐种子明细见 bench_eval.json `paired` / `per_ref` 与 raw_scores_*/ 全部逐图分数
- PatchCore 单侧成绩: g1 glass 0.718–0.829, g1 yoke 0.652–0.723, g2 glass 0.811–0.852, g2 yoke 0.797–0.888
- Ours 显著种子方差示例: yoke g1 ref3001 三种子 0.398/0.620/0.759; glass g1 ref3002 三种子均 0.18 (该 1N 正常裁剪对其失败), 而同参照 PatchCore 0.718

## 3. 五问回答
1. **1N1A vs 1N**: glass 差距 −0.115 (2/5 参照组翻正, ref3002 崩溃 −0.536); yoke −0.131 (1/5 翻正)。同正常支持数、Ours 多拿异常监督, 仍整体落后。
2. **5N5A vs 10N**: glass −0.037 (小而一致), yoke −0.184 (全部参照组一致落后)。支持总数相同、监督组成不同 (Ours 5N+5A vs PC 10N)。
3. **是否保持**: 负差距在 g2 两单元对全部 5 参照组保持; g1 存在参照组依赖的翻转 (glass ref3003/3005、yoke ref3003 为正)。Ours 跨种子方差远大于 PatchCore; 差距由方法差异与正常参照覆盖度共同决定, 单参照组结论不可靠。
4. **收益/成本**: 当前数据下 Ours 无精度收益; 成本上 Ours 每裁剪 4.8–14.9ms vs PatchCore 762–878ms (实测速比约 55–164×), 库内存 Ours 512×64 float32 = 256KB (固定) vs PC g1 0.98MB / g2 9.8MB (随支持线性), 权重 Ours YOLO+mapper_B vs PC backbone 可学参数 24.86M+库。原生配置比较, 未人为压缩 PC 库。
5. **投入方向**: 在冻结实现与查询复用设定下, 同 patch 差值 top8 评分不敌标准邻近重加权, 且差距在监督组成变化 (g2) 下反而扩大; 证据相对指向**评分/特征机制审视**优先于部署验证; 但一切探索性, 机制结论需独立隔离重评 (未完成项)。

## 4. 服务器推理成本 (batch=1, ref3001, Ours seed0, 预热20/计时100, CUDA 事件)
| 方法 | 单元 | 中位数 ms | P95 ms |
|---|---|---|---|
| Ours | glass g1 | 14.85 | 15.04 |
| Ours | yoke g1 | 4.81 | 5.04 |
| Ours | glass g2 | 8.85 | 8.94 |
| Ours | yoke g2 | 4.78 | 4.95 |
| PatchCore | glass g1 | 815.02 | 882.55 |
| PatchCore | yoke g1 | 762.52 | 910.32 |
| PatchCore | glass g2 | 878.17 | 1064.65 |
| PatchCore | yoke g2 | 785.94 | 1013.01 |

- 主计时=已解码裁剪图到图像分数; 排除读盘/模型加载/建库; PatchCore 经 Engine.predict 包装路径计时 (含 dataloader 开销, 已披露); 建库耗时 g1 ≈1.4–1.7s, g2 ≈2.1–2.3s (排除在计时外)
- 批量一致性: PatchCore batch=4 vs 单图 max|Δscore| = 2.33e-03 (分数绝对量级 ~30, 相对差 ~1e-4)
- 峰值显存 (计时进程) 0.34 GB; 不外推 RK3588; 不将裁剪推理称为整图端到端

## 5. 明确未完成 / 边界
- 校准召回、误报率、完整 H1 检验未做 (不使用查询标签选阈值, 不以测试集阈值召回冒充校准)
- 无显著性检验 (共享查询配置不作独立测试集); 源帧级独立隔离重评未做
- 两组非相同 shot/监督条件; 查询集组间不同; k 为裁剪预算非独立源帧数
- 预算内完成: wall ≈ 55min (PatchCore 臂 ≈21min + 延迟 ≈8min + 冒烟/适配), 未超 6h 合同预算

## 6. 结论声明
在本轮冻结输入、冻结实现与探索性查询设定下, **标准 PatchCore 在全部四个单元的配对均值上优于当前双库方法**, 其中 g2 两单元全部参照组一致。该结果如实报告; 未修改 PatchCore 参数、未重跑挑结果、未放宽任何容差。验收依据为执行完整、结果正确与报告诚实。
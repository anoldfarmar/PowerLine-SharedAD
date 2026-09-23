# METHOD_SPEC — M3FEAT-yoke-v1 (首评前冻结)

本文件在首次新增配置查询评分之前提交。数值规则全部来自源码读取与运行时张量断言，不写死维度用于通用逻辑。

## 特征规则
- 公共输入: 320x320, 沿用 m3_common.make_transform(D.IMGSZ) 的当前 YOLO 预处理路径; 主干 ultralytics YOLOv8n (D.WEIGHTS) eval/no_grad, 只前向 model.model[:10] (与 M2/M3 一致), BN 不更新。
- E0 (历史): patches.npy (fp16, N x 40x40x64, P3 原始) -> mapper_B (head state) -> L2 -> 实际 128 维 (运行时从张量读取并断言)。数值路径与 M3DIAG/M3BENCH/M3CAP 完全一致 (M6.crop_sims 默认 chunk=128)。
- E1: patches.npy 同一 P3 特征, 不经 mapper, fp32 cast 后逐位置 L2 -> 实际 64 维 (断言)。
- E2: 一次前向取 feats[4]/[6]/[9] (审计断言 shape B,64,40,40 / B,128,20,20 / B,256,10,10 与模块类型, 记录 model.model[4]/[6]/[9] repr); P4/P5 bilinear align_corners=False 上采样到 40x40; 对齐后每层逐位置 L2; 三层等权 concat -> 整体 L2 -> 实际 448 维 (断言 64+128+256)。新缓存 FP32 (N,40,40,448) npy + 路径清单 JSON。
- 支持/查询路径去重后各提取一次; 查询标签不参与特征构建/建库/候选/参数。

## 检测头配置
- H-DUAL (每特征空间独立): N 库 = 该特征 5N 全 patch (8000) 上 coreset_farthest(cap=256, seed=M6.FPS_SEED, 历史贪心并列规则); F 侧 = 5A 全 patch 按到该特征 N256 的 1-max_cos 稳定升序排 (torch.argsort stable), 保留前 50% = 4000 候选, **库=全部候选不 FPS 不复制**; 保存 keep 索引/来源路径/哈希; 不沿用 E0 候选强加给 E1/E2。评分 sF=max_f cos, sN=max_n cos, patch=sF-sN, image=mean(top8(patch)); 先两独立 max 再减; E0 用 M6.crop_sims (历史数值路径), E1/E2 用同公式分块实现并与直接计算对照。
- H-PC-YOLO (每特征空间): anomalib 2.6.2 官方组件薄适配: fit = PatchcoreModel(train) 逐批 collect embedding -> vstack -> KCenterGreedy(embedding, sampling_ratio=0.1).sample_coreset() (官方 coreset, 随机性依 KCenterGreedy 内部约定); predict = PatchcoreModel(eval).forward -> nearest_neighbors(Euclidean, n_neighbors=1, 官方分块) -> compute_anomaly_score (官方 9 邻域重加权) -> InferenceBatch.pred_score (原始连续分数); anomaly_map 经官方 AnomalyMapGenerator(sigma=4)。fit 只访问 5N 特征。库 dtype FP32。
- 已披露: 三种公共特征均逐位置/整体 L2 归一化; PC 变体与双库共享输入特征但监督组成 (5N vs 5N5A)、原型数 (~800 vs 4000+256)、评分算法不同; 同特征仅指输入一致。

## 复现与数值门限 (全过才进全量)
- G1: E0+H-DUAL 15 配置 vs M3CAP RF 逐路径 max|diff| <= 1e-4 (路径集合/数量/标签映射一致)。
- G2: 特征形状/有限值/L2 范数 (~1)、batch 一致性 (同图两批 diff=0)、YOLO 与 mapper 状态哈希前后相同。
- G3: 双库分块 vs 直接计算一致, 覆盖 4000 大 F 库 (随机 3 crop, max diff <= 1e-5)。
- G4: PC 适配对照: 固定合成特征上 adapter 的 coreset 大小、最近邻距离与索引、pred_score 与官方组件直接调用一致 (并列允许等价邻居, 分数须一致; cdist 对照)。

## 预算与成本口径
- GPU 预算 1 小时: 探针先行 (E2 提取 + 一个 (ref,seed) 双库块 + 一个 PC fit/predict), 超预算先优化缓存/分块, 仍超即停并报告。
- 记录 allocated 口径显存; 缓存评分耗时不称完整推理延迟; 不宣称端到端速度或 RK3588。

## 验收项
- 见合同 acceptance: FEATURE_HEAD_REPORT.md 存在; arm_configs_completed >= 60; repro_E0_gate = 1; gates_all_passed = 1。执行前已写入, 不事后补。
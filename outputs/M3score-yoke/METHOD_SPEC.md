# METHOD_SPEC — M3SCORE-yoke-v1 (冻结于首次新查询评分前)

## 1. 数据与身份
- 查询: data/splits/query_k5.json defect_query[yoke-suspension/rust] + normal_query[yoke-suspension/good]; 正常/故障集合不相交, 数量在运行时断言并记录。
- 正常支持: outputs/M3bench-dualbank-patchcore/SUPPORT_LISTS.json normal_lists[yoke-suspension_ref{j}].N5, j=3001..3005。
- 故障支持: 同文件 fault_supports[yoke-suspension__rust_g2_k5_seed{s}], s=0,1,2。
- 冻结哈希见 FROZEN_INPUTS.json (权重/划分/支持名单/代码/缓存)。

## 2. E1 特征 (A1/A2)
- 提取节点: ultralytics YOLOv8n forward 第 4 层输出 (feats[4], 运行时形状断言 (B,64,40,40), 以实际形状为准)。
- 预处理: MC.make_transform(IMGSZ) 与历史一致; 图像 RGB; dtype float32。
- 归一化: NCHW 上 dim=1 逐位置 L2 (E2 修复后的正确语义); 输出 (N,40,40,64) fp32。
- 模型加载一次; 唯一路径去重后批量 (bs=32); eval+inference_mode; 提取前后 state_dict 哈希必须相同。
- 缓存: outputs 数据盘目录, CACHE_MANIFEST.json 记录路径顺序/权重哈希/节点/预处理/插值(无)/归一化/dtype/代码哈希/内容 SHA256/前后状态哈希。
- 数值门 (新协议容差, 2e-4): 同图 batch=1 vs batch=32 于支持/正常查询/故障查询抽样, 最大绝对差 <= 2e-4, 报告实际值; 超限查 TF32/精度, 不放宽。
- 历史 M3FEAT G2_features 直接写 True 属偏离, 在 M3FEAT_ERRATA.md 披露, 不倒改历史。

## 3. PC 适配 (A3)
- anomalib 2.6.2; 记录 torch_model.py / k_center_greedy.py 路径与 SHA256。
- reshape_embedding: permute(0,2,3,1).reshape(-1,D) → patch 顺序 (b,h,w) 行主序, 用带位置标识合成特征验证。
- euclidean_dist: FP32 展开 -2xy+|x|^2+|y|^2 (clamp 0) → 与 float64 直接距离参考比较, 含近共线合成向量, 记录误差。
- compute_anomaly_score(nn=9) 官方语义: 每图 argmax patch → 其 NN 位置 m* → m* 在库内 9NN(含自身) → 查询 patch 到这 9 行的距离 softmax → w=(1-softmax)[...,0] → score=w*s*。不是对查询取 9NN 平均。独立参考实现按此公式组合, 与适配输出比较 (两边不同代码)。
- nearest_neighbors 索引检查: 距离与 cdist 参考一致; 并列允许等价邻居, 距离须一致。
- fit 仅访问正常支持 (结构断言 + 记录)。
- 评分批量一致性: 固定 memory_bank, batch=1 vs 32, atol=rtol=1e-4。

## 4. 随机性与库 (A4)
- pc_seed = 7000 + (ref_id - 3000) → 7001..7005。
- 每次正式建库前: random.seed / np.random.seed / torch.manual_seed + torch.cuda.manual_seed_all。
- N*: 5N 支持 8000 patch → KCenterGreedy(ratio=0.1) → 800 原型; 保存选择索引(官方 select_coreset_idxs 可得)、张量哈希、shape、dtype、字节数。
- 双建库: 同种子重建, 张量哈希一致 + 全量评分一致 (非仅 AUROC)。
- 历史库未固定种子 → 与历史差异披露。

## 5. 分块评分 (A5)
- 2D 分块最大余弦: 查询块 × 原型块 running-max; S1-S5/R256 全部经同一实现。
- 验证: 查询 (≥2 块含尾块) × 原型 4000 (≥2 块含尾块) vs 直接计算 (小规模), max-cos atol=1e-5; 64 维。
- 禁止生成 N×1600×4000 巨张量。

## 6. 评分公式 (B, 冻结)
单位向量: cN=max_n cos, dC=1-cN, dE=sqrt(max(0,2-2cN)); 数值验证 dE^2=2*dC (报告误差)。
- S1 = max_patch dE
- S2 = mean(top8 dE)
- S3 = mean(top8 dC)
- S4 = 官方 pred_score (nn=9, 同一 N*)
- S5 = mean(top8 (cF − cN_N*))  [等价形式 cF−cN = dC − (1−cF), 仅作解释不改变实现]
- R256 = 同 S5 但正常库换 FPS256 筛选参照 N256 (仅参考臂, 单列原因)
top8 = patch 分数降序前 8 均值 (与历史 agg 语义一致)。
F4000: 每故障 seed 5 图 ×1600 = 8000 patch, 按 1−max_cos(c→N256筛) stable 降序保留前 4000, 全保留不压缩; 保存 keep 索引 + 哈希; F 固定不因结果重选。

## 7. 配置集合 (预注册, 汇总须集合级完全一致)
S1/S2/S3/S4: ref3001..3005 各 5 配置; S5: 5×3 seed=15; R256: 15。合计 50。
S4 直接复用 A7 评分与库, 不重复建库。

## 8. 统计与记录
- AUROC: D.auroc_avg_rank (平均秩, 并列 ok), 由逐图分数独立重算。
- 分数 float64 精度保存 (JSON 全精度, 不截断)。
- 双库臂 (S5/R256) 先按 3 故障种子汇总再按 5 参照组; 报告逐组/逐种子差、均值、范围、正负数、ddof=0 std。
- 共享查询/重叠参照组不视为独立; 不挑组; 不据结果追加搜索。
- 预算记录: 提取/验证/建库/评分/重复复验分项与总 GPU 时间; 峰值 allocated/reserved; 缓存与库大小。
- 检查点: raw_scores 原子写 + state 哈希绑定; 恢复重建完整汇总并断言预注册集合。

## 9. 预算
GPU 1 小时; 探针 (提取+建库+单配置评分) 先行, 超预算先优化或停止; wall 3 小时。

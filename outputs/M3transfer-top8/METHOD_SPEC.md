# METHOD_SPEC — M3TRANSFER-top8-v1 (冻结于首次新增查询评分前)

## 1. 数据
- 查询: data/splits/query_k5.json; glass defect=glass-insulator/missing-cap, normal=glass-insulator/good; yoke 同 M3SCORE。
- 支持: M3BENCH SUPPORT_LISTS.json; YOLO=N5, WRN=N10; 只读。运行时断言: 集合不相交、数量、路径在评测集、N5⊂N10。

## 2. YOLO 分支 (继承 M3SCORE 冻结配置)
- 特征: ultralytics feats[4] (B,64,40,40 断言), NCHW dim=1 逐位置 L2, fp32; 确定性非 TF32。
- 输入 320 + MC.make_transform; 模型一次加载; 去重批量; state_dict 哈希前后一致。
- yoke: 复用 M3SCORE E1 缓存 (SHA256+身份核对, 路径集合断言); glass: 同规则新缓存。
- 库: 5N -> KCenterGreedy ratio=0.1 -> ~800 原型; pc_seed=7000+(ref-3000); yoke 库哈希对 M3SCORE BANK_MANIFEST 断言一致。

## 3. WRN 分支 (历史标准配置确定性重建)
- PatchcoreModel(wide_resnet50_2, layer2+layer3, pre_trained, nn=9); 预处理复刻 M3BENCH: Resize((320,320),antialias)+Normalize(imagenet); 不加 L2。
- 官方 generate_embedding -> (B,1536,40,40) -> 官方 reshape_embedding。
- 库: 10N (16000 patch) -> ratio=0.1 -> 1600 原型; 种子同公式; yoke ref3001 双建库 torch.equal 核验; 建库仅访问正常支持。
- 查询特征流式 (每 batch 对 5 库算距离), 距离 (n_q, P) 存数据盘 npz + SHA256; Git 只存 manifest。
- 精度: 确定性非 TF32 (历史 M3BENCH 为默认 TF32 + 未固定种子, 历史库不可逐分数复现, 披露)。

## 4. 评分 (同库共享)
- PC = 官方 nearest_neighbors(1) 距离/位置 + compute_anomaly_score(nn=9)。
- T8 = 同一距离矩阵沿每图 patch 轴 top8 均值 (合成验证轴正确性; 无重加权/归一化/方向调整)。
- 欧氏: 官方 euclidean_dist 双分支统一; 不对 WRN 用 sqrt(2-2cos)。
- 数值门: WRN 图像分数官方 vs 独立 float64 组合 atol=rtol=1e-4 (正式评分前 assert); batch=1 vs 16 一致性。
- 分块: 官方 nearest_neighbors 内部 chunk 跨块+尾块 (2500 行) vs float64 直接对照; top8 轴验证。

## 5. yoke 锚点门 (先于 glass)
- Y-PC vs M3SCORE S4, Y-T8 vs M3SCORE S2, 全 5 组逐路径 max|diff|<=1e-4, 立即 assert; 通过后 yoke YOLO 10 配置复用为矩阵单元。

## 6. 汇总
- 40 配置集合级断言; 逐图分数 JSON 全精度; 平均秩 AUROC 独立重算 (<1e-12)。
- 每资产 4 臂统计; Y-T8-Y-PC; W-T8-W-PC; 外部 Y-T8-W-PC, Y-T8-W-T8; 新 W-PC-历史 W-PC 逐组。
- 冻结解释 A/B/C/D 见合同; 不自动触发后续实验。

## 7. 预算
GPU 2h 上限; 探针先行; 阶段耗时/显存分项; 不称缓存评分为端到端延迟。

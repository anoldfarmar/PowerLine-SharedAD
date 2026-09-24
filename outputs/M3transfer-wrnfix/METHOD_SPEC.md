# METHOD_SPEC — M3TRANSFER-wrnfix-v1 (冻结于首次新增查询评分前)

## 1. 官方路径审计结论 (先于冻结确认)
安装版 anomalib 2.6.2 PatchcoreModel.forward (torch_model.py, 哈希存档) 含局部池化:
  features = self.feature_extractor(x)
  features = {layer: self.feature_pooler(f) for ...}   # AvgPool2d(3,1,1)
  embedding = self.generate_embedding(features)
M3TRANSFER-top8-v1 薄适配直接 generate_embedding(feature_extractor(x)), 遗漏池化 ->
旧 WRN 结果定性为「未池化WRN变体」, 原始文件不动, 差异在本轮单独报告。

## 2. 共享标准提取 (唯一路径)
std_extract(pm, x_batch): x(fp32) -> pm.feature_extractor (eval) -> pm.feature_pooler(逐层)
  -> pm.generate_embedding -> pm.reshape_embedding -> flat (n*P, 1536)。
支持建库、查询评分、路径门验证全部复用同一函数; 不允许旁路实现。

## 3. 正确性门 (正式评分前立即 assert)
1. 端到端路径门: 同一模型实例、同一预处理批。
   路径1(独立官方): 模型临时 train() 后官方 forward 返回的 embedding (官方内部含池化+融合+展平);
   路径2: std_extract。atol=rtol=1e-4。
   train 前保存 state_dict 与 embedding_store 长度, 验证后恢复并清理; 状态哈希前后一致断言。
2. 评分门: 固定库 (ref3001 yoke N10 库) 下, 官方完整推理 pred_score (eval forward)
   vs 本轮 nearest_neighbors+compute_anomaly_score 链, 逐图 atol=rtol=1e-4。
3. top8 轴合成验证 (patch 轴); batch=1 与正式批量一致性; 形状/有限值/dtype。
4. 库重建: yoke+glass ref3001 同种子双建库 torch.equal。
5. 历史集合断言: glass 用 missing-cap_g2_ref*.json (705 故障查询); yoke 用 g2;
   当前 N10 与冻结名单一致、查询路径集合/标签侧一致后才能比较。

## 4. 配置与评分
- 库: 10N 支持池化后 patch (10x1600) -> KCenterGreedy ratio=0.1 -> 1600 原型 (以实际形状记录);
  pc_seed=7000+(ref-3000); 原型索引/shape/dtype/字节/张量哈希/支持名单/种子全存档。
- W-PC-FIX: 官方 nearest_neighbors(1) + compute_anomaly_score(nn=9)。
- W-T8-FIX: 同库同距离 patch 轴 top8 均值 (无重加权/无归一化/top-k 固定 8)。
- 每资产查询单次主干前向, 同时对 5 库评分; 距离 npz 存数据盘。

## 5. 权重身份
M3BENCH 与本轮同机同环境同 anomalib TimmFeatureExtractor(pre_trained=True, wide_resnet50_2);
timm 默认 pretrained_cfg 实际解析结果 + HF 缓存文件 + 本轮实际加载 state_dict SHA256 存档;
历史 state 哈希未留存 -> 标「历史权重身份未完全核实 (同代码路径+同缓存, 高一致可能)」。
本轮结果称「权重明确的标准WRN配置重评」, 不宣称逐项复现历史。torchvision 95faca4d 与
timm racm_in1k 是不同训练产物, 本轮 timm 路径不加载 torchvision 文件 (如两文件同时在盘则做张量级对比佐证)。

## 6. 汇总与勘误
- 20 配置集合断言; 逐图 JSON 全精度; 平均秩 AUROC 独立重算。
- fix vs 旧未池化变体: 权重路径一致证据成立时解释为池化效应(附历史哈希缺失限定), 否则仅系统差异。
- M3TRANSFER_ERRATA: 撤回 glass g1 历史对比与「库随机性±0.08」单因素归因 (降级为多因素系统差异)。
- 复用 M3TRANSFER YOLO 分数做外部比较 (Y-T8 − W-PC-FIX / W-T8-FIX)。

## 7. 预算
GPU 1h 上限; 探针先行; 阶段耗时/显存分项; 评分耗时非端到端延迟。

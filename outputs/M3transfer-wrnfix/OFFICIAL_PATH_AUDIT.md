# OFFICIAL_PATH_AUDIT (安装版 2.6.2, 非网上 main)

实际安装 forward (torch_model.py sha=a5e5bc1ed22ed86d) 含:

    features = self.feature_extractor(input_tensor)
    features = {layer: self.feature_pooler(feature) for ...}  # AvgPool2d(3,1,1)
    embedding = self.generate_embedding(features)
    embedding = self.reshape_embedding(embedding)
    patch_scores, locations = self.nearest_neighbors(embedding, n_neighbors=1)
    pred_score = self.compute_anomaly_score(...)

M3TRANSFER-top8-v1 薄适配 = generate_embedding(feature_extractor(x)), **遗漏池化** -> 登记为特征路径偏离; 旧 WRN 结果命名「未池化WRN变体」, 原始文件不动。本轮唯一提取函数 std_extract = extractor -> pooler -> generate_embedding -> reshape_embedding, 支持/查询/验证共用。

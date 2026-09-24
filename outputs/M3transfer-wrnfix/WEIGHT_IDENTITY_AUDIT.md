# WEIGHT_IDENTITY_AUDIT

- M3BENCH 冻结记录只存 YOLO 权重哈希, WRN state 哈希未留存 -> **历史 WRN 权重身份未完全核实**。
- 但历史与本轮同机/同环境 (torch 2.8.0+cu128, anomalib 2.6.2) 同代码路径: Patchcore(pre_trained=True, backbone=wide_resnet50_2) -> anomalib TimmFeatureExtractor -> timm create_model(pretrained=True) -> timm 默认 pretrained_cfg={"url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth", "hf_hub_id": "timm/wide_resnet50_2.racm_in1k", "architecture": "wide_resnet50_2", "tag": "racm_i -> HF 缓存 model.safetensors (sha256=03b71d65fb2c73bb)。
- 本轮实际加载 feature_extractor state_dict sha256=99e63a573856efec (冻结, 不按查询结果挑选)。
- torchvision wide_resnet50_2-95faca4d.pth 与 timm racm_in1k 为不同训练产物: conv1.weight max|diff|=3.510e+00 (不同训练产物, timm 路径不加载 torchvision 文件)
- 结论: 分支 B (证据链强一致但无历史哈希直证); 本轮结果称「权重明确的标准WRN配置重评」, 不宣称逐项复现历史。

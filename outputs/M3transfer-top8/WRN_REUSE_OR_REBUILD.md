# WRN_REUSE_OR_REBUILD

审计: M3BENCH 仅保留图像分数 (raw_scores_patchcore/*.json), 未保存 memory_bank、查询特征或逐 patch 距离; 建库随机起点未固定, 历史库无法逐分数复现。

依合同 §七回退: 按原标准配置 (WRN50 layer2+3, 320 输入 + Normalize, ratio=0.1, nn=9) 确定性重建, pc_seed=7000+(ref-3000), yoke ref3001 双建库张量哈希核验; W-PC 与 W-T8 在同一新库同距离上评测; 历史 W-PC 仅作参考 (新-历史逐组见 transfer_eval.json wrn_rebuild_vs_hist); 新旧 W-PC 差异不解释为纯聚合收益。

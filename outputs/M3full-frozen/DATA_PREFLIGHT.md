# DATA_PREFLIGHT — M3FULL-frozen-benchmark-v1

| 资产 | 原始候选 | 可用池 | 正常查询 | 故障查询合计 | 池DJI∩正常查询源 | 池DJI∩故障查询源 |
|---|---|---|---|---|---|---|
| glass-insulator | 720 | 220 | 500 | 705 | 3 | 1 |
| lightning-rod-suspension | 579 | 79 | 500 | 315 | 0 | 0 |
| polymer-insulator-upper-shackle | 773 | 273 | 500 | 747 | 1 | 0 |
| vari-grip | 596 | 96 | 500 | 653 | 63 | 0 |
| yoke-suspension | 6041 | 5541 | 500 | 295 | 430 | 0 |

- 可行组合: glass-insulator/missing-cap, lightning-rod-suspension/rust, polymer-insulator-upper-shackle/corrosion, polymer-insulator-upper-shackle/rust, vari-grip/bird-nest, vari-grip/rust, yoke-suspension/rust
- 阻塞组合: 无
- 预计配置: 280 (复用 80 + 新增 280)

## 口径披露

- 隔离口径: 仅路径级 支持∩查询=空; 源组身份按 M3DIAG5 审计为未知/可能同源 (裸DJI号跨航次重号), 不宣称独立未见或骨干未见
- 既有 60 张正常参照 (M3DIAG3/4) 与 det-train 全部 DJI 碰撞; 本轮支持池沿用 defect_supervised good/normal 池, 属评测池隔离
- 正常支持以裁剪计数, 非独立拍摄源帧数

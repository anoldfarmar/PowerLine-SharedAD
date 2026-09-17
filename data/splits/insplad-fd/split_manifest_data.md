# k-shot 划分清单 (M2 Step-4)

原型池采样单元=源图像组((DJI_\d+)), 查询集按 k 固定(3 种子原型池并集之外的缺陷实例), 正常查询池=FD val/good 分层下采样 ≤500/资产。

| 组合 | 池规模 | 源图组数 |
|---|---|---|
| glass-insulator/missing-cap | 720 | 81 |
| lightning-rod-suspension/rust | 330 | 49 |
| polymer-insulator-upper-shackle/corrosion | 740 | 70 |
| polymer-insulator-upper-shackle/rust | 33 | 31 |
| vari-grip/bird-nest | 373 | 40 |
| vari-grip/rust | 310 | 30 |
| yoke-suspension/rust | 310 | 37 |

- k=1: 原型池/种子组合数=7×3种子; 缺陷查询=2796 实例 / 7 组合; 正常查询=2500 实例 / 5 资产池
- k=5: 原型池/种子组合数=7×3种子; 缺陷查询=2715 实例 / 7 组合; 正常查询=2500 实例 / 5 资产池
- k=10: 原型池/种子组合数=7×3种子; 缺陷查询=2614 实例 / 7 组合; 正常查询=2500 实例 / 5 资产池

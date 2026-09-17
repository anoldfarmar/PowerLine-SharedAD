# k-shot 划分固化清单（M2 Step-4 交付物）

> 任务：M2-proto-head-v0 · 固化日期：2026-09-15 · 脚本：`scripts/m2_kshot_splits.py`
> 划分文件（合同授权写入点）：`/root/auto-exp/data/splits/insplad-fd/`
> - 原型池：`kshot_k{1,5,10}_seed{0,1,2}.json`（9 个）
> - 查询集：`query_k{1,5,10}.json`（3 个，按 k 固定，跨种子可比）
> - 明细表：`split_manifest_data.md`

## 协议设计（文献无 k-shot 惯例，本划分为本课题自定义协议）

1. **建库粒度**：资产×故障组合（7 个），即原型库每组合独立原型（PLAN §3 宏类协议的决策层归并在此之上）。
2. **防泄漏**：以**源图像组**（文件名中 `DJI_\d+`）为采样单元——同一原帧的裁剪不会同时出现在原型池与查询集；k-shot 从 k 个不同源组各取 1 实例（组数不足时同组多实例，本批次 7 组合在 k=10 下均有 ≥30 源组，未触发）。
3. **查询集固定**：缺陷查询 = 全缺陷池 − 该 k 下 3 个种子原型池的并集；因此同一 k 下 3 个种子共用同一查询集，种子间差异只来自原型池。
4. **正常查询池**：FD val/good 按资产分层下采样（≤500/资产，seed 固定），用于原型抑制/背景判别评估。
5. **种子机制**：`sha256(f"insplad-fd-k{k}-s{seed}")` 派生 RNG，划分可复现；无人工干预。
6. **类名归一化**：复用 `scripts/m1_fd_stats.py::LABEL_NORMALIZE`（corrosão→corrosion, normal→good），单一事实来源。

## 缺陷池构成（实测）

| 组合 | 池规模 | 源图组数 | | 组合 | 池规模 | 源图组数 |
|---|---|---|---|---|---|---|
| glass-insulator/missing-cap | 720 | 81 | | vari-grip/bird-nest | 373 | 40 |
| lightning-rod-suspension/rust | 330 | 49 | | vari-grip/rust | 310 | 30 |
| polymer-…-upper-shackle/corrosion | 740 | 70 | | yoke-suspension/rust | 310 | 37 |
| polymer-…-upper-shackle/rust | 33 | 31 | | | | |

## 查询集规模（固化值）

| k | 缺陷查询实例 | 缺陷查询组合 | 正常查询实例 |
|---|---|---|---|
| 1 | 2,796 | 7 | 2,500（5 资产池） |
| 5 | 2,715 | 7 | 2,500 |
| 10 | 2,614 | 7 | 2,500 |

## 划分文件哈希登记（sha256 前 16 位，2026-09-16 基线检测训练 pre-flight 登记）

| 文件 | 哈希 | 文件 | 哈希 |
|---|---|---|---|
| kshot_k1_seed0.json | `c5a94974bf91f576` | kshot_k5_seed2.json | `5558bc0c9b95c3a6` |
| kshot_k1_seed1.json | `938289d60545cf20` | kshot_k10_seed0.json | `934aa9fb0c75ad39` |
| kshot_k1_seed2.json | `49a53afcbe83c8fa` | kshot_k10_seed1.json | `b7887d6b8dd584a1` |
| kshot_k5_seed0.json | `79b672173e851273` | kshot_k10_seed2.json | `cb4483c98c955235` |
| kshot_k5_seed1.json | `26ef5014aa44bb03` | query_k1.json | `ac639b93fcb32599` |
| query_k5.json | `0fcc8c93bc8096ed` | query_k10.json | `1e17f5147b149a99` |

后续任何消费划分文件的任务/评测必须先比对哈希；不一致 = 划分被改动 = 立即停止报告人类。

## 使用纪律（对后续所有任务的约束）

- 任何评测报告引用 k-shot 结果时，划分以此目录文件为准；**禁止重新生成或挑选划分**（PLAN §5.3 可证伪性声明）。
- 3 种子 mean±std 报告；换 k 或换种子数 = 修改协议，须走人工门。
- `polymer-insulator-upper-shackle/rust`（33 实例）k=10 时查询侧仅剩 3 实例，该组合的 10-shot 结果统计意义弱，报告中须标注。

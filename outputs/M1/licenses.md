# 数据集许可条款核查（M1 Step-4）

> 核查日期：2026-09-15。来源：Mendeley 数据集页、GitHub API `license` 字段与 README、datasetninja 页面元数据。

## 1. InsPLAD（主实验数据集）

- **许可：CC BY-NC 3.0**（[Mendeley 官方页](https://data.mendeley.com/datasets/5n3fjgvfyz/1) 明示 "CC BY NC 3.0"，链接 creativecommons.org/licenses/by-nc/3.0）
- GitHub 仓库 [andreluizbvs/InsPLAD](https://github.com/andreluizbvs/InsPLAD)：license 字段为 `Other / NOASSERTION`（未给出标准 SPDX 许可）
- **含义**：允许署名后的共享与改编（含论文使用），**禁止商业用途**。
- **论文使用合规**：✅ 学术论文中的使用（复制图表、统计数据、实验结果）属非商业，合规；需在论文中署名引用。
- **再分发合规**：⚠️ 硕士论文本身非商业出版可接受；若后续衍生数据集/代码以开源形式再分发且含 NC 限制冲突（如 GPL 代码库混合），需避免商业化并保留许可声明。**禁止将基于该数据的模型/数据用于任何商业或工业部署收费场景**（含企业委托测试）。

## 2. STN-PLAD（跨域数据集）

- **许可：GPL-3.0**（GitHub [andreluizbvs/PLAD](https://github.com/andreluizbvs/PLAD) 仓库 license 字段：GNU GPL v3.0；datasetninja 元数据同样给出 gpl-3.0 链接）
- **含义**：GPL-3.0 作用于**代码与数据制品的再分发**：分发衍生数据/代码时须以同许可开放，附源与许可声明。
- **论文使用合规**：✅ 论文中报告实验结果、引用统计不构成"分发"，合规。
- **再分发合规**：⚠️ 若公开发布基于 STN-PLAD 构建的跨域原型/划分文件，需以 GPL-3.0 兼容方式发布。对学位论文无影响。

## 3. CPLID（备选数据集）

- **许可：无明确许可声明**（GitHub [InsulatorData/InsulatorDataSet](https://github.com/InsulatorData/InsulatorDataSet) 无 license 字段；README 无许可/使用条款；来自论文 Tao et al., IEEE T-SMC: Systems 2018 的配套数据）
- **论文使用合规**：✅ 惯例上学术引用与实验使用可接受（公开托管 + 论文配套），但**无书面授权**。
- **再分发合规**：❌ 无许可即默认保留所有权利——**不得再分发**其原始图像；论文中只引用与描述。
- **风险提示**：CPLID 含合成缺陷（贴图生成），与真实缺陷分布差异大，作为备选靶时需在论文声明。

## 4. 汇总表

| 数据集 | 许可 | 论文使用 | 再分发 | 商业使用 | 角色 |
|---|---|---|---|---|---|
| InsPLAD | CC BY-NC 3.0 | ✅ 需署名 | ⚠️ 非商业前提 | ❌ | 主实验 |
| STN-PLAD | GPL-3.0 | ✅ | ⚠️ 同许可开放 | ⚠️ GPL 传导 | 跨域靶 |
| CPLID | 无声明 | ✅ 惯例可用 | ❌ 保留所有权利 | ❌ | 备选 |

## 5. 对本课题的合规动作清单

1. 论文数据集章节逐一署名引用三个数据集（InsPLAD: Oliveira et al. 2024 IJRS；STN-PLAD: PLAD 论文；CPLID: Tao et al. 2018）。
2. 公开代码/划分文件仓库：附 LICENSE 与 DATA_LICENSE 说明（InsPLAD 数据不随仓库分发，仅提供下载指针）。
3. 若答辩后成果转化/落地（RK3588 工业部署），主数据集 CC BY-NC 与企业场景冲突——部署验证应使用合作方自有数据，不使用 InsPLAD 训练权重的商业交付物。

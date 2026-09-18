# M3 修正案：AD 基线支持样本池（选项 A）

- 日期：2026-09-17
- 裁决人：申请人（人类，本会话内明示选择"选项 A（推荐）"）
- 影响范围：仅 M3 Step-5 anomalib 基线的**支持样本池定义**；查询集、k-shot 划分、
  阈值规则、种子、指标定义等全部冻结协议**不变**

## 背景

preflight.json 冻结的支持样本规则为"只来自非查询的 train/good 池"。实现核验时发现该规则不可行：
M2 冻结的 normal_query（每资产 500 张正常查询）物理上就是 good 裁剪池本身，扣除冻结重叠后
非查询 good 池仅剩：

| 资产 | train/good（冻结重叠） | val/good（冻结重叠） | 非查询 good 合计 |
|---|---|---|---|
| glass-insulator | 691（665） | 29（29） | 26 |
| lightning-rod-suspension | 348（347） | 231（231） | **1** |
| polymer-insulator-upper-shackle | 目录不存在 | 31（29） | **2** |
| vari-grip | 358（356） | 238（237） | **3** |
| yoke-suspension | 299（65） | 5742（1322） | 4654 |

即 3/5 资产连 k=5/10 都无法满足，k=1 也仅勉强可行。

## 裁决内容（选项 A）

1. 支持样本池 = 该资产**全部 good 裁剪**（train∪val），冻结 RNG 规则不变
   （`sha256('insplad-fd-ad-support-<asset>-k<k>-s<seed>')` 派生种子，排序池无放回抽 k 张）；
2. 硬红线（已验证）：支持样本**不得命中任何 defect/kshot 冻结条目**——defect/kshot 冻结名
   与 good 池文件名零交集（5 资产全部 0 命中），该红线结构性成立；
3. 与 normal_query 的重叠**允许并逐配置登记**（`normal_query_overlap` 字段，含文件名与 sha256）。

## 偏差方向声明

AD 基线的拟合支持样本与被打分的 normal_query 部分重合：AD 对已拟合样本打分趋近于零异常，
使正常侧分数更低、AD 的 AUROC **略被抬高**。该偏差**有利于 AD 基线**，即对本方案
（H1：与 AD 差距 ≤2 点）是**保守方向**——不会高估本方案的相对优势。

## 与验收的关系

`baselines_eval.json` 的 `query_leakage_check_passed` 语义相应更新为：全部配置无
defect/kshot 泄漏（硬红线通过）即 1。原语义（与任何冻结条目零重叠）因选项 A 不再适用，
如实在本文件与 M3_report.md 披露。preflight.json 为历史冻结产物，不回改。

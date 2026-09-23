# M3CAP-yoke-v1 勘误记录（人类授权 M3FEAT-yoke-v1 简报第二部分）

所有更正只涉及报告文字与口径，历史原始分数与 AUROC 一律未改；数值全部来自机读记录（cap_eval.json / BANK_MANIFEST.json / RESOURCE_COST.json）。

1. 描述性交互：AUROC_RNF − AUROC_RF − AUROC_RN + AUROC_R0 = −0.044291（15 配置均值口径）。原报告"RNF ≈ 两单臂收益代数和"为错误解释，已删除，替换为描述性交互陈述，不作为统计显著或已证实机制。
2. 库维度实际为 128 维 FP32（mapper_B 输出维度；BANK_MANIFEST N256 dim=128, bytes=131072）。原报告"512×64 float32 = 256KB"及"+0.94MB/+0.98MB"等数字错误，已按真实 shape/dtype/字节数更正：R0 0.250 MiB，RN 2.125 MiB，RF 2.078125 MiB，RNF 3.953125 MiB。
3. 显存口径：机读记录为 torch.cuda.max_memory_allocated()（allocated），480,089,600 B = 458 MiB；reserved 未记录。原报告"0.20GB"数字错误且未注明口径，已更正并统一为 allocated/reserved 分开表述。
4. 原始逐图分数、AUROC 与配对统计未做任何更改。

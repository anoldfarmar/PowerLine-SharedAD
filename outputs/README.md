# 实验结果目录

2026-10-27按用户要求清理中间结果，只保留README主线及复现所必需的依据。

| 目录 | 保留原因 |
|---|---|
| `reports/InsPLAD-COCO-top8-full-v1/` | COCO＋top8完整比较 |
| `M3full-frozen/` | YOLO＋top8、标准PatchCore及同库评分对照；保留冻结名单和280份逐图分数 |
| `M3transfer-wrnfix/` | PatchCore官方池化、权重身份与端到端一致性更正依据；CPLID入口读取其中的审计文件 |
| `M3diag5-testaudit/` | 数据源帧、shot语义审计；小图测速入口仍读取审计。应结合SHOT_AUDIT与COUNTING_CORRECTION阅读，不能沿用早期“100%重叠”推断 |
| `reports/InsPLAD-*`、`reports/CPLID-*` | 冻结前缀检测器、共享主干、精度／速度、部署及板端验证，含CPLID PatchCore比较 |
| `reports/M3SHARE-speed-v1/` | 共享与两阶段流水线成本比较 |
| `reports/M3BASELINE-AnomalyDINO-full-v1/` | README已引用的补充对照，保留证据 |
| `reports/README-results-sync-v1/` | README发布与源文件核验记录 |

已删除：M1/M2旧输出、早期原型／mapper、双库、正常库交叉、容量／多层扫描、去噪残差、评分中间诊断、未修正WRN变体，以及旧清理任务报告。完整删除名单、字节数和SHA256见[CLEANUP_MANIFEST.json](CLEANUP_MANIFEST.json)。

算法、部署、数据划分、训练权重及评测数值均未修改。旧实验脚本、合同与登记作为历史保留；已删除输出对应的旧脚本不再承诺开箱复现，需先恢复依赖。历史报告中的路径和哈希是当时快照，不改写成新的运行证据。此次没有重跑训练、评测或板端推理。

## 恢复历史结果

本轮清理前提交：`00be86ed1639d756bb5dfe204620b4eb9b492cc4`。例如恢复一个目录：

```bash
git restore --source=00be86ed1639d756bb5dfe204620b4eb9b492cc4 -- outputs/M3cap-yoke
```

此操作会恢复对应工作区文件；先检查本地修改，恢复后不要误将历史产物重新提交。更早在服务器清理的大缓存不在本轮Git快照内，旧恢复包记录为：

`/root/autodl-tmp/outputs-archive/legacy-cleanup-20261022-Ul6AlQ/legacy-outputs.tar.gz`

其SHA256：`f1053c14f7480c94ccdb8e6ea5485e343ec75dcee80a535f48ba955bd0ec9cb7`。本地未验证该服务器归档可用性。

删除结果通过普通提交保存，不改写Git历史，因此不会显著缩小已有`.git`或后续完整克隆的历史体积。
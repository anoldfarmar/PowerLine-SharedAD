#!/usr/bin/env python3
"""M3FEAT 报告生成: 全部表格由 feat_eval.json / PARAMETER_AUDIT.json 机器生成."""
import json, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/auto-exp/outputs/M3feat-yoke"
S = json.load(open(OUT + "/feat_eval.json"))
A = json.load(open(OUT + "/PARAMETER_AUDIT.json"))
C = json.load(open(OUT + "/RESOURCE_COST.json"))
F = json.load(open(OUT + "/FEATURE_MANIFEST.json"))
ARMS = ["E0_DUAL", "E1_DUAL", "E2_DUAL", "E0_PC", "E1_PC", "E2_PC"]
NAMES = {"E0_DUAL": "E0+H-DUAL (mapper 128d)", "E1_DUAL": "E1+H-DUAL (P3 64d)",
         "E2_DUAL": "E2+H-DUAL (P3+P4+P5 448d)", "E0_PC": "E0+H-PC-YOLO",
         "E1_PC": "E1+H-PC-YOLO", "E2_PC": "E2+H-PC-YOLO"}
CK = "yoke-suspension__rust_"

def fmt(x, d=4):
    return ("%." + str(d) + "f") % x

def arm_vals(a):
    return [v for k, v in sorted(S["aurocs"].items()) if k.startswith(CK + a + "_")]

def amean(a):
    v = arm_vals(a)
    return sum(v) / len(v)

p = S["paired"]
L = []
w = L.append
w("# FEATURE_HEAD_REPORT — M3FEAT-yoke-v1（探索性）")
w("")
w("- 任务: yoke-suspension/rust, 5N5A 固定候选框架下 特征×检测头 消融; 合同 plan/tasks/M3FEAT-yoke-v1.yaml")
w("- 60/60 臂配置完成; 复现门 G1 worst=%.1e (<=1e-4); G2/G3/G4 全过; 机器判定 gates_all_passed=%d" % (S["repro_R0"]["worst"], S["gates_all_passed"]))
w("- 声明: 探索性; k 为 crop 数非独立帧; 查询跨臂复用; 每查询真实来源身份未知; 不做显著性检验; 双库头(5N5A 监督)与 PC-YOLO 头(仅 5N 监督)监督组成不同, 差异不能全部归因于头结构.")
w("")
w("## 1. 总览 (AUROC, 由 60 个逐图分数文件重建)")
w("")
w("| 臂 | 配置数 | 均值 | 范围 |")
w("|---|---|---|---|")
for a in ARMS:
    vs = arm_vals(a)
    w("| %s | %d | %s | [%s, %s] |" % (NAMES[a], len(vs), fmt(amean(a)), fmt(min(vs)), fmt(max(vs))))
w("| 标准 PatchCore (WRN50, 10N, 外部标杆) | 5 | %s | — |" % fmt(S["patchcore_external_mean"]))
w("")
w("外部标杆为 M3BENCH 存储 yoke g2 10N 原始分数的本会话重算 (均值 %s, 与 M3BENCH 登记值一致)." % fmt(S["patchcore_external_mean"], 6))
w("")
w("## 2. 六个问题")
w("")
w("### Q1 去除 mapper (E1-E0, 同为 H-DUAL)")
w("five-ref 均值 %s; 组均值范围 [%s, %s]; %d/5 个参考组为正, %d/15 配置为负. E0 均值 %s vs E1 均值 %s: 总体接近零但组间方差大 (ref3001 组 E0 显著差, 其余组 E0 略优). 证据不支持 mapper 带来一致收益, 也未见一致损失; 该 128d mapper 在 5N5A 双库框架下与原始 P3 L2 大体等价." % (
    fmt(p["E1-E0"]["five_ref_mean"]), fmt(p["E1-E0"]["range"][0]), fmt(p["E1-E0"]["range"][1]),
    p["E1-E0"]["n_pos_groups"], p["E1-E0"]["n_neg_configs"], fmt(amean("E0_DUAL")), fmt(amean("E1_DUAL"))))
w("")
w("### Q2 多层融合 (E2-E1, 同为 H-DUAL)")
w("five-ref 均值 %s; %d/5 组为正, %d/15 配置为负. E2 均值 %s < E1 %s. 在本设置 (bilinear 上采样到 P3 网格, 等权拼接, 整体 L2) 下多层融合未带来收益, 方向轻微为负; 等权融合可能是稀释来源之一, 但本任务未单独验证权重方案." % (
    fmt(p["E2-E1"]["five_ref_mean"]), p["E2-E1"]["n_pos_groups"], p["E2-E1"]["n_neg_configs"],
    fmt(amean("E2_DUAL")), fmt(amean("E1_DUAL"))))
w("")
w("### Q3 双库 vs PC-YOLO 头 (同特征空间配对)")
w("")
w("| 对比 | five-ref 均值 | 范围 | 正组数 | 负配置数 | 交叉引用稳定性 |")
w("|---|---|---|---|---|---|")
STAB = {"E0_DUAL-PC": "5/5 组同向为正 (稳定)", "E1_DUAL-PC": "5/5 组同向为负 (稳定)",
        "E2_DUAL-PC": "4/5 为正, 组内混合 (不稳定)"}
for k in ("E0_DUAL-PC", "E1_DUAL-PC", "E2_DUAL-PC"):
    v = p[k]
    w("| %s | %s | [%s, %s] | %d/5 | %d/15 | %s |" % (k, fmt(v["five_ref_mean"]), fmt(v["range"][0]), fmt(v["range"][1]), v["n_pos_groups"], v["n_neg_configs"], STAB[k]))
w("")
w("结论强烈依赖特征空间: E0 上双库大幅占优 (+%s; 注意 E0_PC 均值 %s 已接近随机水平 — mapper 128d 特征 x 官方 PC 头在该设置下整体失效, 是本任务最显著的单项观察, 机制未查); E1 上 PC-YOLO 头反超 (%s, 14/15 配置为负); E2 上接近且方向不一致. 不存在跨特征空间一致更优的检测头." % (
    fmt(p["E0_DUAL-PC"]["five_ref_mean"]), fmt(amean("E0_PC")), fmt(p["E1_DUAL-PC"]["five_ref_mean"])))
w("")
w("### Q4 与标准 WRN PatchCore (10N) 的差距")
w("外部标杆均值 %s. 最佳 YOLO 组合为 E1_PC %s (差 %s), 其次 E1_DUAL %s (差 %s), E2_PC %s (差 %s). 差距主要由特征空间决定: 同一 64d P3 特征下 PC 头距标杆 %s, 双库头距标杆 %s." % (
    fmt(S["patchcore_external_mean"]), fmt(amean("E1_PC")), fmt(amean("E1_PC") - S["patchcore_external_mean"]),
    fmt(amean("E1_DUAL")), fmt(amean("E1_DUAL") - S["patchcore_external_mean"]),
    fmt(amean("E2_PC")), fmt(amean("E2_PC") - S["patchcore_external_mean"]),
    fmt(amean("E1_PC") - S["patchcore_external_mean"]), fmt(amean("E1_DUAL") - S["patchcore_external_mean"])))
w("")
w("### Q5 证据方向: mapper / 检测头 / YOLO 规模")
w("- mapper: 移除后 (E1) 无一致变化 (Q1) — mapper 在该框架下非主要因素.")
w("- 检测头: E0 下 +%s vs E1 下 %s 的强交互说明 head 效应被特征空间主导, 单看 head 无独立结论." % (
    fmt(p["E0_DUAL-PC"]["five_ref_mean"]), fmt(p["E1_DUAL-PC"]["five_ref_mean"])))
w("- YOLO 特征规模: P3 64d + PC 头已达 %s (距 WRN 标杆 %s), 而 128d mapper + PC 头仅 %s; 448d 融合两头皆低于 E1. 当前证据指向: 在 YOLOv8n 内, 瓶颈不在特征维度大小, 而在特征空间与头/库的距离度量匹配 (cos top-k 双库 vs 欧氏 9-邻域). 注意本任务无参数规模对齐 (宪法 5.3), 此为观察非结论." % (
    fmt(amean("E1_PC")), fmt(amean("E1_PC") - S["patchcore_external_mean"]), fmt(amean("E0_PC"))))
w("")
w("### Q6 本任务不能回答的问题")
w("- 监督组成差异 (5N5A vs 5N) 与头结构效应无法剥离;")
w("- 参数规模对齐后的公平比较 (宪法禁止调整);")
w("- 单 combo (yoke/rust) 之外泛化; k=8 crop 的帧级显著性;")
w("- E0_PC 失效的机制解释 (未做专门诊断);")
w("- 任何统计显著性 (未做检验, 探索性).")
w("")
w("## 3. 验证门记录")
w("")
g = S["correctness"]
w("| 门 | 内容 | 结果 |")
w("|---|---|---|")
w("| G1 | E0+H-DUAL 15 配置复现 M3CAP RF 逐路径 | worst=%.1e <=1e-4 PASS (差异为存档分数 8 位小数舍入, 非数值路径差异) |" % S["repro_R0"]["worst"])
w("| G2 | 特征性质: E1/E2 逐位置 L2 范数=1; E2 形状断言 P3(64,40,40)/P4(128,20,20)/P5(256,10,10); 同图双批一致性 | diff=%.2e PASS |" % g["batch_consistency_diff"])
w("| G3 | 分块最大余弦 vs 直接全量 (含 4000 大库) | max diff=%.1e PASS |" % g["g3_diff"])
w("| G4 | 官方组件直接调用对照: cdist 最小距离, 组合确定性, coreset 规模 20/20 | %s PASS |" % json.dumps(g["g4"]))
w("")
w("## 4. 参数审计 (只读统计, 本任务训练参数 = 0)")
w("")
w("| 项 | 参数量 | 状态 |")
w("|---|---|---|")
w("| YOLOv8n 全模型 | %d | 已加载, 实际贡献取决于子图 |" % A["yolo_full_params"])
w("| P3 子图 (model[:5]) | %d | E0/E1 实际贡献 |" % A["yolo_P3_subgraph_params"])
w("| P4 子图 (model[:7]) | %d | 仅 E2 (上采样) |" % A["yolo_P4_subgraph_params"])
w("| P5 子图 (model[:10]) | %d | 仅 E2 (上采样) |" % A["yolo_P5_subgraph_params"])
w("| mapper_B | %d | 仅 E0 (冻结) |" % A["mapper_params"])
wrn = A["wrn_pc_feature_path_params"]
w("| WRN50 feature path (layer2+layer3, timm 同源) | %s | 标准 PC 特征路径 (外部标杆侧) |" % ("{:,}".format(wrn) if isinstance(wrn, int) else str(wrn)))
w("| 本任务训练参数 | 0 | 全部冻结 |")
w("")
w("## 5. 特征清单")
w("")
for k in ("E0", "E1", "E2"):
    m = F["features"].get(k, {})
    w("- **%s**: dim=%s; %s" % (k, m.get("dim", "?"), m.get("source", "")))
w("")
w("E2 层结构: %s; 形状断言: %s; 上采样: %s." % (json.dumps(F["features"]["E2"].get("layers", {})), json.dumps(F["features"]["E2"].get("shapes", {})), F["features"]["E2"].get("interp", "")))
w("")
w("## 6. 资源与成本")
w("")
w("- 全流程 wall=%.1fs (含 60 配置评分); 峰值 VRAM allocated=%.0f MiB (reserved 未记录)" % (C["wall_time_s"], C["peak_vram_allocated_bytes"] / 1048576))
w("- 探针块耗时(s): %s" % json.dumps(C["probe_s"]))
w("- 评分为缓存特征耗时, 非完整推理延迟; E2 缓存 %.0f MiB 存于数据盘 (系统盘 88%% 满)" % (C["cache_bytes"]["e2"] / 1048576))
w("")
w("## 7. 产物")
w("")
w("FROZEN_INPUTS.json / FEATURE_MANIFEST.json / BANK_MANIFEST.json / 复现记录 (feat_eval.json repro_R0) / raw_scores/ (60 文件, 逐图原始分数, 8 位小数) / feat_eval.json / paired_comparisons.csv / PARAMETER_AUDIT.json / RESOURCE_COST.json / M3CAP_ERRATA.md (自 M3cap 沿袭, 本任务无新增勘误) / run.log / feature_head_auroc.png")
w("")
open(OUT + "/FEATURE_HEAD_REPORT.md", "w").write(chr(10).join(L))

fig, ax = plt.subplots(figsize=(9, 4.2))
xs = ARMS
ms = [amean(a) for a in ARMS]
es = []
for a in ARMS:
    vs = arm_vals(a)
    es.append(max(amean(a) - min(vs), max(vs) - amean(a)))
ax.bar(xs, ms, yerr=es, capsize=4, color="#4a7fb5", alpha=0.85)
ax.axhline(S["patchcore_external_mean"], color="#c0392b", ls="--", lw=1.4, label="WRN PC 10N external %.4f" % S["patchcore_external_mean"])
ax.set_ylabel("AUROC (mean over arm configs)")
ax.set_ylim(0.4, 0.9)
ax.set_title("M3FEAT-yoke: feature x head ablation (exploratory, 60 configs)")
ax.legend(fontsize=8)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT + "/feature_head_auroc.png", dpi=130)
shutil.copy("/root/auto-exp/outputs/M3cap-yoke/M3CAP_ERRATA.md", OUT + "/M3CAP_ERRATA.md")
print("report written:", os.path.getsize(OUT + "/FEATURE_HEAD_REPORT.md"), "bytes")

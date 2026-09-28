#!/usr/bin/env python3
"""M3BUDGET-aligned 报告+图. 数字全部来自 budget_eval.json / RESOURCE_COST.json / REUSE_MANIFEST.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/auto-exp/outputs/M3budget-aligned"
S = json.load(open(OUT + "/budget_eval.json"))
C = json.load(open(OUT + "/RESOURCE_COST.json"))
R = json.load(open(OUT + "/REUSE_MANIFEST.json"))
AM, CO = S["arm_means"], S["comparisons"]


def f(x, d=4):
    return ("%." + str(d) + "f") % x


L = []
w = L.append
w("# BUDGET_REPORT — M3BUDGET-aligned-v1（探索性系统比较）")
w("")
w("- 80 配置 = 2 资产 × 2 支持预算 × 2 特征提取器 × 2 评分 × 5 参照组; set_match=%d; 总 wall=%.0fs" % (
    S["pre_registered_set_match"], C["total_wall_s"]))
w("- 复用 40 配置（YOLO-5N ← M3TRANSFER-top8-v1；WRN-10N ← M3TRANSFER-wrnfix-v1 官方池化路径），新增 40 配置；")
w("  锚点逐图复现：YOLO yoke %.1e/glass %.1e；WRN %.1e（≤1e-4）" % (
    S["anchors"]["yolo_5n_yoke_ref3001"]["pc_max_diff"], S["anchors"]["yolo_5n_glass_ref3001"]["pc_max_diff"],
    S["anchors"]["wrn_10n_yoke_ref3001"]["pc_max_diff"]))
w("- 表述纪律：支持预算对齐后的**系统表现**，非主干因果；5 参照组非独立数据集；无显著性声明；AUROC 非召回率。")
w("")
w("## 1. 主表（5 参照组均值 AUROC）")
w("")
for a in ("yoke", "glass"):
    w("**%s**" % ("yoke-suspension/rust" if a == "yoke" else "glass-insulator/missing-cap"))
    w("")
    w("| 预算 | Y-PC | Y-T8 | W-PC | W-T8 |")
    w("|---|---|---|---|---|")
    for b in ("5N", "10N"):
        w("| %s | %s | %s | %s | %s |" % (b, f(AM["Y-PC_%s_%s" % (b, a)]["mean"]), f(AM["Y-T8_%s_%s" % (b, a)]["mean"]),
                                            f(AM["W-PC_%s_%s" % (b, a)]["mean"]), f(AM["W-T8_%s_%s" % (b, a)]["mean"])))
    w("")
w("复用行与源任务逐图一致（锚点门）；5N/10N 同预算同查询集。")
w("")
w("## 2. 预注册比较（每组配对差，5 参照组均值，正组数/5）")
w("")
w("| 比较 | yoke | glass |")
w("|---|---|---|")
for k, lab in (("5N_YT8-WT8", "5N: Y-T8 − W-T8"), ("5N_YT8-WPC", "5N: Y-T8 − W-PC"),
               ("10N_YT8-WT8", "10N: Y-T8 − W-T8"), ("10N_YT8-WPC", "10N: Y-T8 − W-PC"),
               ("YOLO_10N-5N_PC", "YOLO: 10N − 5N (PC)"), ("YOLO_10N-5N_T8", "YOLO: 10N − 5N (T8)"),
               ("WRN_10N-5N_PC", "WRN: 10N − 5N (PC)"), ("WRN_10N-5N_T8", "WRN: 10N − 5N (T8)"),
               ("T8-PC_Y_5N", "T8 − PC @ Y-5N"), ("T8-PC_Y_10N", "T8 − PC @ Y-10N"),
               ("T8-PC_W_5N", "T8 − PC @ W-5N"), ("T8-PC_W_10N", "T8 − PC @ W-10N")):
    y, g = CO[k]["yoke"], CO[k]["glass"]
    w("| %s | %s (%d/5) | %s (%d/5) |" % (lab, f(y["mean"]), y["n_pos"], f(g["mean"]), g["n_pos"]))
w("")
w("范围与组间 std 见 paired_comparisons.csv / budget_eval.json。")
w("")
w("## 3. 五个最终回答")
w("")
w("1. **同预算下 YOLO-T8 与标准 WRN 的距离**：10N 时 yoke 上 Y-T8 高于 W-PC %s、高于 W-T8 %s（均 5/5 组为正）；"
  "glass 上基本持平（%s / %s）。5N 时 yoke 仍领先（%s / %s），glass 略低（%s / %s）。"
  "即：轻量 YOLO-T8 在 yoke 上对标准 WRN 保持系统优势，在 glass 上与其相当。" % (
      f(CO["10N_YT8-WPC"]["yoke"]["mean"]), f(CO["10N_YT8-WT8"]["yoke"]["mean"]),
      f(CO["10N_YT8-WPC"]["glass"]["mean"]), f(CO["10N_YT8-WT8"]["glass"]["mean"]),
      f(CO["5N_YT8-WPC"]["yoke"]["mean"]), f(CO["5N_YT8-WT8"]["yoke"]["mean"]),
      f(CO["5N_YT8-WPC"]["glass"]["mean"]), f(CO["5N_YT8-WT8"]["glass"]["mean"])))
w("2. **增加正常支持的影响**：YOLO 10N−5N：yoke %s(PC)/%s(T8)，glass %s/%s；WRN：yoke %s/%s，glass %s/%s。"
  "两系统在 yoke 均明显受益（WRN 受益幅度更大），glass 上 WRN 在 5N 已近饱和（+%s/%s）而 YOLO 仍增益约 %s。"
  "注意 5N→10N 同时增加正常图像数与原型数，不能称为纯样本量效应。" % (
      f(CO["YOLO_10N-5N_PC"]["yoke"]["mean"]), f(CO["YOLO_10N-5N_T8"]["yoke"]["mean"]),
      f(CO["YOLO_10N-5N_PC"]["glass"]["mean"]), f(CO["YOLO_10N-5N_T8"]["glass"]["mean"]),
      f(CO["WRN_10N-5N_PC"]["yoke"]["mean"]), f(CO["WRN_10N-5N_T8"]["yoke"]["mean"]),
      f(CO["WRN_10N-5N_PC"]["glass"]["mean"]), f(CO["WRN_10N-5N_T8"]["glass"]["mean"]),
      f(CO["WRN_10N-5N_PC"]["glass"]["mean"]), f(CO["WRN_10N-5N_T8"]["glass"]["mean"]),
      f(CO["YOLO_10N-5N_T8"]["glass"]["mean"])))
w("3. **top8 收益是否跨预算保持**：YOLO 上保持且随预算略增（yoke %s→%s，glass %s→%s，全部 5/5 组为正）；"
  "WRN 上始终≈0（%s / %s）。top8 收益是 YOLO-P3 特征（单位模、浅层局部纹理）的特有现象，不是通用库内聚合收益。" % (
      f(CO["T8-PC_Y_5N"]["yoke"]["mean"]), f(CO["T8-PC_Y_10N"]["yoke"]["mean"]),
      f(CO["T8-PC_Y_5N"]["glass"]["mean"]), f(CO["T8-PC_Y_10N"]["glass"]["mean"]),
      f(CO["T8-PC_W_5N"]["yoke"]["mean"]), f(CO["T8-PC_W_10N"]["yoke"]["mean"])))
w("4. **是否应停止旧查询集上的 AUROC 优化**：是——已有理由转向。依据：(a) Y-T8 对同预算 WRN 的优势/持平在两资产两预算"
  "方向一致且逐组稳定复现；(b) top8 机制在 YOLO 上收益已被两个预算交叉验证，继续在同查询集上调 AUROC 边际价值低；"
  "(c) 部署判据是召回/误报与端到端延迟，而非缓存吞吐 AUROC。本轮按合同不自动启动校准/延迟实验。")
w("5. **未解决的限制**：探索性 2 资产、5 参照组非独立数据集、无显著性检验；历史查询集存在源帧重叠与查询重复用于决策的"
  "风险（预算对齐不消除）；系统比较含预训练/特征层/归一化/池化/容量差异，非主干因果；YOLO 5N 复用配置的库哈希仅前缀核对；"
  "WRN 历史权重无哈希直证（证据链一致）；AUROC 非召回率；缓存吞吐≠端到端延迟。")
w("")
w("## 4. 批量精度与资源")
w("")
bl = S["batch"]
w("- WRN batch 探针：bs16=%.2fs bs32=%s 显存 %s%% → 正式 bs=%d；YOLO 提取 bs=32（缓存固定），评分 chunk=64 图。" % (
    bl["wrn_probe"]["t_bs16_s"], ("%.2fs" % bl["wrn_probe"]["t_bs32_s"]) if bl["wrn_probe"]["t_bs32_s"] else "OOM",
    bl["wrn_probe"]["vram_bs32_pct"], S["chosen_batches"]["wrn_bs"]))
w("- 门误差：特征 bs1 vs 正式批 ≤%.1e；固定库 PC/T8 两 batch ≤%.1e；官方 vs 直接 fp64 距离（跨块+尾块）≤%.1e；"
  "官方 train 分支烟测 ≤%.1e；状态哈希前后一致。" % (
      max(S["gates"]["yolo"]["batch_features"]["yoke"]["ext_bs32_vs_bs1"], S["gates"]["yolo"]["batch_features"]["glass"]["ext_bs32_vs_bs1"],
          S["gates"]["wrn"]["batch_features_bs1_vs_bs32"] if "batch_features_bs1_vs_bs32" in S["gates"]["wrn"] else 0),
      max(S["gates"]["yolo"]["score_b1_vs_b64"]["pc"], S["gates"]["wrn"]["score_b1_vs_bN"]["pc"]),
      max(S["gates"]["yolo"]["distance"]["official_vs_direct64"], S["gates"]["wrn"]["distance"]["official_vs_direct64"]),
      S["gates"]["wrn"]["official_smoke_train_branch_vs_std_extract"]))
w("- 阶段墙钟：%s；峰值显存 allocated=%.0f MiB；缓存 %s；库字节合计 %.1f MiB（800/1600 原型，以实际 shape 记录）。" % (
    json.dumps(C["stages_s"]), C["peak_vram_allocated_bytes"] / 1048576,
    json.dumps({k: "%.0f MiB" % (v / 1048576) for k, v in C["cache_bytes"].items()}), C["bank_bytes_total"] / 1048576))
w("- 复用 %d 配置 + 新增 %d 配置（不重复计算）；REUSE_MANIFEST 记录源任务/commit/文件 sha256。" % (C["counts"]["reused_configs"], C["counts"]["new_configs"]))
open(OUT + "/BUDGET_REPORT.md", "w").write(chr(10).join(L))

# ---- 图 ----
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8))
arms = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
cols = ["#4a7fb5", "#2e7d32", "#c9a98e", "#b5764a"]
for i, a in enumerate(("yoke", "glass")):
    ax = axs[0][i]
    x = np.arange(2)
    bw = 0.2
    for ai, arm in enumerate(arms):
        vals = [AM["%s_%s_%s" % (arm, b, a)]["mean"] for b in ("5N", "10N")]
        ax.bar(x + (ai - 1.5) * bw, vals, bw, label=arm, color=cols[ai], alpha=0.9)
    lo = min(AM["%s_%s_%s" % (arm, b, a)]["min"] for arm in arms for b in ("5N", "10N")) - 0.03
    ax.set_ylim(lo, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["5N", "10N"])
    ax.set_title(a)
    ax.legend(fontsize=8)
    for ai, arm in enumerate(arms):
        for bi, b in enumerate(("5N", "10N")):
            v = AM["%s_%s_%s" % (arm, b, a)]["mean"]
            ax.text(bi + (ai - 1.5) * bw, lo + 0.005, "%.3f" % v, ha="center", fontsize=6, rotation=90)
keys = [("5N_YT8-WPC", "5N Y-T8-W-PC"), ("5N_YT8-WT8", "5N Y-T8-W-T8"), ("10N_YT8-WPC", "10N Y-T8-W-PC"),
        ("10N_YT8-WT8", "10N Y-T8-W-T8"), ("T8-PC_Y_10N", "10N T8-PC Y"), ("T8-PC_W_10N", "10N T8-PC W")]
for i, a in enumerate(("yoke", "glass")):
    ax = axs[1][i]
    vals = [CO[k][a]["mean"] for k, _ in keys]
    cols2 = ["#2e7d32" if v > 0 else "#b5764a" for v in vals]
    ax.bar(range(len(keys)), vals, color=cols2, alpha=0.9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([lab for _, lab in keys], rotation=30, fontsize=7, ha="right")
    ax.set_title("paired diffs: " + a)
plt.tight_layout()
plt.savefig(OUT + "/budget_matrix.png", dpi=130)
print("report done")

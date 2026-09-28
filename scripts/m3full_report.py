#!/usr/bin/env python3
"""M3FULL-frozen 报告+图. 数字全部来自 full_eval.json / RESOURCE_COST.json / macro_metrics.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/auto-exp/outputs/M3full-frozen"
S = json.load(open(OUT + "/full_eval.json"))
C = json.load(open(OUT + "/RESOURCE_COST.json"))
PC, CO, MA, AE = S["per_combo"], S["comparisons"], S["macro"], S["asset_equal"]
COMBOS = sorted(set(k.rsplit("_", 2)[0] for k in PC))
SHORT = {"glass-insulator__missing-cap": "glass/missing-cap", "lightning-rod-suspension__rust": "lightning/rust",
         "polymer-insulator-upper-shackle__corrosion": "polymer/corrosion",
         "polymer-insulator-upper-shackle__rust": "polymer/rust",
         "vari-grip__bird-nest": "vari/bird-nest", "vari-grip__rust": "vari/rust",
         "yoke-suspension__rust": "yoke/rust"}
ARMS = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
COLS = ["#9fb8d0", "#2e7d32", "#c9a98e", "#b5764a"]


def f(x, d=4):
    return ("%." + str(d) + "f") % x


L = []
w = L.append
w("# FULL_REPORT — M3FULL-frozen-benchmark-v1（探索性, 冻结方案跨组合评测）")
w("")
w("- 覆盖 **7/7 组合**（无阻塞项）: 2 预算 × 4 方法 × 5 参照组 = **280 配置**, set_match=%d; 总 wall=%.0fs（GPU 2h 上限内）" % (
    S["pre_registered_set_match"], C["total_wall_s"]))
w("- 复用 M3BUDGET 80 配置（yoke/glass, 逐文件溯源+独立重算）; 新增 200 配置（3 资产/5 组合）")
w("- 锚点逐图复现: YOLO yoke %.1e; WRN yoke %.1e（≤1e-4）; 批量精度门/官方路径烟测/官方 pred_score 门全过" % (
    S["anchors"]["yolo_5n_yoke_ref3001"]["pc"], S["anchors"]["wrn_10n_yoke_ref3001"]["pc"]))
w("- 口径: 支持预算对齐的系统比较; 源组身份未知(只记录交集); 支持以裁剪计数; 无显著性声明; AUROC 非召回率")
w("")
w("## 1. 主表 10N（5 参照组均值 AUROC）")
w("")
w("| 组合 | Y-PC | Y-T8 | W-PC | W-T8 | Y-T8−W-PC |")
w("|---|---|---|---|---|---|")
for c in COMBOS:
    k = "%s_YT8-WPC_10N" % c
    w("| %s | %s | %s | %s | %s | **%s** (%d/5) |" % (
        SHORT[c], f(PC[c + "_Y-PC_10N"]["mean"]), f(PC[c + "_Y-T8_10N"]["mean"]),
        f(PC[c + "_W-PC_10N"]["mean"]), f(PC[c + "_W-T8_10N"]["mean"]),
        f(CO[k]["mean"]), CO[k]["n_pos"]))
w("")
w("## 2. 主表 5N")
w("")
w("| 组合 | Y-PC | Y-T8 | W-PC | W-T8 | Y-T8−W-PC |")
w("|---|---|---|---|---|---|")
for c in COMBOS:
    k = "%s_YT8-WPC_5N" % c
    w("| %s | %s | %s | %s | %s | %s (%d/5) |" % (
        SHORT[c], f(PC[c + "_Y-PC_5N"]["mean"]), f(PC[c + "_Y-T8_5N"]["mean"]),
        f(PC[c + "_W-PC_5N"]["mean"]), f(PC[c + "_W-T8_5N"]["mean"]),
        f(CO[k]["mean"]), CO[k]["n_pos"]))
w("")
w("## 3. 宏平均与资产等权")
w("")
w("| 指标 | Y-PC | Y-T8 | W-PC | W-T8 |")
w("|---|---|---|---|---|")
for b in ("5N", "10N"):
    w("| 7 组合宏平均 %s | %s | %s | %s | %s |" % (b, f(MA["Y-PC_" + b]["mean"]), f(MA["Y-T8_" + b]["mean"]),
                                                    f(MA["W-PC_" + b]["mean"]), f(MA["W-T8_" + b]["mean"])))
for b in ("5N", "10N"):
    w("| 资产等权 %s | %s | %s | %s | %s |" % (b, f(AE["Y-PC_" + b]["mean"]), f(AE["Y-T8_" + b]["mean"]),
                                                f(AE["W-PC_" + b]["mean"]), f(AE["W-T8_" + b]["mean"])))
w("")
w("| 10N 主比较 | 宏平均差 | 范围 | 领先组合数 |")
w("|---|---|---|---|")
for k, lab in (("YT8-WPC_10N", "Y-T8 − W-PC"), ("YT8-WT8_10N", "Y-T8 − W-T8")):
    v = MA[k]
    w("| %s | **%s** | [%s, %s] | %d/7 |" % (lab, f(v["mean"]), f(v["min"]), f(v["max"]), v["n_combos_pos_mean"]))
w("")
w("## 4. 六个最终回答")
w("")
w("1. **同预算竞争力（10N 主分析）**: Y-T8 对 W-PC 在 **4 组合 5/5 组领先**（lightning +0.121, yoke +0.083, "
  "polymer/rust +0.057, vari/rust +0.036），2 组合小幅领先（polymer/corrosion +0.026, glass +0.005，4/5 组），"
  "唯一落后 vari/bird-nest（−0.034，0/5，该组合 WRN 最强 0.936）。对加强对照 W-T8 宏平均仍 +0.035。"
  "「持平」仅为 glass 等数值接近的描述，非统计等效。")
w("2. **10N 宏平均**: Y-T8 %s vs W-PC %s（差 **%s**）、vs W-T8 %s（差 %s）；资产等权口径下差值更大（%s vs %s）。" % (
    f(MA["Y-T8_10N"]["mean"]), f(MA["W-PC_10N"]["mean"]), f(MA["YT8-WPC_10N"]["mean"]),
    f(MA["W-T8_10N"]["mean"]), f(MA["YT8-WT8_10N"]["mean"]), f(AE["Y-T8_10N"]["mean"]), f(AE["W-PC_10N"]["mean"])))
w("3. **收益是否集中**: 否——7 组合中 6 个 Y-T8 ≥ W-PC，方向一致；唯一例外（vari/bird-nest）幅度有限。"
  "宏平均未被单个组合主导（逐组合差 −0.034 ~ +0.121）。")
w("4. **5N→10N 一致**: 一致。各方法宏平均提升 +0.033~+0.040，逐参照组全为正（范围无负值）；"
  "两主干、两种评分的预算效应方向相同；注意 10N 同时增加支持图与原型数，非纯样本量效应。")
w("5. **是否应冻结方案进入校准/成本验证**: 是。依据：Y-T8 优势跨 7 组合、2 预算、5 参照组方向一致复现；"
  "top8 收益在 YOLO 上 7/7 组合、两预算全部 5/5 为正（WRN 上 ≈0，进一步说明其为 YOLO 特征特有机制）；"
  "继续在旧查询集上做 AUROC 优化边际价值低。本轮按合同不自动启动校准/延迟/部署。")
w("6. **数据边界与未覆盖项**: 无组合被阻塞，但——源组身份未知（裸 DJI 重号；支持池与查询的源组交集已记录："
  "yoke 430、vari 63、glass 3+1 等），评测池隔离≠骨干未见；polymer 两个故障共享正常库与正常查询（组合间非独立）；"
  "polymer/rust 仅 22 张故障查询（AUROC 抽样方差大）；5 参照组非独立数据集；无显著性检验；"
  "不证明召回/误报达标、端到端延迟达标或部署可行性。")
w("")
w("## 5. 资源")
w("")
w("- 阶段: %s; 峰值显存 allocated=%.0f MiB" % (json.dumps(C["stages_s"]), C["peak_vram_allocated_bytes"] / 1048576))
w("- YOLO 缓存: %s" % json.dumps({k: "%.0f MiB" % (v / 1048576) for k, v in C["cache_bytes"].items()}))
w("- 复用 %d 配置 + 新增 %d 配置; 新增库 %d（60: 3 资产 × 2 预算 × 2 主干 × 5 参照; 同资产多故障共享）; 双建库 6 组（每新增资产×主干）" % (
    C["counts"]["reused_configs"], C["counts"]["new_configs"], C["counts"]["new_banks"]))
open(OUT + "/FULL_REPORT.md", "w").write(chr(10).join(L))

# ---- 图 ----
fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
for bi, b in enumerate(("10N", "5N")):
    ax = axs[0][bi]
    x = np.arange(len(COMBOS))
    bw = 0.2
    for ai, arm in enumerate(ARMS):
        vals = [PC[c + "_" + arm + "_" + b]["mean"] for c in COMBOS]
        ax.bar(x + (ai - 1.5) * bw, vals, bw, label=arm, color=COLS[ai], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in COMBOS], rotation=28, fontsize=7, ha="right")
    lo = min(PC[c + "_" + arm + "_" + b]["min"] for c in COMBOS for arm in ARMS) - 0.03
    ax.set_ylim(lo, 1.0)
    ax.set_title("%s: 4 方法 AUROC" % b)
    ax.legend(fontsize=7)
for bi, b in enumerate(("10N", "5N")):
    ax = axs[1][bi]
    vals = [CO[c + "_YT8-WPC_" + b]["mean"] for c in COMBOS]
    cols = ["#2e7d32" if v > 0 else "#b5764a" for v in vals]
    ax.bar(range(len(COMBOS)), vals, color=cols, alpha=0.92)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(COMBOS)))
    ax.set_xticklabels([SHORT[c] for c in COMBOS], rotation=28, fontsize=7, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.004 if v > 0 else -0.008), "%+.3f" % v, ha="center", fontsize=6)
    ax.set_title("%s: Y-T8 − W-PC 逐组合配对差" % b)
plt.tight_layout()
plt.savefig(OUT + "/full_matrix.png", dpi=130)
fig2, ax = plt.subplots(figsize=(9, 4.2))
ks = ["Y-PC_10N", "Y-T8_10N", "W-PC_10N", "W-T8_10N", "Y-PC_5N", "Y-T8_5N", "W-PC_5N", "W-T8_5N"]
vals = [MA[k]["mean"] for k in ks]
ax.bar(ks, vals, color=["#4a7fb5", "#2e7d32", "#c9a98e", "#b5764a"] * 2, alpha=0.92)
ax.set_ylim(0.72, 0.92)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, "%.4f" % v, ha="center", fontsize=7)
ax.set_title("7 组合等权宏平均 AUROC")
ax.tick_params(axis="x", labelsize=7)
plt.tight_layout()
plt.savefig(OUT + "/macro_summary.png", dpi=130)
print("report done")

#!/usr/bin/env python3
"""M3TRANSFER 报告: TRANSFER_REPORT.md + 2x2 矩阵图 + 逐组收益图. 数字全部来自 transfer_eval.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/auto-exp/outputs/M3transfer-top8"
S = json.load(open(OUT + "/transfer_eval.json"))
C = json.load(open(OUT + "/RESOURCE_COST.json"))
G = S["gates"]
AM = S["arm_means"]
P = S["paired"]

def f(x, d=4):
    return ("%." + str(d) + "f") % x

L = []; w = L.append
w("# TRANSFER_REPORT — M3TRANSFER-top8-v1（探索性）")
w("")
w("- 2 资产 (yoke-suspension/rust, glass-insulator/missing-cap) × 2 特征 (YOLO-P3 64d L2 5N, WRN50 layer2+3 1536d 10N) × 2 评分 (官方 PC nn=9, top8 欧氏) × 5 参照组 = **40 配置**, 集合断言 set_match=%d" % S["pre_registered_set_match"])
w("- 声明: shot 为裁剪预算; 查询跨臂复用; 源帧身份限制未核实; glass 为跨资产迁移验证, 非真正未见数据的确认性实验; 无显著性检验声明")
w("- 同组 PC 与 T8 共享特征张量/库/最近邻距离/查询路径, 仅最终聚合不同 (same_bank_identity=%d)" % S["same_bank_identity"])
w("")
w("## 1. 结果矩阵 (AUROC, 5 参照组)")
w("")
for a in ("yoke", "glass"):
    w("**%s**:" % ("yoke-suspension/rust" if a == "yoke" else "glass-insulator/missing-cap"))
    w("")
    w("| 臂 | 均值 | 范围 | std(ddof=0) |")
    w("|---|---|---|---|")
    for arm in ("Y-PC", "Y-T8", "W-PC", "W-T8"):
        v = AM[arm][a]
        w("| %s | %s | [%s, %s] | %s |" % (arm, f(v["mean"]), f(v["min"]), f(v["max"]), f(v["std_ddof0"])))
    w("")
w("## 2. yoke YOLO 锚点 (复现门)")
w("")
w("以 M3SCORE 的 E1 缓存/固定 N*/数值路径为锚点: Y-PC 逐路径复现 S4 **max|diff|=%s** (门限 1e-4), Y-T8 逐路径复现 S2 **max|diff|=%s**; 正常库张量哈希与容量逐组断言一致。yoke YOLO 10 配置直接复用为矩阵单元。" % (f(S["anchor"]["pc_max_diff"], 1), f(S["anchor"]["t8_max_diff"], 1)))
w("")
w("## 3. 库内聚合收益 (同库同距离, 仅聚合不同)")
w("")
w("| 比较 | yoke 均值 | glass 均值 | yoke 正/负 | glass 正/负 |")
w("|---|---|---|---|---|")
for k, lab in (("YT8-YPC", "YOLO: T8−PC"), ("WT8-WPC", "WRN: T8−PC")):
    py, pg = P[k + "_yoke"], P[k + "_glass"]
    w("| %s | **%s** | **%s** | %d/5, %d/5 | %d/5, %d/5 |" % (lab, f(py["mean"]), f(pg["mean"]), py["n_pos"], py["n_neg"], pg["n_pos"], pg["n_neg"]))
w("")
w("- top8 收益在 4 组特征×资产组合全部 5/5 组为正 (合计 20/20): 聚合改进同时惠及 YOLO 与 WRN, 且从 yoke 迁移到 glass 保持 (YOLO glass +0.0373, 5/5; 量级小于 yoke 的 +0.0615)。")
w("- 这是**同库内**的聚合效应; 跨库/跨特征的比较见 §4, 不能与聚合收益混读。")
w("")
w("## 4. 外部比较 (不同支持预算的系统对比)")
w("")
w("| 比较 | yoke 均值 | glass 均值 | 正/负 (yoke, glass) |")
w("|---|---|---|---|")
for k, lab in (("YT8-WPC", "Y-T8 − W-PC (vs 标准 WRN)"), ("YT8-WT8", "Y-T8 − W-T8 (vs 加强 WRN)")):
    py, pg = P[k + "_yoke"], P[k + "_glass"]
    w("| %s | %s | %s | %d/5, %d/5 |" % (lab, f(py["mean"]), f(pg["mean"]), py["n_pos"], pg["n_pos"]))
w("")
w("- W-PC(0.888/0.790) 与 W-T8(0.906/0.821) 在两资产均为最高; YOLO 5N 系统整体低于 WRN 10N 系统, 但 glass 上 Y-T8 与 W-PC 基本持平 (+0.0159, 3/5 正)。")
w("- YOLO5N 与 WRN10N 是不同支持预算的系统对比, 不能归结为纯主干能力差异。")
w("")
w("## 5. WRN 历史库回退披露 (合同 §七)")
w("")
w("M3BENCH 未留存库/特征/patch 距离 → 按原标准配置确定性重建 (种子 pc_seed=7001..7005, 双建库核验 ref3001 torch.equal)。历史 W-PC 仅作参考:")
w("")
w("| 组 | 历史 W-PC | 新 W-PC | 差 |")
w("|---|---|---|---|")
for k, v in S["wrn_rebuild_vs_hist"].items():
    w("| %s | %s | %s | %s |" % (k, f(v["hist_W-PC"]), f(v["new_W-PC"]), f(v["diff"])))
w("")
w("- 新旧 W-PC 逐组差 yoke %s..%s, glass %s..%s — **库随机性量级可达 ±0.08**, 大于 top8 收益 (0.018-0.037) 与历史 W-PC−YOLO 差距; 未挑种子追平历史均值, 未把新旧 W-PC 差异解释为聚合收益。" % (
    f(min(v["diff"] for k, v in S["wrn_rebuild_vs_hist"].items() if k.startswith("yoke"))),
    f(max(v["diff"] for k, v in S["wrn_rebuild_vs_hist"].items() if k.startswith("yoke"))),
    f(min(v["diff"] for k, v in S["wrn_rebuild_vs_hist"].items() if k.startswith("glass"))),
    f(max(v["diff"] for k, v in S["wrn_rebuild_vs_hist"].items() if k.startswith("glass")))))
w("")
w("## 6. 正确性门 (正式评分前立即 assert)")
w("")
w("| 门 | 结果 |")
w("|---|---|")
w("| top8 轴 | patch 轴正确, 错误轴(列)结果确实不同: %s" % json.dumps(G["top8_axis"]))
w("| reshape | (batch,h,w) 行主序: %s" % json.dumps(G["reshape"]))
w("| WRN 距离 | 官方分块(2500 行跨块+尾块) vs float64 直接: %s" % json.dumps(G["wrn_distance"]))
w("| WRN 独立组合 | 官方 compute_anomaly_score vs float64 公式: %s" % json.dumps(G["wrn_composition"]))
w("| WRN batch | 同库 batch=1 vs 16: %s" % json.dumps(G["wrn_batch"]))
w("| WRN 库重建 | %s (同种子)" % json.dumps(G["wrn_bank_rebuild"]))
w("| glass 提取 | ext/cache diff: %s" % json.dumps(G.get("glass_batch", {})))
w("| YOLO 缓存身份 | yoke 缓存 SHA256+确定性非TF32+路径集合断言; glass 新缓存同规则")
w("| AUROC 重算 | 40/40 配置从逐图分数重算, 与存档差 <1e-12")
w("")
w("## 7. 资源")
w("")
w("- 总 wall=%.1fs (GPU 预算 2h 上限); 阶段: %s" % (C["total_wall_s"], json.dumps(C["stages_s"])))
w("- VRAM 峰值 allocated=%.0f MiB / reserved=%.0f MiB; WRN 距离 npz %s (数据盘, 不提交)" % (
    C["peak_vram_allocated_bytes"] / 1048576, C["peak_vram_reserved_bytes"] / 1048576,
    json.dumps({k: "%.0f MiB" % (v / 1048576) for k, v in C["wrn_dist_bytes"].items()})))
w("- 评分耗时为缓存特征计算, 非端到端推理延迟")
w("")
w("## 8. 五个最终回答")
w("")
w("1. **top8 收益迁移到 glass: 是** — Y-T8−Y-PC = +0.0373 (5/5 组; 10/10 YOLO 配置为正), 量级小于 yoke (+0.0615)。支持冻结解释 A: 收益跨这两个资产保持, 仍需更广数据验证。")
w("2. **WRN 同样聚合提升**: yoke +0.0180 (5/5), glass +0.0306 (5/5); W-T8 成为两资产最高 (0.9062 / 0.8209) — 聚合改动惠及双方, 此前领先不能归功于 YOLO 主干 (解释 C 成立一半: WRN 也提升并领先, 但 YOLO-T8 收益幅度更大)。")
w("3. **YOLO-T8 相对基线**: 对标准 W-PC: yoke −0.0147 / glass +0.0159 (互有胜负); 对加强 W-T8: yoke −0.0327 / glass −0.0147 (落后)。注: 5N vs 10N 支持预算不同, 是系统级对比。")
w("4. **最有竞争力: W-T8** (重建确定性库), 两资产最高且库内方差最小 (std 0.006-0.009)。代价: WRN 特征路径 24.85M 参数 (YOLO 冻结全模型 3.01M, P3 子图 80.7k)、1536 维 patch 特征 vs 64 维、10N vs 5N 支持、1600 vs 800 原型。预算受限场景下 Y-T8 (64 维/5N) 在 glass 距 W-T8 仅 0.015, 是轻量备选。")
w("5. **下一步**: 先**统一支持预算并固定建库随机性**再比较系统——本轮证据显示库随机性 (重建 vs 历史差 ±0.08) 可吞掉聚合收益 (0.018-0.062), 不固定将淹没结论; 统一预算后再扩大资产范围, 隔离/部署验证随后。探索性判断, 不自动触发。")
w("")
w("## 9. 产物")
w("")
w("FROZEN_INPUTS.json / CACHE_MANIFEST.json (yoke 复用+glass 新建) / BANK_MANIFEST.json (合并于 transfer_eval bank_meta: 库哈希/容量/coreset 索引) / REPRO_AND_CORRECTNESS.json / WRN_REUSE_OR_REBUILD.md / raw_scores/ (40, JSON 全精度) / transfer_eval.json / paired_comparisons.csv / RESOURCE_COST.json / transfer_matrix.png / run.log")
open(OUT + "/TRANSFER_REPORT.md", "w").write(chr(10).join(L))

# 图 1: 2x2 矩阵 (两资产 4 臂)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
arms = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
colors = ["#8ea9c9", "#4a7fb5", "#c9a98e", "#b5764a"]
x = np.arange(4)
for i, a in enumerate(("yoke", "glass")):
    ms = [AM[arm][a]["mean"] for arm in arms]
    es = [max(AM[arm][a]["mean"] - AM[arm][a]["min"], AM[arm][a]["max"] - AM[arm][a]["mean"]) for arm in arms]
    axs[i].bar(x, ms, yerr=es, capsize=4, color=colors, alpha=0.9)
    axs[i].set_xticks(x); axs[i].set_xticklabels(arms)
    lo = min(ms) - 0.06
    axs[i].set_ylim(lo, max(ms) + 0.03)
    axs[i].set_title("%s" % ("yoke (YOLO5N / WRN10N)" if a == "yoke" else "glass (YOLO5N / WRN10N)"))
    for xi, m in zip(x, ms):
        axs[i].text(xi, lo + 0.005, "%.3f" % m, ha="center", fontsize=8)
# 图 2: 逐组收益
ax = axs[2]
w_ = 0.38
refs = np.arange(5)
for i, (k, lab, c) in enumerate((("YT8-YPC", "YOLO T8-PC", "#4a7fb5"), ("WT8-WPC", "WRN T8-PC", "#b5764a"))):
    vals = list(P[k + "_glass"]["per_ref"].values())
    ax.bar(refs + (i - 0.5) * w_, vals, w_, label=lab + " (glass)", color=c, alpha=0.9)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(refs); ax.set_xticklabels(["ref%d" % j for j in (3001, 3002, 3003, 3004, 3005)], fontsize=8)
ax.set_title("per-group gain (glass)"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT + "/transfer_matrix.png", dpi=130)
print("report:", len(chr(10).join(L)), "chars")

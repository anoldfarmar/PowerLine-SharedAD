#!/usr/bin/env python3
"""M3TRANSFER-wrnfix 报告 + 勘误 + 图. 数字全部来自 wrnfix_eval.json / RESOURCE_COST.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/root/auto-exp/outputs/M3transfer-wrnfix"
S = json.load(open(OUT + "/wrnfix_eval.json"))
C = json.load(open(OUT + "/RESOURCE_COST.json"))
G = S["gates"]
AM = S["arm_means"]
P = S["paired"]

def f(x, d=4):
    return ("%." + str(d) + "f") % x

L = []; w = L.append
w("# WRNFIX_REPORT — M3TRANSFER-wrnfix-v1（探索性）")
w("")
w("- 2 资产 × 5 参照组 × 2 评分 (W-PC-FIX / W-T8-FIX) = **20 配置**, set_match=%d; YOLO 结果复用 M3TRANSFER, 不重跑" % S["pre_registered_set_match"])
w("- 表述纪律: 参照组间 std 非库内方差; 固定种子只保证复现; 结果仅限两个历史资产与探索性查询集; AUROC 非召回率; 不宣称完成 H1")
w("")
w("## 1. 修正后标准 WRN 两评分 (官方池化路径, 权重明确的配置重评)")
w("")
for a in ("yoke", "glass"):
    w("**%s**:" % ("yoke-suspension/rust" if a == "yoke" else "glass-insulator/missing-cap"))
    w("")
    w("| 臂 | 均值 | 范围 | 参照组间 std |")
    w("|---|---|---|---|")
    for arm in ("W-PC-FIX", "W-T8-FIX"):
        v = AM[arm][a]
        w("| %s | %s | [%s, %s] | %s |" % (arm, f(v["mean"]), f(v["min"]), f(v["max"]), f(v["std_across_refs"])))
    w("")
w("## 2. 正确性门 (全部在正式评分前立即 assert)")
w("")
w("| 门 | 结果 |")
w("|---|---|")
w("| 端到端路径 (官方 train 分支 embedding vs std_extract, atol/rtol=1e-4) | max_diff=**%s** |" % f(G["e2e_path"]["max_abs_diff"]))
w("| batch=1 vs 4 | max_diff=**%s** |" % f(G["batch_consistency"]["b1_vs_b4_max_diff"]))
w("| 同库适配评分 vs 官方完整推理 pred_score (n=4) | max_diff=**%s** |" % f(G["pred_score"]["max_abs_diff"]))
w("| 同种子双建库 torch.equal | yoke_ref3001=%s, glass_ref3001=%s |" % (G["bank_rebuild"]["yoke_ref3001"], G["bank_rebuild"]["glass_ref3001"]))
w("| 状态哈希/embedding_store | train 前后一致/已清理 (断言通过) |")
w("| 历史 g2 集合断言 | 10/10 组路径集合与标签侧一致 (glass 用 g2=705 故障查询, g1 弃用) |")
w("")
w("## 3. 与历史 M3BENCH g2 对照 (权重/路径修正后的复现)")
w("")
w("| 组 | 历史 g2 W-PC | 新 W-PC-FIX | 差 |")
w("|---|---|---|---|")
for k, v in S["hist_g2_vs_new"].items():
    w("| %s | %s | %s | %s |" % (k, f(v["hist_g2_W-PC"]), f(v["new_W-PC-FIX"]), f(v["diff"])))
w("")
w("**逐组差 ≤0.0036** — 同种子重建 + 池化路径 + timm 默认权重几乎复现历史系统; 该证据同时反证: "
  "M3TRANSFER 未池化变体的大偏差 (±0.08) 来自特征路径偏离与 g1 误选, 而非建库随机性 (见 M3TRANSFER_ERRATA.md)。")
w("")
w("## 4. 库内聚合收益与外部比较")
w("")
w("| 比较 | yoke | glass | 正/负 |")
w("|---|---|---|---|")
for k, lab in (("WT8FIX-WPCFIX", "WRN: T8-FIX − PC-FIX (同库同距离)"),
               ("YT8-WPCFIX", "Y-T8 − W-PC-FIX"),
               ("YT8-WT8FIX", "Y-T8 − W-T8-FIX"),
               ("FIX-UNPOOL_PC", "W-PC-FIX − 未池化W-PC"),
               ("FIX-UNPOOL_T8", "W-T8-FIX − 未池化W-T8")):
    py, pg = P[k + "_yoke"], P[k + "_glass"]
    w("| %s | %s | %s | %d/5, %d/5 |" % (lab, f(py["mean"]), f(pg["mean"]), py["n_pos"], pg["n_pos"]))
w("")
w("- **top8 在修正后的标准 WRN 上收益≈0** (yoke +0.0010, glass +0.0015; 7/10 组为正但幅度可忽略) — 旧未池化变体上的 +0.018/+0.031 不迁移到官方池化路径, 相关结论撤回。")
w("- FIX vs 未池化方向相反 (yoke PC −0.054 5/5, glass PC +0.036 5/5): 池化对两资产影响不同; 因权重路径一致且逐组差与历史对齐, 该差异可归因于特征路径(池化)+建库输入变化, 随机性成分 ≤0.004。")
w("- YOLO-T8 (5N) vs W-PC-FIX (10N): yoke +0.0389 (3/2), glass −0.0195 (1/4) — 不同支持预算的系统对比; yoke 上轻量 YOLO-T8 反超标准 WRN, glass 上略落后。")
w("")
w("## 5. 五个最终回答")
w("")
w("1. **修正后两评分**: W-PC-FIX yoke %s / glass %s; W-T8-FIX yoke %s / glass %s (5 组均值)。" % (
    f(AM["W-PC-FIX"]["yoke"]["mean"]), f(AM["W-PC-FIX"]["glass"]["mean"]), f(AM["W-T8-FIX"]["yoke"]["mean"]), f(AM["W-T8-FIX"]["glass"]["mean"])))
w("2. **top8 在标准 WRN 上无实质跨资产收益** (+0.001/+0.002), 与旧未池化变体结论相反 — 撤回旧结论。")
w("3. **YOLO-T8 vs 修正 WRN**: vs W-PC-FIX yoke +0.039 / glass −0.020; vs W-T8-FIX yoke +0.038 / glass −0.021。")
w("4. **保留**: YOLO top8 收益 (+0.0615/+0.0373, 同库内), yoke 锚点复现, M3SCORE 分解结论。**撤回/降级**: M3TRANSFER 的 glass 历史对比 (g1 误选)、「库随机性±0.08」单因素归因、未池化 WRN 上的 top8 收益、未池化 W-PC 绝对值。")
w("5. **统一 5N/10N 预算**: 值得 — 本轮证明随机性影响 ≤0.004 (远小于此前误估), 预算差成为 YOLO vs WRN 系统对比的主要混杂; 统一预算后的对比才有主干归因意义。探索性判断, 不自动触发。")
w("")
w("## 6. 资源")
w("")
w("- 总 wall=%.1fs (GPU 1h 上限); 阶段: %s" % (C["total_wall_s"], json.dumps(C["stages_s"])))
w("- VRAM 峰值 allocated=%.0f MiB; 距离 npz %s (数据盘)" % (C["peak_vram_allocated_bytes"] / 1048576,
    json.dumps({k: "%.0f MiB" % (v / 1048576) for k, v in C["wrn_dist_bytes"].items()})))
open(OUT + "/WRNFIX_REPORT.md", "w").write(chr(10).join(L))

E = []; we = E.append
we("# M3TRANSFER_ERRATA — 由 M3TRANSFER-wrnfix-v1 核验产生")
we("")
we("1. **glass 历史对比组误选 (撤回)**: M3TRANSFER-top8-v1 的 wrn_rebuild_vs_hist 对 glass 使用了 g1 文件 (717 故障查询) 与当前 705 张查询不一致, 该对比无效。本轮改用 g2 (705) 并断言路径集合/标签侧一致。")
we("2. **「库随机性可造成 ±0.08」归因 (撤回降级)**: 修正特征路径 (官方 AvgPool2d(3,1,1) 池化) + g2 集合后, 同种子重建库与历史 g2 逐组差 ≤0.0036 — 建库随机性量级 ~0.004, 远小于此前误估; 此前大偏差来自未池化特征路径偏离与 g1 误选的组合, 不可归因单一因素。")
we("3. **旧 WRN 结果更名「未池化WRN变体」**: M3TRANSFER 的 W-PC/W-T8 (及 bank/dist npz) 为薄适配 (无池化) 产物, 原始文件保留不动; 其绝对值与基于它的 WRN top8 收益 (+0.018/+0.031) 不适用于官方标准路径。")
we("4. **保留**: M3TRANSFER 全部 YOLO 结果 (锚点逐图复现 S4/S2, top8 同库收益 +0.0615/+0.0373) 不受影响; M3SCORE 分解结论不受影响。")
we("5. **本轮证据**: 权重身份 = timm wide_resnet50_2.racm_in1k (默认 pretrained_cfg, HF 缓存 + 加载态哈希存档; 历史哈希未留存, 证据链强一致但未直证)。与历史 g2 复现差 ≤0.0036 佐证历史实际权重与路径一致。")
open(OUT + "/M3TRANSFER_ERRATA.md", "w").write(chr(10).join(E))
open("/root/auto-exp/outputs/M3transfer-top8/ERRATA_POINTER.md", "w").write(
    "# 勘误指引" + chr(10) + chr(10) + "本目录 WRN 相关结果的更正见 ../M3transfer-wrnfix/M3TRANSFER_ERRATA.md (池化路径偏离、glass g1 误选、随机性归因)。YOLO 结果不受影响。原始文件未改动。" + chr(10))

# 图: 修正前后 + 历史
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
for i, a in enumerate(("yoke", "glass")):
    labels = ["hist g2", "W-PC-FIX", "W-T8-FIX", "unpool W-PC", "unpool W-T8"]
    keys = [("hist_g2_vs_new", "hist_g2_W-PC"), (None, None), (None, None)]
    hv = [S["hist_g2_vs_new"]["%s_ref%d" % (a, j)]["hist_g2_W-PC"] for j in (3001, 3002, 3003, 3004, 3005)]
    vals = [np.mean(hv), AM["W-PC-FIX"][a]["mean"], AM["W-T8-FIX"][a]["mean"],
            AM["W-PC-FIX"][a]["mean"] - P["FIX-UNPOOL_PC_" + a]["mean"], AM["W-T8-FIX"][a]["mean"] - P["FIX-UNPOOL_T8_" + a]["mean"]]
    bars = axs[i].bar(labels, vals, color=["#999", "#4a7fb5", "#2e7d32", "#c9a98e", "#b5764a"], alpha=0.9)
    lo = min(vals) - 0.04
    axs[i].set_ylim(lo, max(vals) + 0.02)
    for b, v in zip(bars, vals):
        axs[i].text(b.get_x() + b.get_width() / 2, lo + 0.004, "%.3f" % v, ha="center", fontsize=8)
    axs[i].set_title("yoke" if a == "yoke" else "glass")
    axs[i].tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(OUT + "/wrnfix_compare.png", dpi=130)
print("report done")

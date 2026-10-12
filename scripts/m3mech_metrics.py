#!/usr/bin/env python3
"""M3MECH 指标汇总: metrics.json + paired_comparisons.csv.
G = AUROC(top8) - AUROC(pure_max); 逐参照配对; 5 参照非独立."""
import csv, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12"
COMBOS = {"yoke-suspension": "yoke-suspension__rust", "vari-grip": "vari-grip__bird-nest"}
ARMS = ["Y-RAW", "Y-POOL", "W-STD", "W-NOPOOL"]
REFS = [3001, 3002, 3003, 3004, 3005]
TOPKS = [4, 8, 16, 32]


def stats(vs):
    v = np.array(vs, np.float64)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "std": float(v.std(ddof=0))}


def load(arm, a, j, lname=None):
    sub = "phaseA" if lname is None else "phaseB"
    name = "%s_%s_ref%d.json" % (COMBOS[a], lname if lname else arm, j)
    r = json.load(open(os.path.join(OUT, "raw_scores", sub, name)))
    dv = r["defect"]
    nv = r["normal"]
    return r, dv, nv


def auc(dv, nv, key):
    u, _ = D.auroc_avg_rank(np.array([v[key] for v in dv.values()], np.float64),
                             np.array([v[key] for v in nv.values()], np.float64))
    return float(u)


def main():
    gates = json.load(open(os.path.join(OUT, "repro_gates.json")))
    A = {}   # per arm-asset-ref-scoring AUROC
    dist_eff = {}  # normal/fault 分布效应
    for arm in ARMS:
        for a in COMBOS:
            for j in REFS:
                r, dv, nv = load(arm, a, j)
                keys = list(dv[list(dv)[0]].keys())
                for k in keys:
                    A[(arm, a, j, k)] = auc(dv, nv, k)
                # 尖峰/分布效应: max/top8 比率与正常分位数
                nmax = np.array([v["pure_max"] for v in nv.values()])
                nt8 = np.array([v["top8"] for v in nv.values()])
                dmax = np.array([v["pure_max"] for v in dv.values()])
                dt8 = np.array([v["top8"] for v in dv.values()])
                dist_eff[(arm, a, j)] = {
                    "normal_spikes_mean_over_top8mean": float((nmax / np.maximum(nt8, 1e-12)).mean()),
                    "normal_max_p995": float(np.percentile(nmax, 99.5)),
                    "fault_max_p005": float(np.percentile(dmax, 0.5)),
                    "normal_top8_p995": float(np.percentile(nt8, 99.5)),
                    "fault_top8_p005": float(np.percentile(dt8, 0.5))}
    B = {}
    for a in COMBOS:
        for ln in ("p3", "p4", "p5"):
            for j in REFS:
                r, dv, nv = load(None, a, j, lname=ln)
                keys = list(dv[list(dv)[0]].keys())
                for k in keys:
                    B[(ln, a, j, k)] = auc(dv, nv, k)
    # G 与配对比较
    paired_rows = []
    G = {}
    for arm in ARMS:
        for a in COMBOS:
            gs = [A[(arm, a, j, "top8")] - A[(arm, a, j, "pure_max")] for j in REFS]
            gs_pc = [A[(arm, a, j, "official_pc")] - A[(arm, a, j, "pure_max")] for j in REFS]
            G[(arm, a)] = {"G_t8_max": stats(gs), "G_pc_max": stats(gs_pc), "per_ref_G": gs}
    for a in COMBOS:
        for (x, y, label) in (("Y-RAW", "Y-POOL", "pooling_YOLO"), ("W-STD", "W-NOPOOL", "pooling_WRN")):
            diffs = [G[(x, a)]["per_ref_G"][i] - G[(y, a)]["per_ref_G"][i] for i in range(len(REFS))]
            paired_rows.append({"asset": a, "comparison": label, "metric": "G(top8)-G(max) 配对差",
                                 "per_ref": diffs, **stats(diffs)})
    # 阶段B比较 (max 口径; 同容量 M=1000)
    for a in COMBOS:
        for (x, y, label) in (("p3", "p4", "P3-P4"), ("p3", "p5", "P3-P5"), ("p4", "p5", "P4-P5")):
            for k in ("pure_max", "top8"):
                diffs = [B[(x, a, j, k)] - B[(y, a, j, k)] for j in REFS]
                paired_rows.append({"asset": a, "comparison": "phaseB_%s_%s" % (label, k),
                                     "metric": "AUROC差 逐参照", "per_ref": diffs, **stats(diffs)})
        # P3 通用容量 vs 历史 ratio 臂 (Y-RAW)
        for k in ("pure_max", "top8"):
            diffs = [B[("p3", a, j, k)] - A[("Y-RAW", a, j, k)] for j in REFS]
            paired_rows.append({"asset": a, "comparison": "P3容量1000_vs_ratio历史_%s" % k,
                                 "metric": "AUROC差 逐参照", "per_ref": diffs, **stats(diffs)})
    with open(os.path.join(OUT, "paired_comparisons.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["asset", "comparison", "metric", "mean", "min", "max", "std", "per_ref"])
        for r_ in paired_rows:
            w.writerow([r_["asset"], r_["comparison"], r_["metric"], round(r_["mean"], 6), round(r_["min"], 6),
                         round(r_["max"], 6), round(r_["std"], 6), "|".join("%.6f" % x for x in r_["per_ref"])])
    metrics = {
        "arms_completed": 4, "repro_gate_ok": int(bool(gates["repro"].get("ok"))),
        "batch_gates_ok": int(bool(gates["batch"].get("ok"))),
        "bank_double_build_ok": int(bool(gates["double_build"].get("ok"))),
        "state_hash_ok": int(bool(gates["state_hash"].get("yolo_final")) and bool(gates["state_hash"].get("wrn_final"))),
        "phase_b_completed": 1, "vis_completed": 0, "set_match": int(gates.get("set_match", 0)),
        "auroc_phaseA": {"|".join(map(str, k)): v for k, v in A.items()},
        "auroc_phaseB": {"|".join(map(str, k)): v for k, v in B.items()},
        "G": {"%s|%s" % k: v for k, v in G.items()},
        "distribution_effects": {"%s|%s|%d" % k: v for k, v in dist_eff.items()},
        "notes": {"refs_not_independent": "5 参照同池不同支持, 非独立数据集, 无显著性声明",
                   "topk_semantics": "topK=图像内最高距离位置数; 与最近邻数 nn=9 无关; top8 无空间感知",
                   "exploratory": "探索性机制诊断, 不作泛化声明"}}
    atomic = os.path.join(OUT, "metrics.json")
    with open(atomic + ".tmp", "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    os.replace(atomic + ".tmp", atomic)
    # 快速摘要
    for a in COMBOS:
        print("== %s ==" % a)
        for arm in ARMS:
            g = G[(arm, a)]["G_t8_max"]
            print("  %-9s G=%+.4f [%+.4f,%+.4f]  t8=%.4f max=%.4f" % (arm, g["mean"], g["min"], g["max"],
                  np.mean([A[(arm, a, j, "top8")] for j in REFS]), np.mean([A[(arm, a, j, "pure_max")] for j in REFS])))
    print("== phaseB (10N M=1000, max口径) ==")
    for a in COMBOS:
        for ln in ("p3", "p4", "p5"):
            print("  %s %-2s max=%.4f t8=%.4f" % (a[:14], ln, np.mean([B[(ln, a, j, "pure_max")] for j in REFS]),
                                                   np.mean([B[(ln, a, j, "top8")] for j in REFS])))
    print("metrics.json + paired_comparisons.csv 完成")


if __name__ == "__main__":
    main()

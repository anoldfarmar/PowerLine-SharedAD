#!/usr/bin/env python3
"""M3MECH-followup-v2 §四: 已有阶段B (P3/P4/P5, M=1000) 分数汇总 — 不新增GPU实验.
四种评分 (pure_max/top8/top0.5pct/top1pct) 层间比较; 实际k与重复评分复用;
固定top8 vs 固定比例的层间结论; P3共同容量 vs 历史容量; 混杂披露."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12/M3MECH-followup-v2"
PB = "/root/autodl-tmp/version10-12/raw_scores/phaseB"
COMBOS = {"yoke-suspension": "yoke-suspension__rust", "vari-grip": "vari-grip__bird-nest"}
LAYERS = ["p3", "p4", "p5"]
SCORES = ["pure_max", "top8", "top0.5pct", "top1pct"]
REFS = [3001, 3002, 3003, 3004, 3005]
META = {
    "p3": {"module": "backbone layer4 (C2f, stride8)", "ch": 64, "grid": "40x40=1600", "capacity": 1000, "cand": 16000, "all_kept": False, "k": {"top8": 8, "top0.5pct": 8, "top1pct": 16}},
    "p4": {"module": "backbone layer6 (C2f, stride16)", "ch": 128, "grid": "20x20=400", "capacity": 1000, "cand": 4000, "all_kept": False, "k": {"top8": 8, "top0.5pct": 2, "top1pct": 4}},
    "p5": {"module": "backbone layer9 (SPPF, stride32, 含SPPF)", "ch": 256, "grid": "10x10=100", "capacity": 1000, "cand": 1000, "all_kept": True, "k": {"top8": 8, "top0.5pct": 1, "top1pct": 1}}}


def auc(dv, nv, key):
    u, st = D.auroc_avg_rank(np.array([v[key] for v in dv.values()], np.float64),
                              np.array([v[key] for v in nv.values()], np.float64))
    assert st == "ok"
    return float(u)


def main():
    S = {}
    kdup = {}
    for a in COMBOS:
        for ln in LAYERS:
            for j in REFS:
                r = json.load(open("%s/%s_%s_ref%d.json" % (PB, COMBOS[a], ln, j)))
                assert r["M"] == 1000 and r["layer"] == ln
                kdup["%s|%s|%d" % (ln, a, j)] = r.get("duplicate_k", False)
                for sc in SCORES:
                    S[(ln, a, j, sc)] = auc(r["defect"], r["normal"], sc)
    # 汇总
    summary = {"per_ref": {"%s|%s|%d|%s" % k: v for k, v in S.items()},
                "k_actual": {ln: META[ln]["k"] for ln in LAYERS},
                "k_duplicate_note": "P5 候选=100, k(0.5%)=k(1%)=1=max, 重复评分复用同一向量",
                "layers": {}}
    for ln in LAYERS:
        summary["layers"][ln] = {sc: {"yoke_mean": float(np.mean([S[(ln, "yoke-suspension", j, sc)] for j in REFS])),
                                       "yoke_all_pos": bool(all(S[(ln, "yoke-suspension", j, sc)] > S[(l2, "yoke-suspension", j, sc)] for j in REFS for l2 in LAYERS if l2 != ln)),
                                       "vari_mean": float(np.mean([S[(ln, "vari-grip", j, sc)] for j in REFS]))} for sc in SCORES}
    # 反转保持性: yoke 最优层 vs vari 最优层 在各评分下
    inversion = {}
    for sc in SCORES:
        y_best = max(LAYERS, key=lambda ln: summary["layers"][ln][sc]["yoke_mean"])
        v_best = max(LAYERS, key=lambda ln: summary["layers"][ln][sc]["vari_mean"])
        inversion[sc] = {"yoke_best": y_best, "vari_best": v_best, "inverted": y_best != v_best}
    summary["inversion"] = inversion
    # P3 共同容量 vs 历史 ratio 容量 (M3MECH phaseA Y-RAW = 同特征同规范 M=1600)
    A1 = {}
    for a in COMBOS:
        for j in REFS:
            r = json.load(open("/root/autodl-tmp/version10-12/raw_scores/phaseA/%s_Y-RAW_ref%d.json" % (COMBOS[a], j)))
            A1[(a, j, "pure_max")] = auc(r["defect"], r["normal"], "pure_max")
            A1[(a, j, "top8")] = auc(r["defect"], r["normal"], "top8")
    summary["p3_capacity_effect"] = {}
    for sc in ("pure_max", "top8"):
        summary["p3_capacity_effect"][sc] = {
            a: {"M1000": float(np.mean([S[("p3", a, j, sc)] for j in REFS])),
                 "M1600_hist": float(np.mean([A1[(a, j, sc)] for j in REFS])),
                 "diff": float(np.mean([S[("p3", a, j, sc)] for j in REFS]) - np.mean([A1[(a, j, sc)] for j in REFS]))} for a in COMBOS}
    summary["layer_meta"] = META
    summary["discipline"] = {"pooled_across_refs": "5参照均值, 非独立数据集",
                              "confounders": "通道数/感受野/网格密度/候选保留方式同时变化, 语义归因不可分离",
                              "no_cherry_pick": "不在结果上拼接各资产最优层宣称新系统"}
    with open(os.path.join(OUT, "layer_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("层间均值 (10N, M=1000):")
    for sc in SCORES:
        print("  %-9s yoke: %s | vari: %s" % (sc,
              " ".join("%s=%.4f" % (ln, summary["layers"][ln][sc]["yoke_mean"]) for ln in LAYERS),
              " ".join("%s=%.4f" % (ln, summary["layers"][ln][sc]["vari_mean"]) for ln in LAYERS)))
    print("反转:", inversion)
    print("P3 容量效应:", json.dumps(summary["p3_capacity_effect"]))


if __name__ == "__main__":
    main()

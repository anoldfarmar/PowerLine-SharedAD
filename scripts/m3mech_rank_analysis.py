#!/usr/bin/env python3
"""M3MECH-followup-v2 §三: 已有分数上的 top8 排序转换分析 (纯CPU).
对每 臂x资产x参照: 逐配对 (故障a, 正常n) 的 m_max/m_t8, 权重(1/0.5/0),
W/L/并列转换, 净权重变化; 恒等式核验 net/(nd*nn) == AUROC(top8)-AUROC(max);
边界邻近性, 每图/源组集中度, 池化配对对比; 固定种子示例配对."""
import csv, hashlib, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12/M3MECH-followup-v2"
SRC = "/root/autodl-tmp/version10-12/raw_scores/phaseA"
ARMS = ["Y-RAW", "Y-POOL", "W-STD", "W-NOPOOL"]
COMBOS = {"yoke-suspension": "yoke-suspension__rust", "vari-grip": "vari-grip__bird-nest"}
REFS = [3001, 3002, 3003, 3004, 3005]
SEED = 20250930


def w(m):
    return np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))


def dji_key(p):
    m = re.search(r"_DJI_(\d{4})_", "_" + os.path.basename(p))
    return m.group(1) if m else "nogroup"


def auroc_from_weights(W):
    return float(W.mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    metrics = {"combos": {}, "identity_max_abs_err": 0.0, "algebra_max_abs_err": 0.0}
    rep_rows = []
    rng = np.random.Generator(np.random.PCG64(SEED))
    for arm in ARMS:
        for a, combo in COMBOS.items():
            for j in REFS:
                r = json.load(open("%s/%s_%s_ref%d.json" % (SRC, combo, arm, j)))
                qn = list(r["normal"].keys())
                qd = list(r["defect"].keys())
                nd, nn = len(qd), len(qn)
                dmax = np.array([r["defect"][p]["pure_max"] for p in qd])
                nmax = np.array([r["normal"][p]["pure_max"] for p in qn])
                dt8 = np.array([r["defect"][p]["top8"] for p in qd])
                nt8 = np.array([r["normal"][p]["top8"] for p in qn])
                m_max = dmax[:, None] - nmax[None, :]
                m_t8 = dt8[:, None] - nt8[None, :]
                Wmax, Wt8 = w(m_max), w(m_t8)
                net = float(Wt8.sum() - Wmax.sum())
                auroc_max, auroc_t8 = auroc_from_weights(Wmax), auroc_from_weights(Wt8)
                ident = abs(net / (nd * nn) - (auroc_t8 - auroc_max))
                metrics["identity_max_abs_err"] = max(metrics["identity_max_abs_err"], ident)
                # 恒等式 m_t8 = m_max + delta_n - delta_f (dn/dn 按侧: normal→nn 维, fault→nd 维)
                dn = nmax - nt8
                df = dmax - dt8
                alg = np.abs(m_t8 - (m_max + dn[None, :] - df[:, None])).max()
                metrics["algebra_max_abs_err"] = max(metrics["algebra_max_abs_err"], float(alg))
                wrong = m_max < 0
                right = m_max > 0
                tie_max = m_max == 0
                Wm = wrong & (m_t8 > 0)
                Lm = right & (m_t8 < 0)
                tie_to = (m_max == 0) & (m_t8 != 0)
                to_tie = (m_max != 0) & (m_t8 == 0)
                # 边界邻近性: 被救对 vs 全部 max 错对 的 |m_max|
                rescued_m = np.abs(m_max[Wm]) if Wm.sum() else np.array([0.0])
                wrong_m = np.abs(m_max[wrong]) if wrong.sum() else np.array([0.0])
                # 集中度: 每故障图/每正常图的被救对数, top10 图份额; 源组
                per_f = Wm.sum(1)
                per_n = Wm.sum(0)
                top10f = float(np.sort(per_f)[::-1][:10].sum()) / max(int(Wm.sum()), 1)
                top10n = float(np.sort(per_n)[::-1][:10].sum()) / max(int(Wm.sum()), 1)
                dj_f = [dji_key(p) for p in qd]
                dj_n = [dji_key(p) for p in qn]
                grp = {}
                for fi, ni in zip(*np.where(Wm)):
                    k = (dj_f[fi], dj_n[ni])
                    grp[k] = grp.get(k, 0) + 1
                grp_counts = sorted(grp.values(), reverse=True)
                # 被破坏对画像
                broken_f_pct = np.percentile([ (dmax <= dmax[i]).mean() for i in np.where(Lm)[0] ], 50) if Lm.sum() else None
                broken_n_pct = np.percentile([ (nmax <= nmax[j_]).mean() for j_ in np.where(Lm)[1] ], 50) if Lm.sum() else None
                rescued_f_pct = np.percentile([ (dmax <= dmax[i]).mean() for i in np.where(Wm)[0] ], 50) if Wm.sum() else None
                rescued_n_pct = np.percentile([ (nmax <= nmax[j_]).mean() for j_ in np.where(Wm)[1] ], 50) if Wm.sum() else None
                key = "%s|%s|%d" % (arm, a, j)
                metrics["combos"][key] = {
                    "nd": nd, "nn": nn,
                    "auroc_max": auroc_max, "auroc_top8": auroc_t8, "G": auroc_t8 - auroc_max,
                    "W_saved": int(Wm.sum()), "L_broken": int(Lm.sum()),
                    "tie_max_pairs": int(tie_max.sum()), "tie_top8_pairs": int((m_t8 == 0).sum()),
                    "tie_involved_transitions": int(tie_to.sum() + to_tie.sum()),
                    "net_weight_change": net,
                    "identity_err": float(ident), "algebra_err": float(alg),
                    "delta_normal_mean": float(dn.mean()), "delta_fault_mean": float(df.mean()),
                    "delta_normal_p99": float(np.percentile(dn, 99)), "delta_fault_p99": float(np.percentile(df, 99)),
                    "rescued_abs_mmax_median": float(np.median(rescued_m)),
                    "wrong_abs_mmax_median": float(np.median(wrong_m)),
                    "rescued_near_boundary_frac": float((rescued_m <= np.median(wrong_m)).mean()),
                    "per_fault_top10_share": top10f, "per_normal_top10_share": top10n,
                    "rescued_source_group_pairs": len(grp),
                    "rescued_top_group_share": float(grp_counts[0]) / max(int(Wm.sum()), 1) if grp_counts else 0.0,
                    "rescued_fault_maxpct_median": None if rescued_f_pct is None else float(rescued_f_pct),
                    "rescued_normal_maxpct_median": None if rescued_n_pct is None else float(rescued_n_pct),
                    "broken_fault_maxpct_median": None if broken_f_pct is None else float(broken_f_pct),
                    "broken_normal_maxpct_median": None if broken_n_pct is None else float(broken_n_pct)}
                # 示例配对 (固定种子): 10 救 + 10 破
                for kind, mask in (("saved", Wm), ("broken", Lm)):
                    fi_list, ni_list = np.where(mask)
                    if len(fi_list):
                        pick = rng.choice(len(fi_list), size=min(10, len(fi_list)), replace=False)
                        for ix in pick:
                            fi, ni = fi_list[ix], ni_list[ix]
                            rep_rows.append({"arm": arm, "asset": a, "ref": j, "kind": kind,
                                              "fault_path": qd[fi], "normal_path": qn[ni],
                                              "fault_source": dj_f[fi], "normal_source": dj_n[ni],
                                              "m_max": float(m_max[fi, ni]), "m_t8": float(m_t8[fi, ni]),
                                              "delta_fault": float(df[fi]), "delta_normal": float(dn[ni])})
    assert metrics["identity_max_abs_err"] <= 1e-9, "净变化恒等式不成立"
    assert metrics["algebra_max_abs_err"] <= 1e-9, "m_t8 分解恒等式不成立"
    with open(os.path.join(OUT, "rank_transition_metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "representative_pairs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rep_rows[0].keys()))
        wr.writeheader()
        wr.writerows(rep_rows)
    print("identity_err=%.2e algebra_err=%.2e combos=%d rep_pairs=%d" % (
        metrics["identity_max_abs_err"], metrics["algebra_max_abs_err"], len(metrics["combos"]), len(rep_rows)))
    # 快速摘要
    for arm in ARMS:
        for a in COMBOS:
            ks = [k for k in metrics["combos"] if k.startswith("%s|%s|" % (arm, a))]
            W_ = sum(metrics["combos"][k]["W_saved"] for k in ks)
            L_ = sum(metrics["combos"][k]["L_broken"] for k in ks)
            G_ = np.mean([metrics["combos"][k]["G"] for k in ks])
            nb = np.mean([metrics["combos"][k]["rescued_near_boundary_frac"] for k in ks])
            print("%-9s %-16s W=%6d L=%5d G=%+.4f 近边界占比=%.2f top10图份额 F=%.2f N=%.2f" % (
                arm, a, W_, L_, G_, nb,
                np.mean([metrics["combos"][k]["per_fault_top10_share"] for k in ks]),
                np.mean([metrics["combos"][k]["per_normal_top10_share"] for k in ks])))


if __name__ == "__main__":
    main()

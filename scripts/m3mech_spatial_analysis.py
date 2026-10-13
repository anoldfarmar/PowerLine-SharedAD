#!/usr/bin/env python3
"""M3MECH-followup-v2 §六: 空间聚集几何统计修正 (纯CPU).
冻结随机样本 top8 坐标: 平均两两距离 / 平均最近邻距离 / 固定邻接规则连通分量数;
实际 40x40 网格上 1000 次无放回随机抽 8 格的经验参照 (固定种子).
只作几何描述: 不证明语义独立/故障定位, 不据此启动任何算法设计."""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12/M3MECH-followup-v2"
MECH = "/root/autodl-tmp/version10-12"
ARMS = ["Y-RAW", "W-STD"]
GRID = 40
K = 8
NDRAW = 1000
SEED = 20250930


def metrics_from_cells(cells):
    pts = np.array(cells, np.float64)
    d = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(pts), 1)
    mp = float(d[iu].mean())
    npd = d + np.eye(len(pts)) * 1e9
    mnn = float(npd.min(1).mean())
    # 连通分量 (固定规则: Chebyshev 距离 <=1 视为相邻)
    seen = set()
    comp = 0
    cellset = set(map(tuple, np.array(cells).tolist()))
    for c in cellset:
        if c in seen:
            continue
        comp += 1
        stack = [c]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nxt = (cur[0] + dx, cur[1] + dy)
                    if nxt in cellset and nxt not in seen:
                        stack.append(nxt)
    return mp, mnn, comp


def main():
    Fi = json.load(open(os.path.join(MECH, "FROZEN_INPUTS.json")))
    npz = np.load(os.path.join(MECH, "cache", "vis_dist", "ref3001_dist_matrices.npz"), allow_pickle=True)
    rng = np.random.Generator(np.random.PCG64(SEED))
    # 经验参照: 40x40 网格无放回抽 8 格, 1000 次
    null = {"mp": [], "mnn": [], "comp": []}
    for _ in range(NDRAW):
        cells = rng.choice(GRID * GRID, size=K, replace=False)
        cells = [(int(c) // GRID, int(c) % GRID) for c in cells]
        mp, mnn, comp = metrics_from_cells(cells)
        null["mp"].append(mp)
        null["mnn"].append(mnn)
        null["comp"].append(comp)
    null = {k: np.array(v) for k, v in null.items()}
    null_stats = {k: {"mean": float(v.mean()), "q05": float(np.percentile(v, 5)),
                       "q50": float(np.percentile(v, 50)), "q95": float(np.percentile(v, 95))} for k, v in null.items()}
    out = {"grid": GRID, "K": K, "n_draws": NDRAW, "seed": SEED,
            "null": null_stats,
            "null_note": "随机参照只检验几何聚集; 不证明语义独立或故障定位; 感受野重叠可使分散位置仍高度相关",
            "per_image": []}
    agg = {}
    for a in ("yoke-suspension", "vari-grip"):
        for role in ("normal", "fault"):
            paths = Fi["frozen_samples"]["%s_%s" % (a, role)]
            for arm in ARMS:
                order = json.loads(str(npz["%s_%s__order" % (arm, a)]))
                Dm = npz["%s_%s" % (arm, a)]
                idx = {p: i for i, p in enumerate(order)}
                for p in paths:
                    drow = Dm[idx[p]]
                    top8 = drow.argsort()[::-1][:K]
                    cells = [(int(t) // GRID, int(t) % GRID) for t in top8]
                    mp, mnn, comp = metrics_from_cells(cells)
                    pv = {"mp": float((null["mp"] <= mp).mean()),
                           "mnn": float((null["mnn"] <= mnn).mean()),
                           "comp": float((null["comp"] <= comp).mean())}
                    rec = {"asset": a, "role": role, "arm": arm, "path": p,
                            "mean_pairwise": mp, "mean_nn": mnn, "components": comp,
                            "p_cluster_mp": pv["mp"], "p_cluster_mnn": pv["mnn"], "p_cluster_comp": pv["comp"]}
                    out["per_image"].append(rec)
                    key = (a, role, arm)
                    agg.setdefault(key, []).append(rec)
    summary = {}
    for (a, role, arm), recs in agg.items():
        summary["%s|%s|%s" % (a, role, arm)] = {
            "n": len(recs),
            "mean_pairwise_mean": float(np.mean([r["mean_pairwise"] for r in recs])),
            "mean_nn_mean": float(np.mean([r["mean_nn"] for r in recs])),
            "components_mean": float(np.mean([r["components"] for r in recs])),
            "frac_p_mp_lt005": float(np.mean([r["p_cluster_mp"] < 0.05 for r in recs])),
            "frac_p_mnn_lt005": float(np.mean([r["p_cluster_mnn"] < 0.05 for r in recs])),
            "frac_p_comp_lt005": float(np.mean([r["p_cluster_comp"] < 0.05 for r in recs]))}
    out["summary"] = summary
    with open(os.path.join(OUT, "spatial_stats.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("经验参照 (40x40 抽8格): mp=%.1f [%.1f,%.1f] mnn=%.1f comp=%.1f" % (
        null_stats["mp"]["mean"], null_stats["mp"]["q05"], null_stats["mp"]["q95"],
        null_stats["mnn"]["mean"], null_stats["comp"]["mean"]))
    for k, v in summary.items():
        print("%-42s mp=%.1f mnn=%.1f 分量=%.1f 显著聚集率 mp=%.2f mnn=%.2f comp=%.2f" % (
            k, v["mean_pairwise_mean"], v["mean_nn_mean"], v["components_mean"],
            v["frac_p_mp_lt005"], v["frac_p_mnn_lt005"], v["frac_p_comp_lt005"]))


if __name__ == "__main__":
    main()

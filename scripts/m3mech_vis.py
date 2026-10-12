#!/usr/bin/env python3
"""M3MECH 可视化: 冻结样本组 + 预注册失败案例组; 固定颜色规则 (臂内正常样本 99.5 分位 vmax).
输出 vis/ 下网格 PNG + contact_sheet.csv + spike_cluster_stats.csv; 置 metrics.json vis_completed=1."""
import csv, json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12"
ARMS = ["Y-RAW", "Y-POOL", "W-STD", "W-NOPOOL"]
LAYERS = ["p3", "p4", "p5"]
S = 320
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    MAGMA = cm.get_cmap("magma")
except Exception:
    MAGMA = None

Fi = json.load(open(os.path.join(OUT, "FROZEN_INPUTS.json")))
COMBOS = {"yoke-suspension": "yoke-suspension__rust", "vari-grip": "vari-grip__bird-nest"}


def order_index(order_json, need):
    idx = {p: i for i, p in enumerate(order_json)}
    return [idx[p] for p in need]


def heat(D_row, H, W, vmax):
    m = D_row.reshape(H, W).astype(np.float32)
    img = Image.fromarray((np.clip(m / max(vmax, 1e-12), 0, 1) * 255).astype(np.uint8))
    img = img.resize((S, S), Image.BILINEAR)
    if MAGMA is not None:
        a = np.asarray(img, np.float32) / 255.0
        rgb = (np.array(MAGMA(a)[..., :3]) * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
    return img


def overlay(orig_img, heat_img, alpha=0.45):
    base = orig_img.convert("RGB").resize((S, S))
    return Image.blend(base, heat_img, alpha)


def markers(img, D_row, P):
    arr = np.array(img)
    H = W = int(round(P ** 0.5))
    flat = D_row.argsort()[::-1]
    for rank, pi in enumerate(flat[:8]):
        r, c = pi // W, pi % W
        y, x = int((r + 0.5) * S / H), int((c + 0.5) * S / W)
        col = (255, 40, 40) if rank == 0 else (255, 165, 0)
        arr[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = col
    return Image.fromarray(arr)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    npzA = np.load(os.path.join(OUT, "cache", "vis_dist", "ref3001_dist_matrices.npz"), allow_pickle=True)
    npzB = np.load(os.path.join(OUT, "cache", "vis_dist", "phaseB_ref3001_dist.npz"), allow_pickle=True)
    rows_csv = []
    spike_rows = []
    for a in ("yoke-suspension", "vari-grip"):
        arm_order = {arm: json.loads(str(npzA["%s_%s__order" % (arm, a)])) for arm in ARMS}
        Ds = {arm: npzA["%s_%s" % (arm, a)] for arm in ARMS}
        P = {arm: Ds[arm].shape[1] for arm in ARMS}
        H = {arm: int(round(P[arm] ** 0.5)) for arm in ARMS}
        # 固定颜色规则: 每臂 vmax = 冻结正常 12 张的 patch 距离 99.5 分位
        vmax = {}
        for arm in ARMS:
            rows = order_index(arm_order[arm], Fi["frozen_samples"]["%s_normal" % a])
            vmax[arm] = float(np.percentile(Ds[arm][rows], 99.5))
        layer_order = {ln: json.loads(str(npzB["%s_%s__order" % (ln, a)])) for ln in LAYERS}
        Dl = {ln: npzB["%s_%s" % (ln, a)] for ln in LAYERS}
        Pl = {ln: Dl[ln].shape[1] for ln in LAYERS}
        Hl = {ln: int(round(Pl[ln] ** 0.5)) for ln in LAYERS}
        vmaxl = {}
        for ln in LAYERS:
            rows = order_index(layer_order[ln], Fi["frozen_samples"]["%s_normal" % a])
            vmaxl[ln] = float(np.percentile(Dl[ln][rows], 99.5))
        # 失败案例组 (预注册规则, 结果驱动)
        rs = json.load(open(os.path.join(OUT, "raw_scores", "phaseA", "%s_%s_ref3001.json" % (COMBOS[a], "Y-RAW"))))
        rsW = json.load(open(os.path.join(OUT, "raw_scores", "phaseA", "%s_%s_ref3001.json" % (COMBOS[a], "W-STD"))))
        qn = Fi["queries"][a]["normal"]
        qd = Fi["queries"][a]["defect"]
        ymax = {p: rs["normal"].get(p, rs["defect"].get(p))["pure_max"] for p in qn + qd}
        wmax = {p: rsW["normal"].get(p, rsW["defect"].get(p))["pure_max"] for p in qn + qd}
        yall = np.array(list(ymax.values())); wall_ = np.array(list(wmax.values()))
        py = {p: float((yall <= v).mean()) for p, v in ymax.items()}
        pw = {p: float((wall_ <= v).mean()) for p, v in wmax.items()}
        F1 = sorted(qn, key=lambda p: -ymax[p])[:5]
        F2 = sorted(qd, key=lambda p: ymax[p])[:5]
        F3 = sorted(qn + qd, key=lambda p: -abs(py[p] - pw[p]))[:5]
        groups = [("fixed_normal", Fi["frozen_samples"]["%s_normal" % a]),
                   ("fixed_fault", Fi["frozen_samples"]["%s_fault" % a]),
                   ("F1_normal_top", F1), ("F2_fault_bottom", F2), ("F3_disagree", F3)]
        for gname, plist in groups:
            for k, p in enumerate(plist):
                orig = Image.open(p).convert("RGB")
                fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
                panels = [(orig, None, None, "original")]
                for arm in ARMS:
                    i = order_index(arm_order[arm], [p])[0]
                    drow = Ds[arm][i]
                    hm = heat(drow, H[arm], H[arm], vmax[arm])
                    img2 = markers(overlay(orig, hm), drow, P[arm]) if arm in ("Y-RAW", "W-STD") else overlay(orig, hm)
                    panels.append((img2, None, None, "%s (vmax=%.3g)" % (arm, vmax[arm])))
                for ax, (im, _, _, title) in zip(axes[0], panels):
                    ax.imshow(im); ax.set_title(title, fontsize=8); ax.axis("off")
                axes[1, 0].imshow(orig.resize((S, S))); axes[1, 0].set_title("original", fontsize=8); axes[1, 0].axis("off")
                for ln, ax in zip(LAYERS, list(axes[1])[1:]):
                    i = order_index(layer_order[ln], [p])[0]
                    ax.imshow(overlay(orig, heat(Dl[ln][i], Hl[ln], Hl[ln], vmaxl[ln])))
                    ax.set_title("%s M=1000 (vmax=%.3g)" % (ln.upper(), vmaxl[ln]), fontsize=8); ax.axis("off")
                role = "normal" if "/good/" in p else "fault"
                fig.suptitle("%s | %s | %s#%d | %s" % (a, gname, role, k + 1, os.path.basename(p)[:60]), fontsize=9)
                fig.tight_layout()
                fp = os.path.join(OUT, "vis", "%s_%s_%02d.png" % (a, gname, k + 1))
                fig.savefig(fp, dpi=90); plt.close(fig)
                rows_csv.append({"asset": a, "group": gname, "role": role, "path": p,
                                  "Y-RAW_max": round(ymax[p], 6), "W-STD_max": round(wmax[p], 6),
                                  "pct_Y": round(py[p], 4), "pct_W": round(pw[p], 4), "grid_png": os.path.basename(fp)})
                # 尖峰/聚类量化 (仅 fixed 组)
                if gname.startswith("fixed"):
                    i = order_index(arm_order["Y-RAW"], [p])[0]
                    drow = Ds["Y-RAW"][i]
                    top8 = drow.argsort()[::-1][:8]
                    cxy = np.array([[pi // H["Y-RAW"], pi % H["Y-RAW"]] for pi in top8], np.float32)
                    pd = np.sqrt(((cxy[:, None] - cxy[None]) ** 2).sum(-1))[np.triu_indices(8, 1)].mean()
                    expd = 0.526 * H["Y-RAW"]  # 均匀随机期望最近邻间距近似
                    spike_rows.append({"asset": a, "group": gname, "role": role, "path": p,
                                        "max_over_top8": float(drow.max() / max(drow[top8].mean(), 1e-12)),
                                        "top8_mean_pairdist_grid": float(pd), "random_expected": float(expd),
                                        "clustered": bool(pd < expd * 0.6)})
        print("[vis] %s 完成" % a, flush=True)
    with open(os.path.join(OUT, "vis", "contact_sheet.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys())); w.writeheader(); w.writerows(rows_csv)
    with open(os.path.join(OUT, "vis", "spike_cluster_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spike_rows[0].keys())); w.writeheader(); w.writerows(spike_rows)
    ncl = sum(1 for r in spike_rows if r["clustered"])
    mp = json.load(open(os.path.join(OUT, "metrics.json")))
    mp["vis_completed"] = 1
    mp["vis_summary"] = {"grids": len(rows_csv), "fixed_clustered_ratio": ncl / max(len(spike_rows), 1),
                          "color_rule": "每臂 vmax=该资产冻结正常12张 patch 距离 99.5 分位; 不做逐图 min-max; 上采样仅显示"}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(mp, f, ensure_ascii=False, indent=1)
    print("vis 完成: %d grids, %d spike rows, clustered %.2f" % (len(rows_csv), len(spike_rows), ncl / max(len(spike_rows), 1)))


if __name__ == "__main__":
    main()

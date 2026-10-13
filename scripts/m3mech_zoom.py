#!/usr/bin/env python3
"""M3MECH 标注辅助: 为 78 张 vis 网格图生成 top1-top8 放大切片表 + 标注模板.
zoom sheet: 2 行 x 9 列; 行1=带 Y-RAW 标记的原图 + Y-RAW top8 放大窗; 行2=同 W-STD.
窗口取自原始分辨率 (40x40 网格 -> 原图像素), 仅显示用."""
import csv, json, os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/version10-12"
Fi = json.load(open(os.path.join(OUT, "FROZEN_INPUTS.json")))
npz = np.load(os.path.join(OUT, "cache", "vis_dist", "ref3001_dist_matrices.npz"), allow_pickle=True)
ARMS = ["Y-RAW", "W-STD"]
GRID = 40
WIN_FRAC = 0.16  # 窗口边长 = 原图短边 16%


def order_index(order, need):
    idx = {p: i for i, p in enumerate(order)}
    return [idx[p] for p in need]


def main():
    os.makedirs(os.path.join(OUT, "vis", "zoom"), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_out = []
    cs = list(csv.DictReader(open(os.path.join(OUT, "vis", "contact_sheet.csv"))))
    for rec in cs:
        a, p = rec["asset"], rec["path"]
        Ds, orders = {}, {}
        for arm in ARMS:
            orders[arm] = json.loads(str(npz["%s_%s__order" % (arm, a)]))
            Ds[arm] = npz["%s_%s" % (arm, a)]
        orig = Image.open(p).convert("RGB")
        W, H = orig.size
        win = max(48, int(WIN_FRAC * min(W, H)))
        fig, axes = plt.subplots(2, 9, figsize=(19.5, 4.6))
        tmpl_rows = []
        for ri, arm in enumerate(ARMS):
            i = order_index(orders[arm], [p])[0]
            drow = Ds[arm][i].astype(np.float32)
            top8 = drow.argsort()[::-1][:8]
            small = orig.resize((330, int(330 * H / W))) if W >= H else orig.resize((int(330 * W / H), 330))
            dr = ImageDraw.Draw(small)
            for rank, pi in enumerate(top8):
                r, c = pi // GRID, pi % GRID
                x0, y0 = c * W / GRID, r * H / GRID
                x1, y1 = (c + 1) * W / GRID, (r + 1) * H / GRID
                sx, sy = small.size[0] / W, small.size[1] / H
                dr.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 30, 30) if rank == 0 else (255, 165, 0), width=2)
            axes[ri, 0].imshow(small)
            axes[ri, 0].set_title("%s 标记 (红=top1)" % arm, fontsize=8)
            axes[ri, 0].axis("off")
            for rank, pi in enumerate(top8):
                r, c = pi // GRID, pi % GRID
                cx, cy = (c + 0.5) * W / GRID, (r + 0.5) * H / GRID
                x0, x1 = int(max(0, min(W, cx - win / 2))), int(max(0, min(W, cx + win / 2)))
                y0, y1 = int(max(0, min(H, cy - win / 2))), int(max(0, min(H, cy + win / 2)))
                crop = orig.crop((x0, y0, x1, y1)).resize((220, 220))
                axes[ri, rank + 1].imshow(crop)
                axes[ri, rank + 1].set_title("#%d d=%.3g" % (rank + 1, drow[pi]), fontsize=8)
                axes[ri, rank + 1].axis("off")
                tmpl_rows.append({"rank": rank + 1, "cell_r": r, "cell_c": c, "dist": round(float(drow[pi]), 6)})
                rows_out.append({"asset": a, "group": rec["group"], "role": rec["role"], "path": p,
                                  "grid_png": rec["grid_png"], "arm": arm, **tmpl_rows[-1],
                                  "label": "", "notes": ""})
        fig.suptitle("ZOOM | %s | %s | %s | %s" % (a, rec["group"], rec["role"], os.path.basename(p)[:58]), fontsize=9)
        fig.tight_layout()
        zp = os.path.join(OUT, "vis", "zoom", rec["grid_png"].replace(".png", "_zoom.png"))
        fig.savefig(zp, dpi=80)
        plt.close(fig)
    fields = ["asset", "group", "role", "path", "grid_png", "arm", "rank", "cell_r", "cell_c", "dist", "label", "notes"]
    with open(os.path.join(OUT, "vis", "labeling_template.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print("zoom sheets: %d, template rows: %d" % (len(cs), len(rows_out)))


if __name__ == "__main__":
    main()

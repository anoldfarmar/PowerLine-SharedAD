#!/usr/bin/env python3
"""M3 Step-6: 腐蚀类 (rust/corrosion) 专项分析 → rust_analysis.json + PNG 热图.

内容:
  1. 全部 5 个 rust/corrosion 组合在四变体下的逐类结果 (1/5/10-shot × 3 种子)
  2. 局部相似度最高响应 patch: patch-pre 变体下, 每组合取 3 个查询样本,
     记录 top-8 响应 patch 网格坐标, 并绘制 overlay PNG
  3. 样本不足组合明确标记统计限制 (不得省略或合并不利结果)

用法: python3 scripts/m3_rust_analysis.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C
from m3_build_patch_bank import Projector, load_head

CACHE = "/root/auto-exp/outputs/M3/cache"
BANKS = "/root/auto-exp/outputs/M3/banks"
OUT = "/root/auto-exp/outputs/M3"


def topk_patch_overlay(crop_path, bank, projector, out_png, topk=8):
    """返回 top-k patch 网格坐标 (行,列,相似度), 并存 overlay PNG."""
    from PIL import Image, ImageDraw
    tf = C.make_transform(C.IMGSZ)
    x = tf(Image.open(crop_path).convert("RGB"))
    p3 = None
    bb = topk_patch_overlay._bb
    with torch.no_grad():
        p3, _ = bb.p3_and_p5gap(x[None])
    p3 = p3[0].permute(1, 2, 0)  # (40,40,64)
    raw = p3.reshape(-1, 64).float()
    with torch.no_grad():
        v = projector(raw)
        sims = (v @ torch.from_numpy(bank).float().to(v.device).T).max(-1).values
    topv, topi = torch.topk(sims, topk)
    coords = [{"grid_rc": [int(i // 40), int(i % 40)],
               "bbox_xyxy_px": [int(i % 40 * 8), int(i // 40 * 8),
                                int(i % 40 * 8 + 8), int(i // 40 * 8 + 8)],
               "sim": float(s)} for i, s in zip(topi, topv)]
    # overlay
    img = x.permute(1, 2, 0).numpy().copy()
    img = (img * 255).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    for r in coords:
        x1, y1, x2, y2 = r["bbox_xyxy_px"]
        dr.rectangle([x1, y1, x2, y2], outline=(255, 40, 40), width=2)
    im.save(out_png)
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    args = ap.parse_args()

    ev = json.load(open(f"{OUT}/proto_v1_eval.json", encoding="utf-8"))
    # 1. 腐蚀类逐类结果 (从主评测取, 不重算 —— 同一冻结协议)
    rust = {"combos": C.RUST_COMBOS, "per_combo": {}}
    for c in C.RUST_COMBOS:
        entry = {"variants": {}, "statistical_limitation": None}
        for variant, per_k in ev["summary"].items():
            entry["variants"][variant] = {
                k: {"auroc": per_k[k]["per_combo_auroc"].get(c),
                    "recall": per_k[k]["per_combo_recall"].get(c)}
                for k in per_k}
        if c == "polymer-insulator-upper-shackle/rust":
            entry["statistical_limitation"] = (
                "33 实例, k=10 查询侧仅 3 实例; AUROC 方差大, 统计意义弱 (冻结划分清单使用纪律)")
        rust["per_combo"][c] = entry

    # 2. 局部最高响应 patch (patch-pre, k=5 seed=0, 每组合 3 样本)
    meta = json.load(open(f"{CACHE}/meta.json", encoding="utf-8"))
    path2idx = {p: i for i, p in enumerate(meta["eval_paths"])}
    proj = Projector("pre")
    proj.proj = proj.proj.to("cuda" if torch.cuda.is_available() else "cpu")
    topk_patch_overlay._bb = C.Backbone(
        "/root/autodl-tmp/runs/yolov8n_det/weights/best.pt",
        "cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{args.splits}/kshot_k5_seed0.json", encoding="utf-8") as f:
        split = json.load(f)
    with open(f"{args.splits}/query_k5.json", encoding="utf-8") as f:
        query = json.load(f)
    bk = np.load(f"{BANKS}/pre/bank_k5_seed0.npz")
    patches_viz = {}
    vdir = f"{OUT}/rust_overlays"
    os.makedirs(vdir, exist_ok=True)
    for c in C.RUST_COMBOS:
        qp = query["defect_query"][c][:3]
        entries = []
        for j, p in enumerate(qp):
            coords = topk_patch_overlay(p, bk[c], proj, f"{vdir}/{c.replace('/', '__')}_{j}.png")
            entries.append({"crop": p, "top8_patches": coords})
        patches_viz[c] = entries
        # k-shot 原型样本也记录 (构造原型的局部依据)
        for j, p in enumerate(split["prototype_pools"][c][:2]):
            coords = topk_patch_overlay(p, bk[c], proj, f"{vdir}/{c.replace('/', '__')}_proto_{j}.png")
    rust["top_response_patches_patch_pre_k5_seed0"] = patches_viz
    C.save_json(rust, f"{OUT}/rust_analysis.json")
    print(f"[done] rust_analysis.json + overlays -> {vdir}")


if __name__ == "__main__":
    main()

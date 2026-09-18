#!/usr/bin/env python3
"""M3 Step-2 前置: 全量特征提取缓存 (一次前向, 三种特征).

缓存内容 (outputs/M3/cache/):
  - eval 清单: 全部 kshot + 缺陷查询 (3 个 k) + 正常查询 (3 个 k) 的并集 (去重)
  - gap_p5.npy  (N,256) fp32  —— actual-M2-v0 用 (model[:10]=SPPF, stride32)
  - gap_p3.npy  (N,64)  fp32  —— P3-GAP 用 (layer4, stride8)
  - patches.npy (N,40,40,64) fp16 memmap —— patch-pre / metric-v1 用
  - metric_patches.npy (M,16,64) fp16 + metric_labels.json —— 基类度量学习训练集
    (yolo_det train 22635 框 × 每框 16 个网格 patch, 类别均衡消费)

用法: python3 scripts/m3_patch_extractor.py --weights <best.pt>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C


def collect_eval_paths(splits):
    """全部评测裁剪 (缺陷+正常, 3 个 k) 的并集, 去重保序."""
    seen, paths = set(), []
    for k in C.SHOTS:
        with open(f"{splits}/query_k{k}.json", encoding="utf-8") as f:
            q = json.load(f)
        for d in [q["defect_query"], q["normal_query"]]:
            for ps in d.values():
                for p in ps:
                    if p not in seen:
                        seen.add(p)
                        paths.append(p)
    for k in C.SHOTS:
        for s in C.SEEDS:
            with open(f"{splits}/kshot_k{k}_seed{s}.json", encoding="utf-8") as f:
                sp = json.load(f)
            for ps in sp["prototype_pools"].values():
                for p in ps:
                    if p not in seen:
                        seen.add(p)
                        paths.append(p)
    return paths


def collect_metric_boxes(yolo_root, per_box=16, seed=0):
    """yolo_det train 框 → (图像路径, 裁剪框) 列表 + 类别标签."""
    import glob
    recs = []
    img_dir = os.path.join(yolo_root, "images", "train")
    lab_dir = os.path.join(yolo_root, "labels", "train")
    for lf in sorted(os.listdir(lab_dir)):
        base = os.path.splitext(lf)[0]
        for ext in (".jpg", ".jpeg", ".png"):
            ip = os.path.join(img_dir, base + ext)
            if os.path.isfile(ip):
                break
        else:
            continue
        from PIL import Image
        W, H = Image.open(ip).size
        for line in open(os.path.join(lab_dir, lf)):
            t = line.split()
            if len(t) != 5:
                continue
            cls, cx, cy, w, h = int(t[0]), *map(float, t[1:])
            x1 = max(0, int((cx - w / 2) * W)); y1 = max(0, int((cy - h / 2) * H))
            x2 = min(W, int((cx + w / 2) * W)); y2 = min(H, int((cy + h / 2) * H))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            recs.append({"img": ip, "box": [x1, y1, x2, y2], "cls": cls})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/root/autodl-tmp/runs/yolov8n_det/weights/best.pt")
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--yolo-root", default="/root/autodl-tmp/data/yolo_det")
    ap.add_argument("--out", default="/root/auto-exp/outputs/M3/cache")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb = C.Backbone(args.weights, dev)
    tf = C.make_transform(C.IMGSZ)
    from PIL import Image

    eval_paths = collect_eval_paths(args.splits)
    print(f"[extract] eval crops: {len(eval_paths)}")

    N, HP, WP, CH = len(eval_paths), C.IMGSZ // 8, C.IMGSZ // 8, 64
    gap5 = np.zeros((N, 256), dtype=np.float32)
    gap3 = np.zeros((N, 64), dtype=np.float32)
    patches = np.lib.format.open_memmap(f"{args.out}/patches.npy", mode="w+",
                                        dtype=np.float16, shape=(N, HP, WP, CH))

    def crop_tf(p):
        return tf(Image.open(p).convert("RGB"))

    bs = 64
    for i in range(0, N, bs):
        xb = torch.stack([crop_tf(p) for p in eval_paths[i:i + bs]])
        p3, p5 = bb.p3_and_p5gap(xb)
        gap5[i:i + bs] = torch.nn.functional.normalize(p5, dim=1).cpu().numpy()
        gap3[i:i + bs] = torch.nn.functional.normalize(p3.mean((2, 3)), dim=1).cpu().numpy()
        patches[i:i + bs] = (p3.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
        if (i // bs) % 20 == 0:
            print(f"  eval {i}/{N}")
    patches.flush()

    # ---- 基类训练框 patch 采样 (metric-v1 训练集) ----
    recs = collect_metric_boxes(args.yolo_root)
    print(f"[extract] metric boxes: {len(recs)}")
    M, PP = len(recs), C.METRIC_CFG["patches_per_box"]
    grid = HP * WP
    mpatches = np.zeros((M, PP, CH), dtype=np.float16)
    for i in range(0, M, bs):
        chunk = recs[i:i + bs]
        imgs = [tf(Image.open(r["img"]).convert("RGB").crop(tuple(r["box"]))) for r in chunk]
        xb = torch.stack(imgs)
        p3, _ = bb.p3_and_p5gap(xb)
        p3 = p3.permute(0, 2, 3, 1).cpu().numpy()  # (b,40,40,64)
        for j, r in enumerate(chunk):
            # 每框固定种子抽 16 个网格位 (框内容不同即天然多样)
            prng = np.random.default_rng(1000 + i + j)
            idx = prng.choice(grid, PP, replace=False)
            mpatches[i + j] = p3[j].reshape(grid, CH)[idx].astype(np.float16)
        if (i // bs) % 50 == 0:
            print(f"  metric {i}/{M}")
    labels = [r["cls"] for r in recs]
    np.save(f"{args.out}/metric_patches.npy", mpatches)
    with open(f"{args.out}/metric_labels.json", "w") as f:
        json.dump({"labels": labels, "grid": grid,
                   "patch_sample_rule": "per-box seed 1000+i, 16 of 1600 grid positions"}, f)

    np.save(f"{args.out}/gap_p5.npy", gap5)
    np.save(f"{args.out}/gap_p3.npy", gap3)
    with open(f"{args.out}/meta.json", "w", encoding="utf-8") as f:
        json.dump({"eval_paths": eval_paths, "metric_boxes": recs}, f, ensure_ascii=False)
    print(f"[done] cache -> {args.out}")


if __name__ == "__main__":
    main()

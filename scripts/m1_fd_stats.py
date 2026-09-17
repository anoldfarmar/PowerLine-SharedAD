#!/usr/bin/env python3
"""M1 数据核查：InsPLAD-FD 子集逐类实例统计。

用法: python3 scripts/m1_fd_stats.py --data /root/autodl-tmp/data/insplad \
        --out outputs/M1/insplad_fd_class_stats.csv [--coco]

产出:
  1. FD 子集 (defect_supervised) 逐 split/类 统计表
     列: split,class,n_images,n_instances,min_box_px
     (FD 为部件裁剪图, 1 图 = 1 实例; min_box_px = 裁剪框最小边长, 即图像最小边)
  2. [--coco] InsPLAD-det COCO 逐资产类框统计 (stdout 打印, 作为条件2证据)
"""
import argparse
import collections
import csv
import json
import os

IMG_EXT = (".jpg", ".jpeg", ".png")

# 标注命名归一化 (审计发现: 葡语命名/不同资产类别名不一致)
LABEL_NORMALIZE = {"corrosão": "corrosion", "normal": "good"}


def fd_stats(data_root):
    fd_root = os.path.join(data_root, "defect_supervised")
    if not os.path.isdir(fd_root):
        raise SystemExit(f"FD 子集不存在: {fd_root}")
    rows = []
    from PIL import Image

    for asset in sorted(os.listdir(fd_root)):
        asset_dir = os.path.join(fd_root, asset)
        if not os.path.isdir(asset_dir):
            continue
        for split in sorted(os.listdir(asset_dir)):
            split_dir = os.path.join(asset_dir, split)
            if split not in ("train", "val") or not os.path.isdir(split_dir):
                continue
            for label in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, label)
                if not os.path.isdir(cls_dir):
                    continue
                files = [f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXT)]
                min_side = None
                for f in files:
                    with Image.open(os.path.join(cls_dir, f)) as im:
                        s = min(im.size)
                    if min_side is None or s < min_side:
                        min_side = s
                cls_norm = f"{asset}/{LABEL_NORMALIZE.get(label, label)}"
                rows.append({
                    "split": split,
                    "class": cls_norm,
                    "n_images": len(files),
                    "n_instances": len(files),  # 部件裁剪: 1 图 = 1 实例
                    "min_box_px": min_side,
                })
    return rows


def coco_stats(data_root):
    """逐资产类框统计 (InsPLAD-det), 用于条件2 (总量支撑基类训练)."""
    for split in ("train", "val"):
        path = os.path.join(data_root, "annotations", f"instances_{split}.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        names = {c["id"]: c["name"] for c in d["categories"]}
        cnt = collections.Counter(names[a["category_id"]] for a in d["annotations"])
        print(f"[det/{split}] images={len(d['images'])} boxes={len(d['annotations'])}")
        for name, n in sorted(cnt.items(), key=lambda x: -x[1]):
            print(f"  {name}: {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/autodl-tmp/data/insplad")
    ap.add_argument("--out", default="outputs/M1/insplad_fd_class_stats.csv")
    ap.add_argument("--coco", action="store_true")
    args = ap.parse_args()

    rows = fd_stats(args.data)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["split", "class", "n_images", "n_instances", "min_box_px"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"[m1_fd_stats] 写出 {len(rows)} 行 -> {args.out}")

    if args.coco:
        coco_stats(args.data)


if __name__ == "__main__":
    main()

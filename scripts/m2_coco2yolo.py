#!/usr/bin/env python3
"""M2 Step-2: InsPLAD-det COCO -> ultralytics YOLO 格式转换 + FD 类名清洗清单.

用法: python3 scripts/m2_coco2yolo.py \
    --coco-dir /root/autodl-tmp/data/insplad/annotations \
    --img-root /root/autodl-tmp/data/insplad \
    --out /root/autodl-tmp/data/yolo_det

产出:
  <out>/images/{train,val}/   (符号链接, 不复制图像)
  <out>/labels/{train,val}/   (YOLO txt: cls cx cy w h, 归一化坐标)
  <out>/data.yaml             (ultralytics 数据集配置, 18 类)
stdout 打印转换统计 (用于 data_prep_report.md)
"""
import argparse
import collections
import json
import os


def convert_split(coco_path, img_root, out, split, image_id_to_split):
    with open(coco_path, encoding="utf-8") as f:
        d = json.load(f)
    cat_ids = sorted(c["id"] for c in d["categories"])
    cid2yolo = {cid: i for i, cid in enumerate(cat_ids)}
    names = {cid2yolo[c["id"]]: c["name"] for c in d["categories"]}

    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    img_dir = f"{out}/images/{split}"
    os.makedirs(img_dir, exist_ok=True)

    imgs = {im["id"]: im for im in d["images"]}
    per_img_boxes = collections.defaultdict(list)
    skipped = collections.Counter()
    for ann in d["annotations"]:
        im = imgs[ann["image_id"]]
        # COCO train 包含 46 组重复图像条目(不同 image_id 同文件名), 按文件名聚合防止覆盖丢框
        key = os.path.basename(im["file_name"])
        x, y, w, h = ann["bbox"]
        W, H = im["width"], im["height"]
        if w <= 0 or h <= 0 or x < -W * 0.5 or y < -H * 0.5:
            skipped[key] += 1
            continue
        # 裁剪到图像边界 (COCO bbox 允许越界)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, W), min(y + h, H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            skipped[key] += 1
            continue
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        nw, nh = (x1 - x0) / W, (y1 - y0) / H
        cls = cid2yolo[ann["category_id"]]
        per_img_boxes[key].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    n_linked, n_empty = 0, 0
    for im_id, im in imgs.items():
        base = os.path.basename(im["file_name"])
        src = os.path.join(img_root, split, base)
        if not os.path.isfile(src):
            # 部分图像不在对应 split 目录, 全库按文件名搜索 (数据包布局不一致, 记入报告)
            hits = [os.path.join(dp, f) for dp, _, fs in os.walk(img_root)
                    for f in fs if f == base]
            if not hits:
                skipped["MISSING:" + base] += 1
                continue
            src = hits[0]
        dst = os.path.join(img_dir, base)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(src), dst)
        n_linked += 1
        lines = per_img_boxes.get(base, [])
        if not lines:
            n_empty += 1  # 背景图: 生成空 label, YOLO 训练惯例
        stem = os.path.splitext(base)[0]
        with open(f"{out}/labels/{split}/{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        image_id_to_split[im_id] = split

    cls_counts = collections.Counter()
    for lines in per_img_boxes.values():
        for ln in lines:
            cls_counts[int(ln.split()[0])] += 1
    return names, len(imgs), n_linked, n_empty, cls_counts, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-dir", default="/root/autodl-tmp/data/insplad/annotations")
    ap.add_argument("--img-root", default="/root/autodl-tmp/data/insplad")
    ap.add_argument("--out", default="/root/autodl-tmp/data/yolo_det")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    image_id_to_split = {}

    stats = {}
    names_all = None
    for split in ("train", "val"):
        r = convert_split(
            os.path.join(args.coco_dir, f"instances_{split}.json"),
            args.img_root, args.out, split, image_id_to_split,
        )
        names, n_img, n_linked, n_empty, cls_counts, skipped = r
        names_all = names
        stats[split] = (n_img, n_linked, n_empty, cls_counts, dict(skipped))
        print(f"[{split}] coco_images={n_img} linked={n_linked} background={n_empty}")
        print(f"  boxes per class: {dict(sorted(cls_counts.items()))}")
        if skipped:
            print(f"  skipped_annotations={sum(v for k, v in skipped.items() if not k.startswith('MISSING'))}"
                  f" missing_images={sum(v for k, v in skipped.items() if k.startswith('MISSING'))}")

    # data.yaml (类名保持 COCO 原名; sphere/plate 类别稀少保留不动, 报告中说明)
    names_yml = "\n".join(f"  {i}: {n}" for i, n in sorted(names_all.items()))
    with open(f"{args.out}/data.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {args.out}\ntrain: images/train\nval: images/val\nnames:\n{names_yml}\n")
    print(f"[done] dataset -> {args.out} (data.yaml: {len(names_all)} classes)")


if __name__ == "__main__":
    main()

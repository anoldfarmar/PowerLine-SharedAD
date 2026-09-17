#!/usr/bin/env python3
"""M2 Step-5/6: 原型头 v0 初评 —— 冻结主干 + P3 特征 + 类级余弦原型头.

协议 (合同 steps[4,5]; 承 kshot_splits 协议):
  - 特征: 冻结 YOLOv8n 主干 (基类检测训练产物) 的 P3 特征图, 全局平均池化
  - 原型: 每个 资产×故障组合 的 k-shot 实例特征均值 (L2 归一化后余弦)
  - 打分: 裁剪图特征 vs 该组合原型 的余弦相似度
  - 指标: 逐组合 AUROC (缺陷查询 vs 同资产正常查询) + 固定规则召回
    (阈值 = 同资产正常打分 mean+2std, 无任何调参)
  - k ∈ {1,5,10} × 种子 {0,1,2}, mean±std

用法: python3 scripts/m2_proto_head_v0.py --weights <det_best.pt> \
        --splits /root/auto-exp/data/splits/insplad-fd \
        --fd-root /root/autodl-tmp/data/insplad/defect_supervised \
        --out outputs/M2
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1_fd_stats import LABEL_NORMALIZE  # 单一事实来源
from m2_kshot_splits import SRC_RE

SEEDS = [0, 1, 2]
SHOTS = [1, 5, 10]
COMBOS = ["glass-insulator/missing-cap", "lightning-rod-suspension/rust",
          "polymer-insulator-upper-shackle/corrosion", "polymer-insulator-upper-shackle/rust",
          "vari-grip/bird-nest", "vari-grip/rust", "yoke-suspension/rust"]
ASSET_OF = {c: c.split("/")[0] for c in COMBOS}


def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class P3Extractor:
    """冻结 YOLOv8 主干的 P3 特征提取器 (stride-8 层)."""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(weights).model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # P3 = 主干最后一层输出 (ultralytics yolov8: model.model[9] 为 stride-8 特征出口,
        # 取 [0:10] 前向; 具体索引在首次前向时自校验)
        self.backbone = self.model.model[:10]
        self.device = device
        self._c = None

    @torch.no_grad()
    def __call__(self, batch_img):  # batch_img: [B,3,H,W] 0-1
        x = batch_img.to(self.device)
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        v = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        v = F.normalize(v, dim=1)
        return v.cpu()


def make_transform(imgsz):
    from PIL import Image
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((imgsz, imgsz)),
        T.ToTensor(),
    ])


def extract_paths(model, paths, tf, bs=64):
    from PIL import Image
    vs = []
    for i in range(0, len(paths), bs):
        imgs = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]
        vs.append(model(torch.stack(imgs)))
    return torch.cat(vs) if vs else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="基类检测 best.pt")
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--fd-root", default="/root/autodl-tmp/data/insplad/defect_supervised")
    ap.add_argument("--out", default="outputs/M2")
    ap.add_argument("--imgsz", type=int, default=320)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    ext = P3Extractor(args.weights, dev)
    tf = make_transform(args.imgsz)

    def cache_key(p):
        return re.sub(r"[/ ]", "_", p.split("/insplad/")[-1])

    cache = {}
    def feat(paths):
        todo = [p for p in paths if cache_key(p) not in cache]
        for i in range(0, len(todo), 64):
            vs = extract_paths(ext, todo[i:i + 64], tf)
            for p, v in zip(todo[i:i + 64], vs):
                cache[cache_key(p)] = v
        return torch.stack([cache[cache_key(p)] for p in paths])

    results = {}
    for k in SHOTS:
        per_seed = []
        for seed in SEEDS:
            split = load_json(f"{args.splits}/kshot_k{k}_seed{seed}.json")
            query = load_json(f"{args.splits}/query_k{k}.json")
            protos = {}
            for c, paths in split["prototype_pools"].items():
                protos[c] = F.normalize(feat(paths).mean(0, keepdim=True), dim=1)
            normals = {a: feat(paths) for a, paths in query["normal_query"].items()}
            asset_norm = {}
            for akey, v in normals.items():
                asset_norm.setdefault(akey.split("/")[0], v)
            aurocs, recalls = {}, {}
            for c in COMBOS:
                asset = ASSET_OF[c]
                qd = feat(query["defect_query"][c])
                if qd.shape[0] == 0 or asset not in asset_norm or asset_norm[asset].shape[0] == 0:
                    print(f"[warn] skip {c}: qd={qd.shape[0]} normal_pool={'missing' if asset not in asset_norm else asset_norm[asset].shape[0]}")
                    continue
                s_d = (qd @ protos[c].T).flatten()
                s_n = (asset_norm[asset] @ protos[c].T).flatten()
                y = np.r_[np.ones(len(s_d)), np.zeros(len(s_n))]
                s = torch.cat([s_d, s_n]).numpy()
                # AUROC (rank-based)
                r = np.argsort(np.argsort(s)) + 1
                aurocs[c] = float((r[y == 1].sum() - len(s_d) * (len(s_d) + 1) / 2)
                                  / (len(s_d) * len(s_n)))
                thr = float(s_n.mean() + 2 * s_n.std())
                recalls[c] = float((s_d > thr).float().mean())
            per_seed.append({"auroc": aurocs, "recall": recalls,
                             "mean_auroc": float(np.mean(list(aurocs.values()))),
                             "mean_recall": float(np.mean(list(recalls.values())))})
        results[f"k{k}"] = per_seed
        ma = [s["mean_auroc"] for s in per_seed]
        mr = [s["mean_recall"] for s in per_seed]
        print(f"k={k}: AUROC {np.mean(ma):.4f}±{np.std(ma):.4f} | "
              f"recall {np.mean(mr):.4f}±{np.std(mr):.4f}")

    with open(f"{args.out}/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"task": "M2-proto-head-v0", "weights": args.weights,
                   "imgsz": args.imgsz, "seeds": len(SEEDS), "seed_ids": SEEDS,
                   "shots": SHOTS, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"[done] metrics -> {args.out}/metrics.json")


if __name__ == "__main__":
    main()

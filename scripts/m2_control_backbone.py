#!/usr/bin/env python3
"""M2 对照实验: COCO 预训练骨架 vs det 微调骨架 的原型判别力 (只读划分文件).

动机: v0 全量评测用 det best.pt 出现 AUROC<0.5 系统性倒挂;
PLAN §2 v0 规格为"冻结预训练主干", 需确认 det 微调是否损伤原型特征.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/auto-exp/scripts")
from m2_proto_head_v0 import P3Extractor, make_transform, ASSET_OF, extract_paths

SPLITS = "/root/auto-exp/data/splits/insplad-fd"
WEIGHTS = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/third_party/yolov8n.pt"
TAG = sys.argv[2] if len(sys.argv) > 2 else "coco"
COMBOS = ["glass-insulator/missing-cap", "vari-grip/bird-nest",
          "yoke-suspension/rust", "polymer-insulator-upper-shackle/corrosion"]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = P3Extractor(WEIGHTS, dev)
    tf = make_transform(320)
    cache = {}

    def feat(paths):
        todo = [p for p in paths if p not in cache]
        for i in range(0, len(todo), 64):
            vs = extract_paths(ext, todo[i:i + 64], tf)
            for p, v in zip(todo[i:i + 64], vs):
                cache[p] = v
        return torch.stack([cache[p] for p in paths])

    sp = json.load(open(f"{SPLITS}/kshot_k5_seed0.json"))
    q = json.load(open(f"{SPLITS}/query_k5.json"))
    norm = {a: feat(ps) for a, ps in q["normal_query"].items()}
    asset_norm = {}
    for a, v in norm.items():
        asset_norm.setdefault(a.split("/")[0], v)

    for c in COMBOS:
        proto = F.normalize(feat(sp["prototype_pools"][c]).mean(0, keepdim=True), dim=1)
        qd = feat(q["defect_query"][c])
        sn = asset_norm[ASSET_OF[c]]
        s = torch.cat([(qd @ proto.T).flatten(), (sn @ proto.T).flatten()]).numpy()
        y = np.r_[np.ones(len(qd)), np.zeros(len(sn))]
        r = np.argsort(np.argsort(s)) + 1
        auroc = (r[y == 1].sum() - len(qd) * (len(qd) + 1) / 2) / (len(qd) * len(sn))
        print(f"[{TAG}] {c}: AUROC={auroc:.4f} (n_defect={len(qd)}, n_norm={len(sn)})")


if __name__ == "__main__":
    main()

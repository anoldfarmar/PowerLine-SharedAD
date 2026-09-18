#!/usr/bin/env python3
"""M3 公共库: 特征提取 / 评测协议 / 哈希工具.

承 M2 冻结协议 (scripts/m2_proto_head_v0.py + data/splits/insplad-fd 清单):
  - 输入裁剪 resize 320x320, ToTensor, 无增强
  - AUROC = 逐组合 (缺陷查询 vs 同资产正常查询), rank-based
  - 召回阈值 = 同资产正常打分 mean+2std (固定规则, 零调参)
  - k ∈ {1,5,10} × 种子 {0,1,2}

特征层审计结论 (M3 pre-flight, 见 outputs/M3/feature_layer_audit.json):
  - M2 v0 的 model.model[:10] 实际输出 = layer 9 (SPPF), stride-32 P5, 256ch
    —— M2 报告标注的 "P3 (stride-8)" 为笔误级错误, M2 产物不改动, 以
      actual-M2-v0 名称如实登记
  - 真 P3 = layer 4 (C2f) 输出, 64ch, stride 8
"""
import hashlib
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

COMBOS = ["glass-insulator/missing-cap", "lightning-rod-suspension/rust",
          "polymer-insulator-upper-shackle/corrosion", "polymer-insulator-upper-shackle/rust",
          "vari-grip/bird-nest", "vari-grip/rust", "yoke-suspension/rust"]
RUST_COMBOS = ["lightning-rod-suspension/rust", "polymer-insulator-upper-shackle/corrosion",
               "polymer-insulator-upper-shackle/rust", "vari-grip/rust", "yoke-suspension/rust"]
ASSET_OF = {c: c.split("/")[0] for c in COMBOS}
SEEDS = [0, 1, 2]
SHOTS = [1, 5, 10]
IMGSZ = 320
P3_LAYER = 4          # 审计验证: stride 8, 64ch
P5_SPPF_LAYER = 9     # model[:10] 的实际输出层 (SPPF), stride 32, 256ch
# ---- preflight 固化决策 (写入 preflight.json 后才允许评测 query) ----
PATCH_DIM = 128       # patch 向量维度 (合同: 128 维)
PATCH_PROJ_SEED = 0   # 64 -> 128 固定种子随机正交投影
PATCH_TOPK = 8        # 图像级聚合 = 每图 patch-库最大余弦相似度的 top-8 均值
CORESET_CAP = 256     # 每资产×故障组合原型库上限 (合同)
CORESET_METHOD = "greedy_farthest_point"  # 固定种子 k-center 最远点采样
CORESET_SEED = 0
# metric-v1 训练超参数 (仅基类数据)
METRIC_CFG = {
    "input": "P3 patch (64d, stride 8)",
    "head": "Linear(64,128) BN ReLU Linear(128,128)",
    "embed_dim": 128,
    "loss": "supervised_contrastive",
    "temperature": 0.07,
    "epochs": 30,
    "patches_per_box": 16,
    "batch_boxes": 128,
    "balanced_classes": 18,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "seed": 0,
    "train_data": "yolo_det train boxes (18 基类资产, 22635 框) —— 不含目标故障 query/k-shot",
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify_split_hashes(splits_dir, manifest_hashes):
    """manifest_hashes: {文件名: 16位前缀}. 返回 (verified数, 不一致列表)."""
    ok, bad = 0, []
    for fn, expect in manifest_hashes.items():
        p = os.path.join(splits_dir, fn)
        if not os.path.isfile(p):
            bad.append((fn, "missing"))
            continue
        got = sha256_file(p)[:16]
        if got == expect:
            ok += 1
        else:
            bad.append((fn, f"got {got} expect {expect}"))
    return ok, bad


MANIFEST_HASHES = {
    "kshot_k1_seed0.json": "c5a94974bf91f576", "kshot_k5_seed2.json": "5558bc0c9b95c3a6",
    "kshot_k1_seed1.json": "938289d60545cf20", "kshot_k10_seed0.json": "934aa9fb0c75ad39",
    "kshot_k1_seed2.json": "49a53afcbe83c8fa", "kshot_k10_seed1.json": "b7887d6b8dd584a1",
    "kshot_k5_seed0.json": "79b672173e851273", "kshot_k10_seed2.json": "cb4483c98c955235",
    "kshot_k5_seed1.json": "26ef5014aa44bb03", "query_k1.json": "ac639b93fcb32599",
    "query_k5.json": "0fcc8c93bc8096ed", "query_k10.json": "1e17f5147b149a99",
}


class Backbone:
    """冻结 YOLOv8n 主干. 前向逐层执行, 可取 P3 (layer4) 与 P5 (layer9) 输出."""

    def __init__(self, weights, device):
        from ultralytics import YOLO
        self.model = YOLO(weights).model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.model[:10]  # 只前向到 SPPF, 与 M2 一致
        self.device = device

    @torch.no_grad()
    def forward_layers(self, x):
        """返回 {layer_idx: 特征图} (0..9)."""
        feats, cur = {}, x
        for i, layer in enumerate(self.layers):
            cur = layer(x) if i == 0 else layer(cur)
            if isinstance(cur, (list, tuple)):
                cur = cur[-1]
            feats[i] = cur
        return feats

    @torch.no_grad()
    def p3_and_p5gap(self, batch_img):
        """P3 特征图 (B,64,h,w) + P5 GAP (B,256) —— 一次前向."""
        x = batch_img.to(self.device)
        feats = self.forward_layers(x)
        p3 = feats[P3_LAYER]
        p5 = F.adaptive_avg_pool2d(feats[P5_SPPF_LAYER], 1).flatten(1)
        return p3, p5


def make_transform(imgsz=IMGSZ):
    from PIL import Image
    import torchvision.transforms as T
    return T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])


def load_batches(paths, tf, bs=64):
    from PIL import Image
    for i in range(0, len(paths), bs):
        yield torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])


def patch_projection_matrix(c_in=64, c_out=PATCH_DIM, seed=PATCH_PROJ_SEED):
    """固定种子随机半正交投影 64 -> 128 (仅初始化一次, 不参与任何训练)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(c_out, c_in, generator=g))
    return q.T.contiguous()  # (c_in, c_out)


def auroc_recall(s_d, s_n):
    """M2 协议: rank-based AUROC + 固定阈值召回 (正常分 mean+2std)."""
    s_d = np.asarray(s_d, dtype=np.float64)
    s_n = np.asarray(s_n, dtype=np.float64)
    s = np.concatenate([s_d, s_n])
    y = np.r_[np.ones(len(s_d)), np.zeros(len(s_n))]
    r = np.argsort(np.argsort(s)) + 1
    auroc = float((r[y == 1].sum() - len(s_d) * (len(s_d) + 1) / 2)
                  / (len(s_d) * len(s_n)))
    thr = float(s_n.mean() + 2 * s_n.std())
    recall = float((s_d > thr).mean())
    return auroc, recall, thr


def mean_std(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "per_seed": [float(v) for v in vals]}


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print(f"[saved] {path}")

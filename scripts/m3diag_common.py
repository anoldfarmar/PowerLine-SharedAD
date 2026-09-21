#!/usr/bin/env python3
"""M3DIAG 公共层: 冻结协议 (outputs/M3diag/frozen_protocol.json) 的代码化.
复用历史只读组件: m3_common (Backbone/make_transform/常量/固定正交投影),
m3_patch_extractor (collect_metric_boxes / collect_eval_paths, 保证框清单与评测清单和历史逐字节一致).
本文件不修改任何历史模块.
指标修正: auroc_avg_rank 用平均秩 (tie-corrected); 历史 m3_common.auroc_recall 是序数秩, 本任务不使用.
"""
import os, sys, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C
from m3_patch_extractor import collect_metric_boxes, collect_eval_paths

ROOT = "/root/auto-exp"
OUT = ROOT + "/outputs/M3diag"
CACHE = OUT + "/cache"
SPLITS = ROOT + "/data/splits/insplad-fd"
WEIGHTS = "/root/autodl-tmp/runs/yolov8n_det/weights/best.pt"
WEIGHTS_SHA_PREFIX = "ea1526c29d4afcfb"
VICREG_REF_SHA = "88c4439e2c508156"

COMBOS, ASSET_OF, SHOTS, SEEDS, IMGSZ = C.COMBOS, C.ASSET_OF, C.SHOTS, C.SEEDS, C.IMGSZ
CORESET_CAP, CORESET_SEED, PATCH_TOPK = C.CORESET_CAP, C.CORESET_SEED, C.PATCH_TOPK

TRAIN_CFG = {
    "epochs": 30, "batch_boxes": 128, "patches_per_box": 16,
    "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.0001,
    "embed_dim": 128, "mapper_train_seed": 0,
    "supcon_temperature": 0.07,
    "vicreg": {"sim": 25.0, "std": 25.0, "cov": 1.0, "var_eps": 0.0001,
               "ref": "facebookresearch/vicreg main_vicreg.py@main sha256_prefix=" + VICREG_REF_SHA},
    "aug": {"brightness": [0.9, 1.1], "contrast": [0.9, 1.1],
            "order": "brightness then contrast",
            "sampled": "per-box rng(2000+g) one-shot, fixed for all epochs",
            "forbidden": "random crop/flip/rotation/grayscale/strong hue/blur/CutOut/MixUp"},
    "positions": "per-box rng(1000+g) choose 16 of 1600 grid positions (same as historical)",
}

def sha256_file(p):
    return C.sha256_file(p)

def mapper_arch():
    """与历史 metric-v1 同构: Linear(64,128)-BN-ReLU-Linear(128,128)."""
    return torch.nn.Sequential(
        torch.nn.Linear(64, TRAIN_CFG["embed_dim"]),
        torch.nn.BatchNorm1d(TRAIN_CFG["embed_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(TRAIN_CFG["embed_dim"], TRAIN_CFG["embed_dim"]))

def supcon_loss(z, labels, temperature=0.07):
    """z: (B,d) 已 L2 归一化; labels: (B,). 复现 metric-v1 机制 (框级标签)."""
    sim = z @ z.T / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    eye = torch.eye(len(z), device=z.device)
    pos = mask - eye
    sim_max, _ = sim.max(1, keepdim=True)
    log_prob = sim - sim_max - torch.log(
        torch.exp(sim - sim_max).sum(1, keepdim=True) - torch.exp(sim - sim_max).diag()[:, None] + 1e-12)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    loss = -(log_prob * pos)[valid].sum(1)[valid] / pos_cnt[valid]
    return loss.mean()

def _off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_losses(z1, z2):
    """官方语义 (main_vicreg.py@main): 作用于 L2 归一化之前的投影输出.
    本任务样本单元 = patch 嵌入 (batch_boxes x 16 = 2048/步), 除子用实际批量 N-1."""
    vc = TRAIN_CFG["vicreg"]
    repr_loss = F.mse_loss(z1, z2)
    x = z1 - z1.mean(dim=0)
    y = z2 - z2.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + vc["var_eps"])
    std_y = torch.sqrt(y.var(dim=0) + vc["var_eps"])
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
    n = z1.shape[0]; d = z1.shape[1]
    cov_x = (x.T @ x) / (n - 1)
    cov_y = (y.T @ y) / (n - 1)
    cov_loss = _off_diagonal(cov_x).pow(2).sum() / d + _off_diagonal(cov_y).pow(2).sum() / d
    total = vc["sim"] * repr_loss + vc["std"] * std_loss + vc["cov"] * cov_loss
    return total, repr_loss, std_loss, cov_loss

def auroc_avg_rank(s_d, s_n):
    """平均秩 AUROC (同分取平均秩) + 状态标记. 返回 (auroc|None, flag)."""
    s_d = np.asarray(s_d, dtype=np.float64)
    s_n = np.asarray(s_n, dtype=np.float64)
    s = np.concatenate([s_d, s_n])
    nd, nn = len(s_d), len(s_n)
    if nd == 0 or nn == 0:
        return None, "single_class"
    if not np.isfinite(s).all():
        return None, "nonfinite"
    if float(np.ptp(s)) == 0.0:
        return None, "constant_score"
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    y = np.r_[np.ones(nd), np.zeros(nn)]
    auroc = float((ranks[y == 1].sum() - nd * (nd + 1) / 2) / (nd * nn))
    return auroc, "ok"

def frame_key(path):
    """(航次前缀, DJI编号) 帧标识; 跨包命名不一致的局限见协议 data_boundary."""
    import re
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(.*)_DJI_([0-9]+)(_.*)?$", stem)
    return (m.group(1), m.group(2)) if m else (stem, None)

def load_head_state(p, dev="cpu"):
    ck = torch.load(p, map_location=dev)
    head = mapper_arch()
    head.load_state_dict(ck["state_dict"])
    head.eval()
    return head

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    print("[saved] " + path)
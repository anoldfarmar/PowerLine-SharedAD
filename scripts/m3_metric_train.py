#!/usr/bin/env python3
"""M3 Step-4: metric-v1 —— 基类度量学习投影头 (128d).

训练数据: 仅 yolo_det train 框裁剪的 P3 patch 向量 (18 个基类资产, 22635 框,
每框 16 patch) —— 严禁使用目标故障 query / 目标故障 k-shot 样本反向传播
(preflight.json 固化决策, 训练集与冻结划分文件零交集)。

损失: 批内 supervised contrastive (温度 0.07), 类别均衡采样, 固定 seed=0。
产出: outputs/M3/metric_head.pt + outputs/M3/metric_train_log.json

用法: python3 scripts/m3_metric_train.py
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C

CACHE = "/root/auto-exp/outputs/M3/cache"


def supcon_loss(z, labels, temperature=0.07):
    """z: (B,d) 已 L2 归一化; labels: (B,). 标准监督对比损失."""
    sim = z @ z.T / temperature                                    # (B,B)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/auto-exp/outputs/M3")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(C.METRIC_CFG["seed"])
    np.random.seed(C.METRIC_CFG["seed"])

    X = np.load(f"{CACHE}/metric_patches.npy")           # (M,16,64) fp16
    labels = np.array(json.load(open(f"{CACHE}/metric_labels.json"))["labels"])
    X = torch.from_numpy(X).float()                      # (M,16,64)
    y = torch.from_numpy(labels)
    M, PP, CH = X.shape
    print(f"[metric] train patches: {M} boxes × {PP}, classes={y.unique().numel()}")

    head = torch.nn.Sequential(
        torch.nn.Linear(CH, C.METRIC_CFG["embed_dim"]),
        torch.nn.BatchNorm1d(C.METRIC_CFG["embed_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(C.METRIC_CFG["embed_dim"], C.METRIC_CFG["embed_dim"]),
    ).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=C.METRIC_CFG["lr"],
                            weight_decay=C.METRIC_CFG["weight_decay"])

    # 类别均衡: 每步每类采 box, 不足类全取
    by_cls = collections.defaultdict(list)
    for i, c in enumerate(labels):
        by_cls[int(c)].append(i)
    n_cls = len(by_cls)
    boxes_per_step = C.METRIC_CFG["batch_boxes"]
    per_cls = max(1, boxes_per_step // n_cls)
    steps_per_epoch = max(1, M // boxes_per_step)
    g = torch.Generator().manual_seed(C.METRIC_CFG["seed"])
    log = {"cfg": C.METRIC_CFG, "train_boxes": M, "epochs": [], "final_head": "metric_head.pt"}

    for ep in range(C.METRIC_CFG["epochs"]):
        perm = {c: torch.randperm(len(v), generator=g).tolist() for c, v in by_cls.items()}
        ptr = {c: 0 for c in by_cls}
        ep_loss, seen = 0.0, 0
        head.train()
        for _ in range(steps_per_epoch):
            batch = []
            for c in sorted(by_cls):
                v = by_cls[c]
                take = [v[(ptr[c] + j) % len(v)] for j in range(min(per_cls, len(v)))]
                ptr[c] = (ptr[c] + per_cls) % len(v)
                batch.extend(take)
            xb = X[batch].reshape(-1, CH).to(dev)
            yb = y[batch].repeat_interleave(PP).to(dev)
            z = F.normalize(head(xb), dim=1)
            loss = supcon_loss(z, yb, C.METRIC_CFG["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(batch)
            seen += len(batch)
        ep_loss /= seen
        log["epochs"].append({"epoch": ep, "supcon_loss": round(ep_loss, 5)})
        if ep % 5 == 0 or ep == C.METRIC_CFG["epochs"] - 1:
            print(f"[metric] epoch {ep}: supcon={ep_loss:.4f}")

    torch.save({"state_dict": head.state_dict(), "cfg": C.METRIC_CFG}, f"{args.out}/metric_head.pt")
    C.save_json(log, f"{args.out}/metric_train_log.json")
    print(f"[done] head -> {args.out}/metric_head.pt")


if __name__ == "__main__":
    main()

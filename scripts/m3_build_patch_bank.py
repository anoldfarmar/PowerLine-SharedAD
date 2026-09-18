#!/usr/bin/env python3
"""M3 Step-2/4: patch 原型库构建 (coreset 截断 ≤256×128d / 组合).

两种特征空间 (合同: 主聚合/coreset 方法已在 preflight.json 固化):
  - pre:    raw P3 patch (64d) → 固定种子随机正交投影 128d → L2 norm → coreset
  - metric: raw P3 patch (64d) → 训练好的度量投影头 128d → L2 norm → coreset
coreset = 贪心最远点采样 (k-center), 固定 seed=0, 上限 256 向量/组合。

产出: outputs/M3/banks/<variant>/bank_k{k}_seed{s}.npz
      outputs/M3/banks/<variant>/bank_manifest.json (库大小/哈希/内存)

用法: python3 scripts/m3_build_patch_bank.py --variant pre
      python3 scripts/m3_build_patch_bank.py --variant metric --head outputs/M3/metric_head.pt
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

CACHE = "/root/auto-exp/outputs/M3/cache"


def load_head(p):
    """度量投影头: Linear(64,128) BN ReLU Linear(128,128)."""
    ck = torch.load(p, map_location="cpu")
    head = torch.nn.Sequential(
        torch.nn.Linear(64, 128), torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(), torch.nn.Linear(128, 128))
    head.load_state_dict(ck["state_dict"])
    return head.eval()


class Projector:
    def __init__(self, variant, head_path=None):
        self.variant = variant
        if variant == "pre":
            self.proj = C.patch_projection_matrix()  # (64,128) 固定
            self.head = None
        elif variant == "metric":
            assert head_path, "metric 变体需要 --head"
            self.proj = None
            self.head = load_head(head_path)
        else:
            raise ValueError(variant)

    def __call__(self, patches_64d):
        """(N,64) -> (N,128) L2 归一化."""
        if self.proj is not None:
            v = patches_64d @ self.proj
        else:
            with torch.no_grad():
                v = self.head(patches_64d)
        return F.normalize(v, dim=1)


def coreset_farthest(x, cap, seed):
    """贪心最远点采样. x: (N,d) 已归一化 torch GPU/CPU. 返回选中索引 (升序)."""
    N = x.shape[0]
    if N <= cap:
        return list(range(N))
    g = torch.Generator().manual_seed(seed)
    chosen = [int(torch.randint(N, (1,), generator=g))]
    dmin = 1 - x @ x[chosen[0]]  # 余弦距离
    for _ in range(cap - 1):
        nxt = int(torch.argmax(dmin))
        chosen.append(nxt)
        dmin = torch.minimum(dmin, 1 - x @ x[nxt])
    return sorted(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["pre", "metric"], required=True)
    ap.add_argument("--head", default=None)
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--out", default="/root/auto-exp/outputs/M3/banks")
    args = ap.parse_args()

    meta = json.load(open(f"{CACHE}/meta.json", encoding="utf-8"))
    path2idx = {p: i for i, p in enumerate(meta["eval_paths"])}
    patches = np.load(f"{CACHE}/patches.npy", mmap_mode="r")  # (N,40,40,64) fp16
    proj = Projector(args.variant, args.head)
    outdir = f"{args.out}/{args.variant}"
    os.makedirs(outdir, exist_ok=True)

    manifest = {"variant": args.variant, "coreset": C.CORESET_METHOD,
                "coreset_cap": C.CORESET_CAP, "coreset_seed": C.CORESET_SEED,
                "banks": {}}
    for k in C.SHOTS:
        for s in C.SEEDS:
            with open(f"{args.splits}/kshot_k{k}_seed{s}.json", encoding="utf-8") as f:
                split = json.load(f)
            banks = {}
            for c, paths in split["prototype_pools"].items():
                idxs = [path2idx[p] for p in paths]
                raw = torch.from_numpy(
                    np.concatenate([np.asarray(patches[i]).reshape(-1, 64) for i in idxs])
                ).float()
                with torch.no_grad():
                    vecs = proj(raw)  # (k*1600, 128) L2 normed
                keep = coreset_farthest(vecs, C.CORESET_CAP, C.CORESET_SEED)
                bank = vecs[keep].cpu().numpy()
                banks[c] = bank
            np.savez_compressed(f"{outdir}/bank_k{k}_seed{s}.npz", **banks)
            manifest["banks"][f"k{k}_seed{s}"] = {
                c: int(v.shape[0]) for c, v in banks.items()}
            print(f"[bank] {args.variant} k={k} seed={s}: "
                  f"{ {c: v.shape[0] for c, v in banks.items()} }")

    total_vec = sum(sum(m.values()) for m in manifest["banks"].values())
    manifest["total_vectors"] = total_vec
    manifest["total_mem_mb_fp32"] = round(total_vec * 128 * 4 / 1e6, 2)
    C.save_json(manifest, f"{outdir}/bank_manifest.json")


if __name__ == "__main__":
    main()

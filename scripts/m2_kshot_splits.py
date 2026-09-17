#!/usr/bin/env python3
"""M2 Step-4: InsPLAD-FD k-shot 划分固化 ×3 种子.

设计 (合同 steps[3]):
  - 原型池: 每个资产×故障组合抽 k 个实例 (k ∈ {1,5,10}); 同源图像(同一 DJI 原图)的裁剪
    不得同时出现在原型池与查询集 —— 以源图像 id 为分组单元, k-shot 采样按"组"抽,
    不足 k 组时允许同组多实例并在清单中标记。
  - 查询集: 每个 k 固定 —— 缺陷池减去该 k 下 3 个种子的原型池并集 (跨种子可比)。
  - 正常查询池: FD val/good 按资产分层下采样 (上限 500/资产, 与缺陷查询量同量级), 固定 seed。
  - 类名归一化复用 scripts/m1_fd_stats.py 的 LABEL_NORMALIZE。

用法: python3 scripts/m2_kshot_splits.py \
    --fd-root /root/autodl-tmp/data/insplad/defect_supervised \
    --out /root/auto-exp/data/splits/insplad-fd

产出: kshot_k{k}_seed{s}.json ×9 + split_manifest.md 数据
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re

from m1_fd_stats import LABEL_NORMALIZE  # 单一事实来源: 类名归一化

IMG_EXT = (".jpg", ".jpeg", ".png")
SRC_RE = re.compile(r"(DJI_\d+)", re.I)  # 源图像 id, 防同源泄漏的分组键
NORMAL_QUERY_CAP = 500


def collect(fd_root):
    """返回 {combo(asset/label): [(file, src_group)]}, 仅故障类 (label != good)."""
    pools = collections.defaultdict(list)
    normals = collections.defaultdict(list)
    for asset in sorted(os.listdir(fd_root)):
        ad = os.path.join(fd_root, asset)
        if not os.path.isdir(ad):
            continue
        for split in ("train", "val"):
            sd = os.path.join(ad, split)
            if not os.path.isdir(sd):
                continue
            for label in sorted(os.listdir(sd)):
                ld = os.path.join(sd, label)
                if not os.path.isdir(ld):
                    continue
                lab = LABEL_NORMALIZE.get(label, label)
                for f in sorted(os.listdir(ld)):
                    if not f.lower().endswith(IMG_EXT):
                        continue
                    m = SRC_RE.search(f)
                    grp = m.group(1) if m else f  # 无源 id 时退化为文件级组
                    rec = (os.path.join(ld, f), grp)
                    if lab == "good":
                        normals[f"{asset}/good"].append(rec)
                    else:
                        pools[f"{asset}/{lab}"].append(rec)
    return pools, normals


def sample_k_grouped(recs, k, rng):
    """按源图像组抽取 k 个实例; 组内随机序, 组间随机序; 组数不足时允许同组多实例."""
    groups = collections.defaultdict(list)
    for f, g in recs:
        groups[g].append(f)
    keys = sorted(groups)
    rng.shuffle(keys)
    picked, used = [], set()
    for g in keys:
        need = k - len(picked)
        if need <= 0:
            break
        take = groups[g][:need]
        picked.extend(take)
        used.add(g)
    return sorted(picked), sorted(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd-root", default="/root/autodl-tmp/data/insplad/defect_supervised")
    ap.add_argument("--out", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pools, normals = collect(args.fd_root)
    manifest = ["# k-shot 划分清单 (M2 Step-4)", "",
                f"原型池采样单元=源图像组({SRC_RE.pattern}), 查询集按 k 固定(3 种子原型池并集之外的缺陷实例), "
                f"正常查询池=FD val/good 分层下采样 ≤{NORMAL_QUERY_CAP}/资产。", ""]
    manifest.append("| 组合 | 池规模 | 源图组数 |")
    manifest.append("|---|---|---|")
    for c in sorted(pools):
        gs = {g for _, g in pools[c]}
        manifest.append(f"| {c} | {len(pools[c])} | {len(gs)} |")
    manifest.append("")

    # 每个 k: 先固化 3 种子原型池, 再算查询集
    for k in sorted(set(args.shots)):
        proto_by_seed = {}
        for seed in sorted(set(args.seeds)):
            rng = random.Random(hashlib.sha256(f"insplad-fd-k{k}-s{seed}".encode()).hexdigest())
            p = {c: sample_k_grouped(recs, k, rng)[0] for c, recs in pools.items()}
            proto_by_seed[seed] = p
            path = os.path.join(args.out, f"kshot_k{k}_seed{seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"k": k, "seed": seed,
                           "prototype_pools": p,
                           "normal_query_pool": None,  # 查询集统一写 query 文件
                           }, f, ensure_ascii=False, indent=1)
        union_proto = {rec for p in proto_by_seed.values() for recs in p.values() for rec in recs}
        query = {c: sorted(f for f, _ in recs if f not in union_proto)
                 for c, recs in pools.items()}
        rng = random.Random(hashlib.sha256(f"insplad-fd-normal-q{k}".encode()).hexdigest())
        normal_q = {}
        for asset_pool in sorted(normals):
            recs = [f for f, _ in normals[asset_pool] if f not in union_proto]
            rng.shuffle(recs)
            normal_q[asset_pool] = sorted(recs[:NORMAL_QUERY_CAP])
        qpath = os.path.join(args.out, f"query_k{k}.json")
        with open(qpath, "w", encoding="utf-8") as f:
            json.dump({"k": k, "defect_query": query, "normal_query": normal_q},
                      f, ensure_ascii=False, indent=1)
        n_defect = sum(len(v) for v in query.values())
        n_norm = sum(len(v) for v in normal_q.values())
        manifest.append(f"- k={k}: 原型池/种子组合数={len(proto_by_seed[0])}×{len(args.seeds)}种子; "
                        f"缺陷查询={n_defect} 实例 / {len(query)} 组合; 正常查询={n_norm} 实例 / {len(normal_q)} 资产池")

    with open(os.path.join(args.out, "split_manifest_data.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    print("\n".join(manifest))
    print(f"[done] splits -> {args.out}")


if __name__ == "__main__":
    main()

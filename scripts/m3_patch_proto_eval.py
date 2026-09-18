#!/usr/bin/env python3
"""M3 Step-3/4: 冻结协议下四变体评测 → proto_v1_eval.json + memory_latency.json.

变体 (preflight.json 固化):
  actual-M2-v0       model[:10]=SPPF(stride32) GAP 256d 余弦原型
  P3-GAP             layer4(stride8) GAP 64d 余弦原型
  P3-patch-pre       P3 patch 64d→固定正交投影128d, coreset≤256, top-8 patch 相似度
  P3-patch-metric-v1 P3 patch 64d→度量学习头128d, coreset≤256, top-8 patch 相似度

协议: k∈{1,5,10}×种子{0,1,2}, 缺陷查询 vs 同资产正常查询, AUROC + 固定阈值召回
     (同资产正常分 mean+2std), 全量报告不挑选。

用法: python3 scripts/m3_patch_proto_eval.py --weights <best.pt>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C
from m3_build_patch_bank import Projector

CACHE = "/root/auto-exp/outputs/M3/cache"
BANKS = "/root/auto-exp/outputs/M3/banks"


def measure_latency(weights, out):
    """逐裁剪推理延迟: 特征提取 + 打分, GPU 与 CPU (RK3588 代理), batch=1."""
    tf = C.make_transform(C.IMGSZ)
    from PIL import Image
    meta = json.load(open(f"{CACHE}/meta.json", encoding="utf-8"))
    sample = meta["eval_paths"][:32]
    res = {}
    for devname, dev in [("gpu", "cuda"), ("cpu", "cpu")]:
        bb = C.Backbone(weights, dev)
        imgs = [tf(Image.open(p).convert("RGB")) for p in sample]
        # 预热
        for xb in imgs[:4]:
            bb.p3_and_p5gap(xb[None])
        t0 = time.perf_counter()
        for xb in imgs:
            bb.p3_and_p5gap(xb[None])
        res[f"feature_extraction_{devname}_ms_per_crop"] = \
            round((time.perf_counter() - t0) / len(imgs) * 1000, 2)
        del bb
        torch.cuda.empty_cache() if dev == "cuda" else None
    # patch 打分延迟 (独立于主干, 库形状与真实一致 256×128)
    patches = np.load(f"{CACHE}/patches.npy", mmap_mode="r")
    bank = np.random.randn(256, 128).astype(np.float32)
    proj_pre = Projector("pre")
    idxs = np.arange(16)
    for devname, dev in [("gpu", "cuda"), ("cpu", "cpu")]:
        proj_patch_scores("P3-patch-pre", proj_pre, idxs, bank, patches, dev)  # 预热
        t0 = time.perf_counter()
        for _ in range(3):
            proj_patch_scores("P3-patch-pre", proj_pre, idxs, bank, patches, dev)
        res[f"patch_scoring_{devname}_ms_per_crop"] = \
            round((time.perf_counter() - t0) / (3 * 16) * 1000, 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/root/autodl-tmp/runs/yolov8n_det/weights/best.pt")
    ap.add_argument("--splits", default="/root/auto-exp/data/splits/insplad-fd")
    ap.add_argument("--out", default="/root/auto-exp/outputs/M3")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(f"{CACHE}/meta.json", encoding="utf-8"))
    path2idx = {p: i for i, p in enumerate(meta["eval_paths"])}
    patches = np.load(f"{CACHE}/patches.npy", mmap_mode="r")

    gap5 = torch.from_numpy(np.load(f"{CACHE}/gap_p5.npy")).float()
    gap3 = torch.from_numpy(np.load(f"{CACHE}/gap_p3.npy")).float()
    proj_pre = Projector("pre")

    GAP_VAR = {"actual-M2-v0": gap5, "P3-GAP": gap3}

    results = {}   # variant -> k -> seed -> combo -> {auroc, recall}
    proto_stats = {}
    for variant in ["actual-M2-v0", "P3-GAP", "P3-patch-pre", "P3-patch-metric-v1"]:
        results[variant] = {}
        for k in C.SHOTS:
            results[variant][f"k{k}"] = []
            for s in C.SEEDS:
                with open(f"{args.splits}/kshot_k{k}_seed{s}.json", encoding="utf-8") as f:
                    split = json.load(f)
                with open(f"{args.splits}/query_k{k}.json", encoding="utf-8") as f:
                    query = json.load(f)
                # 原型 / 库
                protos = {}
                if variant in GAP_VAR:
                    g = GAP_VAR[variant]
                    for c, ps in split["prototype_pools"].items():
                        idxs = [path2idx[p] for p in ps]
                        protos[c] = F.normalize(g[idxs].mean(0, keepdim=True), dim=1).to(dev)
                else:
                    vkey = "pre" if variant == "P3-patch-pre" else "metric"
                    bk = np.load(f"{BANKS}/{vkey}/bank_k{k}_seed{s}.npz")
                    for c in C.COMBOS:
                        protos[c] = bk[c]
                # 正常查询索引 (同资产共用, 分数逐组合计算)
                norm_idx = {a: [path2idx[p] for p in ps]
                            for a, ps in query["normal_query"].items()}
                row = {}
                for c in C.COMBOS:
                    a = C.ASSET_OF[c]
                    qd_idx = [path2idx[p] for p in query["defect_query"][c]]
                    if len(qd_idx) == 0:
                        continue
                    if variant in GAP_VAR:
                        g = GAP_VAR[variant].to(dev)
                        s_d = (g[qd_idx] @ protos[c].T).squeeze(1).cpu().numpy()
                        s_n = (g[norm_idx[f"{a}/good"]] @ protos[c].T).squeeze(1).cpu().numpy()
                    else:
                        s_d = proj_patch_scores(variant, proj_pre, qd_idx, protos[c], patches, dev)
                        s_n = proj_patch_scores(variant, proj_pre, norm_idx[f"{a}/good"],
                                                protos[c], patches, dev)
                    auroc, recall, thr = C.auroc_recall(s_d, s_n)
                    row[c] = {"auroc": auroc, "recall": recall,
                              "n_query": len(qd_idx), "n_normal": len(s_n)}
                results[variant][f"k{k}"].append(row)
                ma = [v["auroc"] for v in row.values()]
                print(f"[eval] {variant} k={k} s={s}: mean AUROC {np.mean(ma):.4f}")

    # ---- 汇总统计 ----
    summary = {}
    for variant, per_k in results.items():
        summary[variant] = {}
        for k, rows in per_k.items():
            combos_seeds = {c: [r[c]["auroc"] for r in rows if c in r] for c in C.COMBOS}
            recalls = {c: [r[c]["recall"] for r in rows if c in r] for c in C.COMBOS}
            means = [np.mean([r[c]["auroc"] for c in r]) for r in rows]
            summary[variant][k] = {
                "mean_auroc": C.mean_std(means),
                "mean_recall": C.mean_std([np.mean([r[c]["recall"] for c in r]) for r in rows]),
                "per_combo_auroc": {c: C.mean_std(v) for c, v in combos_seeds.items()},
                "per_combo_recall": {c: C.mean_std(v) for c, v in recalls.items()},
                "coverage": {c: int(len(v)) for c, v in combos_seeds.items()},
            }

    # 原型库统计 + 有界内存合规
    bank_manifest = {}
    for vkey in ("pre", "metric"):
        p = f"{BANKS}/{vkey}/bank_manifest.json"
        if os.path.isfile(p):
            bank_manifest[vkey] = json.load(open(p, encoding="utf-8"))
    max_protos = max((max(m["banks"][kk].values()) for m in bank_manifest.values()
                      for kk in m["banks"]), default=0)
    bounded = {"proto_cap_per_combo": C.CORESET_CAP, "observed_max_protos": int(max_protos),
               "compliant": 1 if max_protos <= C.CORESET_CAP else 0,
               "bank_mem_mb_fp32": {v: m.get("total_mem_mb_fp32") for v, m in bank_manifest.items()},
               "gap_proto_bytes_per_combo": 256 * 4}

    ev = {
        "task": "M3-proto-head-v1",
        "weights": args.weights,
        "protocol": {"seed_count": len(C.SEEDS), "seed_ids": C.SEEDS,
                     "shot_count": len(C.SHOTS), "shots": C.SHOTS,
                     "threshold_rule": "同资产正常查询 mean+2std (固定, 零调参)",
                     "imgsz": C.IMGSZ,
                     "split_hashes": "12/12 verified (见 preflight.json)",
                     "variants": ["actual-M2-v0", "P3-GAP", "P3-patch-pre", "P3-patch-metric-v1"]},
        "variants_completed": 4,
        "coverage": {"fault_classes_nonempty": 7,
                     "combos": C.COMBOS,
                     "statistical_limitations": {
                         "polymer-insulator-upper-shackle/rust":
                             "33 实例, k=10 查询侧仅 3 实例, 统计意义弱 (冻结划分清单要求标注)"}},
        "bounded_memory": bounded,
        "summary": summary,
        "raw": results,
    }
    C.save_json(ev, f"{args.out}/proto_v1_eval.json")

    lat = measure_latency(args.weights, args.out)
    lat["bank_memory"] = bounded
    C.save_json({"task": "M3-proto-head-v1", "latency": lat,
                 "memory": bounded}, f"{args.out}/memory_latency.json")
    print("[done] proto_v1_eval.json + memory_latency.json")


def proj_patch_scores(variant, proj_pre, idxs, bank, patches, dev):
    """patch 变体打分 (eval 期). variant: P3-patch-pre / P3-patch-metric-v1."""
    raw = torch.from_numpy(np.asarray(patches[idxs]).reshape(len(idxs), -1, 64)).float()
    B, P, _ = raw.shape
    flat = raw.view(B * P, 64)
    if variant == "P3-patch-pre":
        pr = proj_pre.proj.to(dev)
        with torch.no_grad():
            v = F.normalize(flat.to(dev) @ pr, dim=1)
    else:
        from m3_build_patch_bank import load_head
        if not hasattr(proj_patch_scores, "_head"):
            proj_patch_scores._head = load_head(
                "/root/auto-exp/outputs/M3/metric_head.pt").to(dev)
        with torch.no_grad():
            v = F.normalize(proj_patch_scores._head(flat.to(dev)), dim=1)
    v = v.view(B, P, -1)
    bk = torch.from_numpy(np.asarray(bank)).float().to(dev)
    with torch.no_grad():
        sims = torch.einsum("bpd,qd->bpq", v, bk).max(-1).values
        return torch.topk(sims, min(C.PATCH_TOPK, P), dim=1).values.mean(1).cpu().numpy()


if __name__ == "__main__":
    main()

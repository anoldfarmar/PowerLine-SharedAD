#!/usr/bin/env python3
"""M3DIAG 评测: A/B/C 三变体 -> banks, 原始分数, diag_eval.json, vis.
协议: coreset<=256 (greedy FPS seed 0), 余弦, 全 1600 patch, top-8 聚合;
AUROC = 平均秩 (修正历史序数秩同分隐患); 召回规则沿用 M2 冻结 mean+2std;
原始逐图分数全部保存; 支持与查询重叠逐配置检查; 常数分/单标签显式标记不硬算.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
from m3_build_patch_bank import coreset_farthest

HP = D.IMGSZ // 8

def get_spec(variant, heads):
    if variant == "A":
        return ("proj", torch.from_numpy(np.asarray(C.patch_projection_matrix())).float())
    return ("head", heads[variant])

def embed(spec, raw64, dev):
    kind, obj = spec
    with torch.no_grad():
        if kind == "proj":
            z = raw64.to(dev) @ obj.to(dev)
        else:
            z = obj(raw64.to(dev))
    return F.normalize(z, dim=1)

def score_crops(patches, idxs, spec, bank, dev, chunk=128):
    out = []
    bk = torch.from_numpy(np.asarray(bank)).float().to(dev)
    for i in range(0, len(idxs), chunk):
        ids = idxs[i:i + chunk]
        raw = torch.from_numpy(np.asarray(patches[ids]).reshape(len(ids), -1, 64)).float()
        Bn, Pn, _ = raw.shape
        v = embed(spec, raw.view(-1, 64), dev).view(Bn, Pn, -1)
        with torch.no_grad():
            sims = torch.einsum("bpd,qd->bpq", v, bk).max(-1).values
            s = torch.topk(sims, min(D.PATCH_TOPK, Pn), dim=1).values.mean(1)
        out.append(s.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "需要 GPU"
    variants = args.variants.split(",")

    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    heads = {}
    if "B" in variants:
        heads["B"] = D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev)
    if "C" in variants:
        heads["C"] = D.load_head_state(D.OUT + "/mapper_C.pt", dev).to(dev)
    specs = {v: get_spec(v, heads) for v in variants}

    results = {}
    overlap_log = {}
    raw_dir = D.OUT + "/raw_scores"
    os.makedirs(raw_dir, exist_ok=True)
    for v in variants:
        results[v] = {}
        for k in D.SHOTS:
            results[v]["k%d" % k] = []
            for s in D.SEEDS:
                with open(D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s), encoding="utf-8") as f:
                    split = json.load(f)
                with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
                    query = json.load(f)
                banks = {}
                for c, ps in split["prototype_pools"].items():
                    idxs = [path2idx[p] for p in ps]
                    raw = torch.from_numpy(
                        np.asarray(patches[idxs]).reshape(-1, 64)).float()
                    vecs = embed(specs[v], raw, dev)
                    keep = coreset_farthest(vecs, D.CORESET_CAP, D.CORESET_SEED)
                    banks[c] = vecs[keep].cpu().numpy()
                norm_idx = {a: [path2idx[p] for p in ps]
                            for a, ps in query["normal_query"].items()}
                row = {}
                raw_out = {}
                ov_paths = 0; ov_frames = 0; sup_all = []
                for c in D.COMBOS:
                    a = D.ASSET_OF[c]
                    qd = query["defect_query"].get(c, [])
                    if len(qd) == 0:
                        continue
                    qd_idx = [path2idx[p] for p in qd]
                    qn_idx = norm_idx[a + "/good"]
                    sup_paths = split["prototype_pools"][c]
                    sup_all += sup_paths
                    ov_paths += len(set(sup_paths) & set(qd))
                    ov_frames += len({D.frame_key(p) for p in sup_paths} &
                                     {D.frame_key(p) for p in qd})
                    s_d = score_crops(patches, qd_idx, specs[v], banks[c], dev)
                    s_n = score_crops(patches, qn_idx, specs[v], banks[c], dev)
                    auroc, flag = D.auroc_avg_rank(s_d, s_n)
                    thr = None; recall = None
                    if flag == "ok":
                        thr = float(np.mean(s_n) + 2 * np.std(s_n))
                        recall = float((s_d > thr).mean())
                    row[c] = {"auroc": auroc, "flag": flag, "recall": recall,
                              "n_query": len(qd), "n_normal": len(qn_idx)}
                    raw_out[c] = {"defect": [[p, float(x)] for p, x in zip(qd, s_d)],
                                  "normal": [[p, float(x)] for p, x in
                                             zip([eval_paths[i] for i in qn_idx], s_n)]}
                overlap_log["%s_k%d_s%d" % (v, k, s)] = {
                    "support_path_in_query": ov_paths,
                    "support_frame_in_query": ov_frames,
                    "support_crops": len(sup_all),
                    "note": "path级重叠必须为0; 源帧重叠为M2既定继承边界(见协议)"}
                D.save_json(raw_out, raw_dir + "/%s_k%d_seed%d.json" % (v, k, s))
                results[v]["k%d" % k].append(row)
                ma = [x["auroc"] for x in row.values() if x["auroc"] is not None]
                print("[eval] %s k=%d s=%d mean AUROC %.4f" % (v, k, s, float(np.mean(ma))))

    summary = {"variants_completed": variants,
               "query_leakage_check_passed": int(all(
                   o["support_path_in_query"] == 0 for o in overlap_log.values())),
               "overlap_log": overlap_log,
               "metric_implementation": "average-rank AUROC (M2/M3 用序数秩; 差异含同分处理影响, 已分开报告)",
               "per_variant": {}}
    for v in variants:
        pv = {}
        for k in D.SHOTS:
            rows = results[v]["k%d" % k]
            combos_seeds = {c: [r[c]["auroc"] for r in rows if c in r] for c in D.COMBOS}
            pv["k%d" % k] = {
                "per_combo_auroc": {c: {"mean": None if any(x is None for x in vals) else float(np.mean(vals)),
                                        "std": None if any(x is None for x in vals) else float(np.std(vals)),
                                        "per_seed": vals,
                                        "flags": [r[c]["flag"] for r in rows if c in r]}
                                     for c, vals in combos_seeds.items()},
                "mean_over_combos_per_seed": [float(np.mean([r[c]["auroc"] for c in r
                                                             if r[c]["auroc"] is not None]))
                                              for r in rows]}
        summary["per_variant"][v] = pv
    if all(v in results for v in ["A", "B", "C"]):
        diffs = {"C-B": {}, "B-A": {}, "C-A": {}}
        for k in D.SHOTS:
            for c in D.COMBOS:
                for si in range(3):
                    def g(v):
                        rr = results[v]["k%d" % k][si]
                        return rr[c]["auroc"] if c in rr and rr[c]["auroc"] is not None else None
                    va, vb, vc = g("A"), g("B"), g("C")
                    if None in (va, vb, vc):
                        continue
                    for key, val in (("C-B", vc - vb), ("B-A", vb - va), ("C-A", vc - va)):
                        diffs[key].setdefault("k%d" % k, {}).setdefault(c, []).append(val)
        for key in diffs:
            for kk in diffs[key]:
                for c in diffs[key][kk]:
                    vals = diffs[key][kk][c]
                    diffs[key][kk][c] = {"mean": float(np.mean(vals)),
                                         "std": float(np.std(vals)), "per_seed": vals}
        summary["paired_diffs"] = diffs
    D.save_json(summary, D.OUT + "/diag_eval.json")

    if "C" in variants:
        make_vis(path2idx, patches, specs["C"], dev)
    print("[done] diag_eval.json")

def make_vis(path2idx, patches, spec, dev):
    from PIL import Image
    os.makedirs(D.OUT + "/vis", exist_ok=True)
    with open(D.SPLITS + "/kshot_k5_seed0.json", encoding="utf-8") as f:
        split = json.load(f)
    with open(D.SPLITS + "/query_k1.json", encoding="utf-8") as f:
        query = json.load(f)
    for c in D.COMBOS:
        ps = split["prototype_pools"][c]
        idxs = [path2idx[p] for p in ps]
        raw = torch.from_numpy(np.asarray(patches[idxs]).reshape(-1, 64)).float()
        vecs = embed(spec, raw, dev)
        bank = vecs[coreset_farthest(vecs, D.CORESET_CAP, D.CORESET_SEED)].cpu().numpy()
        sample_path = sorted(query["defect_query"][c])[0]
        i = path2idx[sample_path]
        p = torch.from_numpy(np.asarray(patches[i]).reshape(-1, 64)).float()
        vv = embed(spec, p, dev)
        with torch.no_grad():
            sims = (vv @ torch.from_numpy(bank).float().to(dev).T).max(-1).values
        mp = sims.cpu().numpy().reshape(HP, HP)
        mp = (mp - mp.min()) / (mp.max() - mp.min() + 1e-12)
        heat = np.zeros((HP, HP, 3), dtype=np.uint8)
        heat[..., 0] = (255 * mp).astype(np.uint8)
        heat[..., 1] = (80 * (1 - mp)).astype(np.uint8)
        heat[..., 2] = (30 * (1 - mp)).astype(np.uint8)
        him = Image.fromarray(heat).resize((320, 320), Image.NEAREST)
        orig = Image.open(sample_path).convert("RGB").resize((320, 320))
        canvas = Image.new("RGB", (650, 320), (255, 255, 255))
        canvas.paste(orig, (0, 0)); canvas.paste(him, (330, 0))
        canvas.save(D.OUT + "/vis/%s.png" % c.replace("/", "__"))
    print("[vis] 7 combos saved (原图 | C 库最高响应热图; 无像素GT, 仅观察)")

if __name__ == "__main__":
    main()
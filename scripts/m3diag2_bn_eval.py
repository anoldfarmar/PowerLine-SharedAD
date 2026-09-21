#!/usr/bin/env python3
"""M3DIAG2 stage-3 eval: C (original) and C2 (BN recalibrated), frozen pipeline."""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D

OUT2 = D.ROOT + "/outputs/M3diag2"
os.makedirs(OUT2, exist_ok=True)
from m3diag_bank_eval import embed, score_crops
from m3_build_patch_bank import coreset_farthest

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "need GPU"
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    specs = {
        "C": ("head", D.load_head_state(D.OUT + "/mapper_C.pt", dev).to(dev)),
        "C2": ("head", D.load_head_state(OUT2 + "/mapper_C_bnrecal.pt", dev).to(dev)),
    }
    raw_dir = OUT2 + "/raw_scores2"
    os.makedirs(raw_dir, exist_ok=True)
    results = {}
    for v in ("C", "C2"):
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
                    raw = torch.from_numpy(np.asarray(patches[idxs]).reshape(-1, 64)).float()
                    vecs = embed(specs[v], raw, dev)
                    banks[c] = vecs[coreset_farthest(vecs, D.CORESET_CAP, D.CORESET_SEED)].cpu().numpy()
                norm_idx = {a: [path2idx[p] for p in ps] for a, ps in query["normal_query"].items()}
                row = {}
                raw_out = {}
                for c in D.COMBOS:
                    a = D.ASSET_OF[c]
                    qd = query["defect_query"].get(c, [])
                    if len(qd) == 0:
                        continue
                    qd_idx = [path2idx[p] for p in qd]
                    qn_idx = norm_idx[a + "/good"]
                    s_d = score_crops(patches, qd_idx, specs[v], banks[c], dev)
                    s_n = score_crops(patches, qn_idx, specs[v], banks[c], dev)
                    auroc, flag = D.auroc_avg_rank(s_d, s_n)
                    thr = float(np.mean(s_n) + 2 * np.std(s_n)) if flag == "ok" else None
                    recall = float((s_d > thr).mean()) if flag == "ok" else None
                    row[c] = {"auroc": auroc, "flag": flag, "recall": recall,
                              "n_query": len(qd), "n_normal": len(qn_idx)}
                    raw_out[c] = {"defect": [[p, float(x)] for p, x in zip(qd, s_d)],
                                  "normal": [[p, float(x)] for p, x in
                                             zip([eval_paths[i] for i in qn_idx], s_n)]}
                D.save_json(raw_out, raw_dir + "/%s_k%d_seed%d.json" % (v, k, s))
                results[v]["k%d" % k].append(row)
                ma = [x["auroc"] for x in row.values() if x["auroc"] is not None]
                print("[bn-eval] %s k=%d s=%d mean AUROC %.4f" % (v, k, s, float(np.mean(ma))))
    out = {"variants": ["C", "C2"], "std_ddof": 0, "per_variant": {}}
    for v in ("C", "C2"):
        pv = {}
        for k in D.SHOTS:
            rows = results[v]["k%d" % k]
            pv["k%d" % k] = {
                "per_combo_auroc": {c: {"mean": float(np.mean([r[c]["auroc"] for r in rows if c in r and r[c]["auroc"] is not None])),
                                        "per_seed": [r[c]["auroc"] for r in rows if c in r],
                                        "flags": [r[c]["flag"] for r in rows if c in r]}
                                     for c in D.COMBOS if c in rows[0]},
                "mean_over_combos_per_seed": [float(np.mean([r[c]["auroc"] for c in r if r[c]["auroc"] is not None])) for r in rows]}
        out["per_variant"][v] = pv
    diffs = {}
    for k in D.SHOTS:
        for c in D.COMBOS:
            for si in range(3):
                va = results["C"]["k%d" % k][si][c]["auroc"]
                vb = results["C2"]["k%d" % k][si][c]["auroc"]
                if va is None or vb is None:
                    continue
                diffs.setdefault("k%d" % k, {}).setdefault(c, []).append(vb - va)
    out["paired_diff_C2_minus_C"] = {}
    for kk in diffs:
        out["paired_diff_C2_minus_C"][kk] = {}
        for c in diffs[kk]:
            v = diffs[kk][c]
            out["paired_diff_C2_minus_C"][kk][c] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}
    D.save_json(out, OUT2 + "/diag_eval_bn.json")
    print("[done] diag_eval_bn.json")

if __name__ == "__main__":
    main()
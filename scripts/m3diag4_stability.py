#!/usr/bin/env python3
"""M3DIAG4: P1 normal-reference sampling stability (contract v1). Only variable: ref sampling seed (3001-3005).
Order: checks -> seed3000 repro gate (<=1e-4 vs M3DIAG3 P1) -> build 5 new group banks -> 60 configs -> aggregate.
seed3000 excluded from main summary. P0/P-random AUROCs recomputed from M3DIAG3 raw scores, not re-run.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
import m3diag3_faultbank as F3
from m3diag_bank_eval import embed, score_crops
from m3_build_patch_bank import coreset_farthest

OUT3 = D.ROOT + "/outputs/M3diag3"
OUT4 = D.ROOT + "/outputs/M3diag4"
COMBOS = F3.COMBOS
KS = [5, 10]
SUPPORT_SEEDS = [0, 1, 2]
NEW_SEEDS = [3001, 3002, 3003, 3004, 3005]
REPRO_TOL = 1e-4

def load_split(combo, a, k, s2, path2idx):
    with open(D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s2), encoding="utf-8") as f:
        split = json.load(f)
    with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
        query = json.load(f)
    sup = split["prototype_pools"][combo]
    qd = query["defect_query"][combo]
    qn = query["normal_query"][a + "/good"]
    return sup, qd, qn, [path2idx[p] for p in qd], [path2idx[p] for p in qn]

def p1_bank_and_scores(patches, vecs, nb, qd_idx, qn_idx, spec, dev):
    s_norm = 1.0 - (vecs @ nb.T).max(1).values
    assert torch.isfinite(s_norm).all()
    keep = torch.argsort(-s_norm, stable=True)[:vecs.shape[0] // 2]
    v = vecs[keep]
    bank = v[torch.as_tensor(coreset_farthest(v, 256, F3.FPS_SEED), device=dev)].cpu().numpy()
    return score_crops(patches, qd_idx, spec, bank, dev), score_crops(patches, qn_idx, spec, bank, dev)

def auroc_from_raw(rec):
    sd = np.array([x for _, x in rec["defect"]], float)
    sn = np.array([x for _, x in rec["normal"]], float)
    a, fl = D.auroc_avg_rank(sd, sn)
    return a, fl

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "need GPU"
    os.makedirs(OUT4 + "/raw_scores4", exist_ok=True)
    assert D.sha256_file(D.WEIGHTS).startswith(D.WEIGHTS_SHA_PREFIX)
    ok, bad = C.verify_split_hashes(D.SPLITS, C.MANIFEST_HASHES)
    assert ok and not bad, bad
    rg = json.load(open(OUT4 + "/ref_groups.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    print("[m4] checks ok")

    def build_bank(paths):
        arr = F3.extract_crops(paths, dev)
        vecs = embed(spec, torch.from_numpy(arr.reshape(-1, F3.CH)).float(), dev)
        assert torch.isfinite(vecs).all()
        return vecs[torch.as_tensor(coreset_farthest(vecs, 256, F3.FPS_SEED), device=dev)]

    # --- seed3000 repro gate (vs archived M3DIAG3 P1 scores) ---
    repro_max, n_repro = 0.0, 0
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        nb = build_bank(json.load(open(OUT3 + "/normal_ref.json"))["assets"][a]["selected"])
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                sup, qd, qn, qd_idx, qn_idx = load_split(combo, a, k, s2, path2idx)
                s_d, s_n = p1_bank_and_scores(patches, embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, F3.CH)).float(), dev), nb, qd_idx, qn_idx, spec, dev)
                old = json.load(open(OUT3 + "/raw_scores3/P1_%s__k%d_seed%d.json" % (combo.replace("/", "__"), k, s2)))
                for side, nw in (("defect", dict(zip(qd, map(float, s_d)))), ("normal", dict(zip(qn, map(float, s_n))))):
                    o = {p: x for p, x in old[side]}
                    assert set(o) == set(nw), (combo, k, s2, side)
                    repro_max = max(repro_max, max(abs(o[p] - nw[p]) for p in o))
                    n_repro += 1
    assert repro_max <= REPRO_TOL, "P1 seed3000 repro failed: %.2e" % repro_max
    print("[m4] repro gate seed3000: %d path-checks, max|diff|=%.2e" % (n_repro, repro_max))

    # --- new reference group banks (dedup crops, extract once) ---
    banks = {}
    for a in rg["assets"]:
        uniq = sorted({f for sd in NEW_SEEDS for f in rg["assets"][a]["groups"][str(sd)]})
        pool = embed(spec, torch.from_numpy(F3.extract_crops(uniq, dev).reshape(-1, F3.CH)).float(), dev)
        base = {f: i * 1600 for i, f in enumerate(uniq)}
        for sd in NEW_SEEDS:
            rows = torch.tensor([base[f] + o for f in rg["assets"][a]["groups"][str(sd)] for o in range(1600)], device=dev)
            sub = pool[rows]
            banks[(a, sd)] = sub[torch.as_tensor(coreset_farthest(sub, 256, F3.FPS_SEED), device=dev)]
        print("[m4] %s: %d unique crops -> 5 banks" % (a[:12], len(uniq)))

    # --- 60 configs ---
    results = {}
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                sup, qd, qn, qd_idx, qn_idx = load_split(combo, a, k, s2, path2idx)
                vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup]]).reshape(-1, F3.CH)).float(), dev)
                for sd in NEW_SEEDS:
                    s_d, s_n = p1_bank_and_scores(patches, vecs, banks[(a, sd)], qd_idx, qn_idx, spec, dev)
                    auroc, flag = D.auroc_avg_rank(s_d, s_n)
                    results[(combo, k, s2, sd)] = auroc
                    D.save_json({"defect": [[p, float(x)] for p, x in zip(qd, s_d)],
                                 "normal": [[p, float(x)] for p, x in zip(qn, s_n)]},
                                OUT4 + "/raw_scores4/P1_ref%d_%s__k%d_seed%d.json" % (sd, combo.replace("/", "__"), k, s2))
                print("[m4] %s k=%d s=%d: %s" % (combo[:12], k, s2,
                    {sd: round(results[(combo, k, s2, sd)], 4) for sd in NEW_SEEDS}))

    # --- reuse controls: recompute P0 / P-random AUROCs from M3DIAG3 raw ---
    base_auroc = {}
    for combo in COMBOS:
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                for arm in ("P0", "P-random"):
                    rec = json.load(open(OUT3 + "/raw_scores3/%s_%s__k%d_seed%d.json" % (arm, combo.replace("/", "__"), k, s2)))
                    auroc, _ = auroc_from_raw(rec)
                    base_auroc[(arm, combo, k, s2)] = auroc
    print("[m4] controls reused (P0/P-random recomputed from archived raw scores)")

    # --- aggregate ---
    summ = {"ref_groups_completed": len(NEW_SEEDS), "new_seeds": NEW_SEEDS, "std_ddof": 0,
            "repro_gate": {"n_path_checks": n_repro, "max_abs_diff": repro_max, "tol": REPRO_TOL, "passed": True},
            "note": "exploratory; P0/P-random reused (not new experiments); seed3000 excluded from main summary; glass pool 21 crops, group overlap unavoidable",
            "per_config": {}, "per_ref_group": {}, "group_stats": {}, "volatility": {}}
    for combo in COMBOS:
        for k in KS:
            for s2 in SUPPORT_SEEDS:
                cfg = "%s__k%d_seed%d" % (combo.replace("/", "__"), k, s2)
                summ["per_config"][cfg] = {
                    "P1_by_ref": {str(sd): results[(combo, k, s2, sd)] for sd in NEW_SEEDS},
                    "P0": base_auroc[("P0", combo, k, s2)],
                    "P-random": base_auroc[("P-random", combo, k, s2)]}
    for combo in COMBOS:
        for k in KS:
            key = "%s__k%d" % (combo.replace("/", "__"), k)
            summ["per_ref_group"][key] = {}
            summ["group_stats"][key] = {}
            for name, arm in (("P1_minus_P0", "P0"), ("P1_minus_Prandom", "P-random")):
                per_group = {}
                for sd in NEW_SEEDS:
                    ds = [results[(combo, k, s2, sd)] - base_auroc[(arm, combo, k, s2)] for s2 in SUPPORT_SEEDS]
                    per_group[str(sd)] = {"per_seed": ds, "mean": float(np.mean(ds))}
                summ["per_ref_group"][key][name] = per_group
                gmeans = [per_group[str(sd)]["mean"] for sd in NEW_SEEDS]
                summ["group_stats"][key][name] = {"mean": float(np.mean(gmeans)), "median": float(np.median(gmeans)),
                    "min": float(np.min(gmeans)), "max": float(np.max(gmeans)),
                    "n_pos_groups": int(sum(1 for v in gmeans if v > 0))}
    for combo in COMBOS:
        for k in KS:
            key = "%s__k%d" % (combo.replace("/", "__"), k)
            summ["volatility"][key] = {}
            for s2 in SUPPORT_SEEDS:
                aurocs = [results[(combo, k, s2, sd)] for sd in NEW_SEEDS]
                summ["volatility"][key][str(s2)] = {"std_across_refs": float(np.std(aurocs)),
                    "range": float(np.max(aurocs) - np.min(aurocs))}
    D.save_json(summ, OUT4 + "/stability_eval.json")
    print("[m4] stability_eval.json saved")

if __name__ == "__main__":
    main()
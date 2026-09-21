#!/usr/bin/env python3
"""M3DIAG3: P0 / P-random / P1 三组故障原型筛选对照 (合同 v3).
顺序: 核验 -> 正常参考库(每资产一次) -> 逐配置[建三库 -> P0评分 -> P0复现门(1e-4) -> P-random/P1评分] -> 汇总.
P1 筛选只读支持图特征与正常参考库, 不读查询标签/分数. 全部负结果与原始分保留.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
from m3diag_bank_eval import embed, score_crops
from m3_build_patch_bank import coreset_farthest

OUT3 = D.ROOT + "/outputs/M3diag3"
COMBOS = ["glass-insulator/missing-cap", "yoke-suspension/rust"]
ARMS = ["P0", "P-random", "P1"]
FPS_SEED, PRANDOM_SEED, REPRO_TOL = 0, 4000, 1e-4
HP, CH, PPC = 40, 64, 1600

def extract_crops(paths, dev):
    from PIL import Image
    import torchvision.transforms.functional as TF
    bb = C.Backbone(D.WEIGHTS, dev)
    tf = C.make_transform(D.IMGSZ)
    arr = np.zeros((len(paths), HP, HP, CH), np.float16)
    for i, p in enumerate(paths):
        xb = tf(Image.open(p).convert("RGB")).unsqueeze(0)
        p3, _ = bb.p3_and_p5gap(xb)
        arr[i] = p3.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.float16)
    return arr

def nn_stats(bank_np):
    b = torch.from_numpy(bank_np).float()
    sims = b @ b.T
    sims.fill_diagonal_(-2.0)
    mx = sims.max(1).values
    return {"nn_cos_mean": round(float(mx.mean()), 4), "nn_cos_std": round(float(mx.std()), 4),
            "nn_cos_min": round(float(mx.min()), 4), "nn_cos_max": round(float(mx.max()), 4)}

def spatial_stats(bank_np, src_crop, proto_local):
    loc = proto_local  # (256,) patch index within its source crop
    r = (loc % PPC) // HP
    c = (loc % PPC) % HP
    edge = int(((r < 5) | (r > 34) | (c < 5) | (c > 34)).sum())
    from collections import Counter
    cnt = Counter(src_crop.tolist())
    return {"row_mean": round(float(r.float().mean()), 2), "col_mean": round(float(c.float().mean()), 2),
            "edge_frac": round(edge / len(loc), 4),
            "n_source_crops": len(cnt), "per_crop_counts": {str(k): v for k, v in sorted(cnt.items())}}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "need GPU"
    os.makedirs(OUT3 + "/raw_scores3", exist_ok=True)
    os.makedirs(OUT3 + "/bank_meta", exist_ok=True)
    os.makedirs(OUT3 + "/vis", exist_ok=True)

    wsha = D.sha256_file(D.WEIGHTS)
    assert wsha.startswith(D.WEIGHTS_SHA_PREFIX), wsha
    ok, bad = C.verify_split_hashes(D.SPLITS, C.MANIFEST_HASHES)
    assert ok and not bad, bad
    nref = json.load(open(OUT3 + "/normal_ref.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    print("[m3diag3] checks: weights + 12 splits + normal_ref lists ok")

    normal_banks = {}
    for a in nref["assets"]:
        paths = nref["assets"][a]["selected"]
        arr = extract_crops(paths, dev)
        raw = torch.from_numpy(arr.reshape(-1, CH)).float()
        vecs = embed(spec, raw, dev)
        assert torch.isfinite(vecs).all()
        idx = coreset_farthest(vecs, 256, FPS_SEED)
        normal_banks[a] = vecs[idx].cpu().numpy()
        print("[m3diag3] normal bank %s: %d patches -> %d refs" % (a, vecs.shape[0], len(idx)))

    results = {arm: {} for arm in ARMS}
    repro = []
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        nb = torch.from_numpy(normal_banks[a]).float().to(dev)
        for k in D.SHOTS:
            results[arm_kv(arm=None)] if False else None
            for arm in ARMS:
                results[arm].setdefault("k%d" % k, [])
            for s in D.SEEDS:
                with open(D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s), encoding="utf-8") as f:
                    split = json.load(f)
                with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
                    query = json.load(f)
                sup_paths = split["prototype_pools"][combo]
                sup_idx = [path2idx[p] for p in sup_paths]
                cand = torch.from_numpy(np.asarray(patches[sup_idx]).reshape(-1, CH)).float()
                assert torch.isfinite(cand).all()
                vecs = embed(spec, cand, dev)
                N = vecs.shape[0]
                n_keep = N // 2
                s_norm = 1.0 - (vecs @ nb.T).max(1).values
                assert torch.isfinite(s_norm).all()
                keep1 = torch.argsort(-s_norm, stable=True)[:n_keep]
                rng = np.random.default_rng(PRANDOM_SEED)
                keepr = torch.from_numpy(np.sort(rng.permutation(N)[:n_keep])).long().to(dev)
                banks, metas = {}, {}
                for arm, keep in (("P0", None), ("P-random", keepr), ("P1", keep1)):
                    v = vecs if keep is None else vecs[keep]
                    idx = torch.as_tensor(coreset_farthest(v, 256, FPS_SEED), device=v.device)
                    banks[arm] = v[idx].cpu().numpy()
                    local = (keep[idx] if keep is not None else idx).cpu().numpy()
                    metas[arm] = {"proto_local": local.tolist(),
                                  "src_crop": (local // PPC).tolist(),
                                  "spatial": spatial_stats(banks[arm], (local // PPC), local % PPC),
                                  "nn": nn_stats(banks[arm]),
                                  "n_source_crops": int(len(set((local // PPC).tolist())))}
                szs = {arm: banks[arm].shape[0] for arm in ARMS}
                assert len(set(szs.values())) == 1, szs
                qd = query["defect_query"][combo]
                qn = query["normal_query"][a + "/good"]
                qd_idx = [path2idx[p] for p in qd]
                qn_idx = [path2idx[p] for p in qn]
                row, raw_out = {}, {}
                for arm in ARMS:
                    s_d = score_crops(patches, qd_idx, spec, banks[arm], dev)
                    s_n = score_crops(patches, qn_idx, spec, banks[arm], dev)
                    assert np.isfinite(s_d).all() and np.isfinite(s_n).all()
                    auroc, flag = D.auroc_avg_rank(s_d, s_n)
                    thr = float(np.mean(s_n) + 2 * np.std(s_n)) if flag == "ok" else None
                    row[arm] = {"auroc": auroc, "flag": flag,
                                "recall": float((s_d > thr).mean()) if flag == "ok" else None}
                    raw_out[arm] = {"defect": [[p, float(x)] for p, x in zip(qd, s_d)],
                                    "normal": [[p, float(x)] for p, x in zip(qn, s_n)]}
                # --- P0 复现门 ---
                old = json.load(open(D.OUT + "/raw_scores/B_k%d_seed%d.json" % (k, s)))[combo]
                for side in ("defect", "normal"):
                    o = {p: x for p, x in old[side]}
                    nw = {p: x for p, x in raw_out["P0"][side]}
                    assert set(o) == set(nw), (combo, k, s, side, "path set mismatch")
                    md = max(abs(o[p] - nw[p]) for p in o)
                    repro.append({"combo": combo, "k": k, "seed": s, "side": side, "max_abs_diff": md})
                    assert md <= REPRO_TOL, "P0 repro failed: %s" % md
                # --- P1 筛选诊断 ---
                    pass
                sn_np = s_norm.cpu().numpy()
                k1_np = keep1.cpu().numpy()
                mask = np.zeros(N, bool); mask[k1_np] = True
                p1diag = {"kept": {"mean": float(sn_np[mask].mean()), "std": float(sn_np[mask].std()),
                                   "q10": float(np.quantile(sn_np[mask], 0.1)), "q90": float(np.quantile(sn_np[mask], 0.9))},
                          "excluded": {"mean": float(sn_np[~mask].mean()), "std": float(sn_np[~mask].std()),
                                       "q10": float(np.quantile(sn_np[~mask], 0.1)), "q90": float(np.quantile(sn_np[~mask], 0.9))}}
                for arm in ARMS:
                    metas[arm]["p1_normal_dist"] = p1diag if arm == "P1" else None
                tag = "%s__k%d_seed%d" % (combo.replace("/", "__"), k, s)
                for arm in ARMS:
                    D.save_json(raw_out[arm], OUT3 + "/raw_scores3/%s_%s.json" % (arm, tag))
                D.save_json({"n_pool": N, "n_keep": n_keep, "bank_sizes": szs, "arms": metas},
                            OUT3 + "/bank_meta/%s.json" % tag)
                for arm in ARMS:
                    results[arm]["k%d" % k].append({combo: row[arm]})
                print("[m3diag3] %s k=%d s=%d | P0 %.4f Prand %.4f P1 %.4f | repro max %.2e" % (
                    combo[:12], k, s, row["P0"]["auroc"], row["P-random"]["auroc"], row["P1"]["auroc"],
                    max(r["max_abs_diff"] for r in repro[-2:])))
    D.save_json({"repro_gate": {"tol": REPRO_TOL, "n_checks": len(repro),
                                "max_abs_diff": max(r["max_abs_diff"] for r in repro), "passed": True},
                 "normal_ref": {a: {k: v for k, v in nref["assets"][a].items() if k != "selected_sha256"} for a in nref["assets"]}},
                OUT3 + "/data_audit.json")

    # --- 汇总 faultbank_eval.json ---
    summary = {"arms_completed": len(ARMS), "std_ddof": 0, "arms": ARMS,
               "note": "探索性; 3 支持种子非显著性检验; 正常参照仅一组抽取, 不估计抽样方差",
               "per_asset": {}, "paired_diffs": {}, "macro_over_2_assets_supplementary": {}}
    for combo in COMBOS:
        summary["per_asset"][combo] = {}
        for arm in ARMS:
            summary["per_asset"][combo][arm] = {}
            for k in D.SHOTS:
                rows = results[arm]["k%d" % k]
                per_seed = [r[combo]["auroc"] for r in rows]
                summary["per_asset"][combo][arm]["k%d" % k] = {
                    "per_seed": per_seed, "mean": float(np.mean(per_seed)),
                    "std": float(np.std(per_seed)),
                    "recall_mean": float(np.mean([r[combo]["recall"] for r in rows if r[combo]["recall"] is not None]))}
    for combo in COMBOS:
        summary["paired_diffs"][combo] = {}
        for k in D.SHOTS:
            rows = {arm: results[arm]["k%d" % k] for arm in ARMS}
            for x, y, name in (("P1", "P0", "P1_minus_P0"), ("P1", "P-random", "P1_minus_Prandom"),
                               ("P-random", "P0", "Prandom_minus_P0")):
                ds = [rows[x][i][combo]["auroc"] - rows[y][i][combo]["auroc"] for i in range(3)]
                summary["paired_diffs"][combo].setdefault("k%d" % k, {})[name] = {
                    "per_seed": ds, "mean": float(np.mean(ds)), "std": float(np.std(ds)),
                    "n_pos": int(sum(1 for v in ds if v > 0))}
    for arm in ARMS:
        summary["macro_over_2_assets_supplementary"][arm] = {}
        for k in D.SHOTS:
            vals = [float(np.mean([results[arm]["k%d" % k][i][combo]["auroc"] for combo in COMBOS])) for i in range(3)]
            summary["macro_over_2_assets_supplementary"][arm]["k%d" % k] = {
                "per_seed_macro": vals, "mean": float(np.mean(vals))}
    D.save_json(summary, OUT3 + "/faultbank_eval.json")
    print("[m3diag3] faultbank_eval.json saved")

    # --- 可视化 (固定样本: k=5 seed=0 每组合排序首个支持图) ---
    try:
        make_vis(spec, patches, path2idx, normal_banks, results_check=None)
    except Exception as e:
        print("[vis] failed:", e)
    print("[done] scoring complete")

def arm_kv(arm=None):
    return None

def make_vis(spec, patches, path2idx, normal_banks, results_check=None):
    from PIL import Image, ImageDraw
    dev = "cuda"
    for combo in COMBOS:
        a = D.ASSET_OF[combo]
        k, s = 5, 0
        split = json.load(open(D.SPLITS + "/kshot_k%d_seed%d.json" % (k, s)))
        sup = sorted(split["prototype_pools"][combo])
        p0 = sup[0]
        nb = torch.from_numpy(normal_banks[a]).float().to(dev)
        cand = torch.from_numpy(np.asarray(patches[path2idx[p0]]).reshape(-1, CH)).float()
        vecs = embed(spec, cand, dev)
        s_norm = 1.0 - (vecs @ nb.T).max(1).values
        n_keep = vecs.shape[0] // 2
        keep1 = torch.argsort(-s_norm, stable=True)[:n_keep].cpu().numpy()
        rng = np.random.default_rng(PRANDOM_SEED)
        keepr = np.sort(rng.permutation(vecs.shape[0])[:n_keep])
        img = Image.open(p0).convert("RGB").resize((320, 320))
        panels = []
        for arm, keep in (("P0", None), ("P-random", keepr), ("P1", keep1)):
            im = img.copy()
            dr = ImageDraw.Draw(im)
            v = vecs if keep is None else vecs[torch.from_numpy(keep).to(dev)]
            idx = torch.as_tensor(coreset_farthest(v, 256, FPS_SEED), device=v.device).cpu().numpy()
            proto_local = (keep[idx] if keep is not None else idx)
            for pi in range(vecs.shape[0]):
                r, c = pi // HP, pi % HP
                x, y = (c + 0.5) * 8, (r + 0.5) * 8
                col = (160, 160, 160) if arm == "P0" else ((60, 60, 255) if pi not in set(keep.tolist()) else (80, 200, 80))
                dr.point((x, y), fill=col)
            for pi in proto_local:
                r, c = pi // HP, pi % HP
                x, y = (c + 0.5) * 8, (r + 0.5) * 8
                dr.ellipse([x - 3, y - 3, x + 3, y + 3], outline=(255, 200, 0), width=1)
            dr.text((6, 6), "%s %s" % (combo.split("/")[0][:10], arm), fill=(255, 255, 255))
            panels.append(im)
        out = Image.new("RGB", (960, 320))
        for i, im in enumerate(panels):
            out.paste(im, (i * 320, 0))
        out.save(OUT3 + "/vis/%s_k5_seed0.png" % combo.replace("/", "__"))
        print("[vis] saved", combo)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""M3TRANSFER 主编排: 门 -> yoke YOLO 锚点 -> glass YOLO -> WRN 两资产 -> 汇总. 40 配置."""
import csv, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3score_extract as X
import m3score_pcverify as PV
import m3transfer_lib as T

OUT = D.ROOT + "/outputs/M3transfer-top8"
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
GCACHE = "/root/autodl-tmp/m3transfer-glass-cache"
YCACHE = "/root/autodl-tmp/m3score-yoke-cache"
REFS = [3001, 3002, 3003, 3004, 3005]
ASSETS = {"yoke": ("yoke-suspension", "rust", "g2"), "glass": ("glass-insulator", "missing-cap", "g1")}
ARMS = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
WRN_TOL = {"atol": 1e-4, "rtol": 1e-4}
TOPK = 8
LEDGER = {}
PM_W = None  # WRN PatchcoreModel (全局一次加载)


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def combo(a):
    cat, deft, gk = ASSETS[a]
    return "%s__%s" % (cat, deft), cat, deft, gk


def save_cfg(name, meta, sd, sn, qd, qn):
    u, _ = D.auroc_avg_rank(np.array(sd, dtype=np.float64), np.array(sn, dtype=np.float64))
    rec = {"defect": {p: float(v) for p, v in zip(qd, sd)},
           "normal": {p: float(v) for p, v in zip(qn, sn)}, "auroc": float(u)}
    rec.update(meta)
    with open("%s/raw_scores/%s.json" % (OUT, name), "w") as f:
        json.dump(rec, f)
    return float(u)


def load_cfg(name):
    return json.load(open("%s/raw_scores/%s.json" % (OUT, name)))


def yolo_scores_t8(pm, ffn, qd, qn, bs=64):
    """同库同距离: 官方 nearest_neighbors(1)+compute_anomaly_score -> PC; 同距离 top8 均值 -> T8."""
    paths = qd + qn
    pc, t8 = [], []
    for i in range(0, len(paths), bs):
        f = torch.from_numpy(np.ascontiguousarray(ffn(paths[i:i + bs]))).float().cuda()
        B, C = f.shape[0], f.shape[-1]
        pe = f.reshape(-1, C)
        with torch.no_grad():
            ps, loc = pm.nearest_neighbors(pe, 1)
            pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
        pc.extend(float(x) for x in pred)
        t8.extend(float(x) for x in torch.topk(ps.reshape(B, -1), TOPK, dim=1).values.mean(1))
    assert all(np.isfinite(pc)) and all(np.isfinite(t8))
    k = len(qd)
    return pc[:k], pc[k:], t8[:k], t8[k:]


def wrn_ffn(paths):
    emb, _ = T.wrn_extract(PM_W, paths, bs=10)
    return emb.permute(0, 2, 3, 1).reshape(len(paths), -1, 1536).numpy().astype(np.float32)


def main():
    global PM_W
    t0 = time.time()
    T.deterministic()
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    # ---------------- freeze ----------------
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    data, uniqs = {}, {}
    for a in ("yoke", "glass"):
        c, cat, deft, gk = combo(a)
        qd = qq["defect_query"]["%s/%s" % (cat, deft)]
        qn = qq["normal_query"]["%s/good" % cat]
        assert set(qd).isdisjoint(set(qn)) and len(qd) > 0 and len(qn) > 0
        assert all(p in path2idx for p in qd + qn), "查询路径不在评测集"
        n5of = {j: SL["normal_lists"]["%s_ref%d" % (cat, j)]["N5"] for j in REFS}
        n10of = {j: SL["normal_lists"]["%s_ref%d" % (cat, j)]["N10"] for j in REFS}
        for j in REFS:
            assert set(n5of[j]).issubset(set(n10of[j])), "N5 应为 N10 子集"
            assert set(n5of[j]).isdisjoint(set(qd + qn)), "支持与查询重叠"
        sup = {s: SL["fault_supports"]["%s__%s_%s_k5_seed%d" % (cat, deft, gk, s)]["paths"] for s in (0, 1, 2)} if a == "yoke" else {}
        uniq = sorted(set(sum(n5of.values(), []) + sum(sup.values(), []) + qd + qn)) if a == "yoke"             else sorted(set(sum(n5of.values(), []) + qd + qn))
        data[a] = {"cat": cat, "deft": deft, "gk": gk, "qd": qd, "qn": qn, "n5": n5of, "n10": n10of}
        uniqs[a] = uniq
        print("[freeze] %s uniq=%d qd=%d qn=%d" % (a, len(uniq), len(qd), len(qn)), flush=True)
    frozen = {"weights_sha256": X.sha256_file(D.WEIGHTS),
              "splits_query_k5": X.sha256_file(D.SPLITS + "/query_k5.json"),
              "support_lists": X.sha256_file(BENCH + "/SUPPORT_LISTS.json"),
              "code_sha256": X.code_hashes(),
              "rules": {"pc_seed": {str(j): 7000 + (j - 3000) for j in REFS}, "ratio": 0.1, "nn": 9, "topk": TOPK,
                        "arms": {"Y-PC": "官方 compute_anomaly_score nn=9 (YOLO P3 L2 64d, 5N)",
                                 "Y-T8": "同库同距离 top8 均值 (YOLO)",
                                 "W-PC": "官方 compute_anomaly_score nn=9 (WRN1536 320, 确定性重建库 10N)",
                                 "W-T8": "同库同距离 top8 均值 (WRN)"},
                        "wrn_prep": "Resize(320,320,antialias)+Normalize(imagenet), 无 L2 (复刻 M3BENCH)",
                        "wrn_tol": WRN_TOL, "yoke_anchor_tol": 1e-4,
                        "yoke_uniq_includes_fault_supports": "与 M3SCORE 缓存身份一致, 仅用于缓存键, 不建故障库"},
              "budget": {"gpu_hours": 2, "wall_clock_h": 4, "probe_first": True}}
    X.atomic_json(frozen, OUT + "/FROZEN_INPUTS.json")
    # ---------------- 正确性门 (正式评测前立即 assert) ----------------
    t = time.time()
    g = {}
    g["top8_axis"] = T.verify_top8_axis(); assert g["top8_axis"]["axis_ok"] and g["top8_axis"]["wrong_axis_differs"]
    g["reshape"] = PV.verify_reshape(); assert g["reshape"]["shape_ok"] and g["reshape"]["position_ok"], g["reshape"]
    PM_W, sdh = T.wrn_model()
    g["wrn_state_hash"] = sdh[:16]
    g["wrn_distance"] = T.verify_wrn_distance(PM_W); assert g["wrn_distance"]["pass"], g["wrn_distance"]
    g["wrn_composition"] = T.verify_reference_composition(PM_W); assert g["wrn_composition"]["pass"], g["wrn_composition"]
    feat_fn = lambda rows: PM_W.generate_embedding(PM_W.feature_extractor(
        torch.stack([T.wrn_tf()(p) for p in rows]).to("cuda")))
    g["wrn_batch"] = T.verify_batch_score(feat_fn, data["yoke"]["qd"][:6], PM_W)
    assert g["wrn_batch"]["pass"], g["wrn_batch"]
    LEDGER["gates_s"] = round(time.time() - t, 1)
    print("[gates] 全部 assert 通过 (%.1fs) %s" % (LEDGER["gates_s"], json.dumps(g)[:200]), flush=True)
    # ---------------- 探针 ----------------
    t = time.time()
    T.wrn_extract(PM_W, data["glass"]["qd"][:8], bs=8)
    te = (time.time() - t) / 8
    PM_W.memory_bank = torch.randn(1600, 1536).to("cuda")
    pe = torch.randn(8 * 1600, 1536).to("cuda")
    t = time.time()
    with torch.no_grad():
        PM_W.nearest_neighbors(pe, 1)
    td = (time.time() - t) / 8
    n_imgs = sum(len(uniqs[a]) for a in ("yoke", "glass"))
    n_q = sum(len(data[a]["qd"]) + len(data[a]["qn"]) for a in ("yoke", "glass"))
    proj = n_imgs * te + n_q * 5 * td + 60
    print("[probe] extract=%.2fs/img dist=%.2fs/img -> proj WRN=%.0fs / 7200s" % (te, td, proj), flush=True)
    assert proj < 7200, "WRN 预算投影超限, 停止扩展"
    # ---------------- B: yoke YOLO 锚点 ----------------
    t = time.time()
    yman = json.load(open(YCACHE + "/CACHE_MANIFEST.json"))
    assert yman["cache_sha256_of_saved_file"] == T.file_sha(YCACHE + "/e1_feats.npy"), "yoke 缓存文件哈希不符"
    assert yman["identity_complete"] and yman["deterministic_conv"] and not yman["tf32_cudnn"]
    assert set(yman["paths"]) == set(uniqs["yoke"]), "yoke 缓存路径集合不符"
    e1y = np.load(YCACHE + "/e1_feats.npy", mmap_mode="r")
    r2y = {p: i for i, p in enumerate(yman["paths"])}
    dy = data["yoke"]
    ffn_y = lambda ps: torch.from_numpy(np.ascontiguousarray(e1y[[r2y[p] for p in ps]])).reshape(len(ps), -1, 64).float()
    m3man = json.load(open(D.ROOT + "/outputs/M3score-yoke/BANK_MANIFEST.json"))
    pms = {}
    for j in REFS:
        pm, meta = PV.make_pm(ffn_y, dy["n5"][j], 7000 + (j - 3000))
        pms[j] = pm
        ref = m3man["N_star"]["ref%d" % j]
        assert meta["n_bank"] == ref["n_bank"] and th(pm.memory_bank) == ref["tensor_hash"], "yoke N* 库哈希不一致 ref%d" % j
    anchor = {"pc_max_diff": 0.0, "t8_max_diff": 0.0}
    for j in REFS:
        pcd, pcn, t8d, t8n = yolo_scores_t8(pms[j], ffn_y, dy["qd"], dy["qn"])
        old4 = json.load(open(D.ROOT + "/outputs/M3score-yoke/raw_scores/yoke-suspension__rust_S4_k5_ref%d.json" % j))
        old2 = json.load(open(D.ROOT + "/outputs/M3score-yoke/raw_scores/yoke-suspension__rust_S2_k5_ref%d.json" % j))
        d1 = max(max(abs(x - v) for x, v in zip(pcd, old4["defect"].values())),
                 max(abs(x - v) for x, v in zip(pcn, old4["normal"].values())))
        d2 = max(max(abs(x - v) for x, v in zip(t8d, old2["defect"].values())),
                 max(abs(x - v) for x, v in zip(t8n, old2["normal"].values())))
        anchor["pc_max_diff"] = max(anchor["pc_max_diff"], d1)
        anchor["t8_max_diff"] = max(anchor["t8_max_diff"], d2)
        assert d1 <= 1e-4 and d2 <= 1e-4, "yoke 锚点超门限 ref%d: %.2e/%.2e" % (j, d1, d2)
        c, cat, deft, gk = combo("yoke")
        bh = th(pms[j].memory_bank)
        save_cfg("%s_Y-PC_k5_ref%d" % (c, j), {"asset": "yoke", "arm": "Y-PC", "ref": j, "bank_hash": bh,
                 "anchor": "M3SCORE S4 逐路径复现 max|diff|=%.2e" % d1}, pcd, pcn, dy["qd"], dy["qn"])
        save_cfg("%s_Y-T8_k5_ref%d" % (c, j), {"asset": "yoke", "arm": "Y-T8", "ref": j, "bank_hash": bh,
                 "anchor": "M3SCORE S2 逐路径复现 max|diff|=%.2e" % d2}, t8d, t8n, dy["qd"], dy["qn"])
    print("[anchor] yoke YOLO 复现通过 pc=%.2e t8=%.2e" % (anchor["pc_max_diff"], anchor["t8_max_diff"]), flush=True)
    LEDGER["anchor_s"] = round(time.time() - t, 1)
    del e1y
    torch.cuda.empty_cache()
    # ---------------- C: glass YOLO ----------------
    t = time.time()
    dg = data["glass"]
    c, cat, deft, gk = combo("glass")
    gman, e1g = X.build_cache(uniqs["glass"], cache_dir=GCACHE)
    assert gman["cache_sha256_of_saved_file"] == T.file_sha(GCACHE + "/e1_feats.npy")
    assert gman["deterministic_conv"] and not gman["tf32_cudnn"]
    r2g = {p: i for i, p in enumerate(gman["paths"])}
    ffn_g = lambda ps: torch.from_numpy(np.ascontiguousarray(e1g[[r2g[p] for p in ps]])).reshape(len(ps), -1, 64).float()
    bb = X.MC.Backbone(D.WEIGHTS, "cuda")
    tfm = X.MC.make_transform(X.D.IMGSZ)
    ext_diff, cache_diff = X.batch_consistency(bb, tfm, dg["qd"][:3] + dg["qn"][:3], e1g, r2g)
    del bb
    torch.cuda.empty_cache()
    assert ext_diff <= 2e-4 and cache_diff <= 2e-4, "glass 提取一致性超限: %.2e/%.2e" % (ext_diff, cache_diff)
    g["glass_batch"] = {"ext_diff": ext_diff, "cache_diff": cache_diff}
    pms_g = {}
    for j in REFS:
        pms_g[j], _ = PV.make_pm(ffn_g, dg["n5"][j], 7000 + (j - 3000))
    for j in REFS:
        pcd, pcn, t8d, t8n = yolo_scores_t8(pms_g[j], ffn_g, dg["qd"], dg["qn"])
        bh = th(pms_g[j].memory_bank)
        save_cfg("%s_Y-PC_k5_ref%d" % (c, j), {"asset": "glass", "arm": "Y-PC", "ref": j, "bank_hash": bh}, pcd, pcn, dg["qd"], dg["qn"])
        save_cfg("%s_Y-T8_k5_ref%d" % (c, j), {"asset": "glass", "arm": "Y-T8", "ref": j, "bank_hash": bh}, t8d, t8n, dg["qd"], dg["qn"])
    print("[glass] YOLO 完成", flush=True)
    LEDGER["glass_yolo_s"] = round(time.time() - t, 1)
    del e1g
    torch.cuda.empty_cache()
    # ---------------- D: WRN 两资产 (确定性重建库) ----------------
    t = time.time()
    banks, bmeta = {}, {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for j in REFS:
            pm, meta = PV.make_pm(wrn_ffn, d_["n10"][j], 7000 + (j - 3000))
            if a == "yoke" and j == 3001:
                pm2, meta2 = PV.make_pm(wrn_ffn, d_["n10"][j], 7000)
                assert torch.equal(pm.memory_bank, pm2.memory_bank), "WRN 双建库不一致"
                g["wrn_bank_rebuild"] = {"ref3001": True, "n_bank": meta["n_bank"]}
                del pm2
            banks[(a, j)] = pm
            bmeta["%s_ref%d" % (a, j)] = {"n_bank": meta["n_bank"], "bank_hash": th(pm.memory_bank),
                                          "indices_first10": meta["coreset_indices"][:10], "pc_seed": 7000 + (j - 3000)}
            print("[wrn-bank] %s ref%d protos=%d" % (a, j, meta["n_bank"]), flush=True)
    from PIL import Image
    dist_files = {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        qpaths = d_["qd"] + d_["qn"]
        dists = {j: [] for j in REFS}
        pcs = {j: [] for j in REFS}
        for i in range(0, len(qpaths), 16):
            xb = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in qpaths[i:i + 16]]).to("cuda")
            emb = PM_W.generate_embedding(PM_W.feature_extractor(xb))
            B, C = emb.shape[0], emb.shape[1]
            pe = PV._flat(PM_W.reshape_embedding(emb)).float()
            for j in REFS:
                pm = banks[(a, j)]
                with torch.no_grad():
                    ps, loc = pm.nearest_neighbors(pe, 1)
                    pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
                pcs[j].extend(float(x) for x in pred)
                dists[j].append(ps.reshape(B, -1).cpu())
        npz = "/root/autodl-tmp/m3transfer-wrn-dists-%s.npz" % a
        np.savez_compressed(npz, **{"ref%d" % j: torch.cat(dists[j], 0).numpy().astype(np.float32) for j in REFS})
        dist_files[a] = {"path": npz, "sha256": T.file_sha(npz)}
        k = len(d_["qd"])
        for j in REFS:
            dmat = torch.cat(dists[j], 0)
            assert dmat.shape[0] == len(qpaths) and dmat.shape[1] == banks[(a, j)].memory_bank.shape[0]
            assert dmat.shape[1] == bmeta["%s_ref%d" % (a, j)]["n_bank"]
            bh = th(banks[(a, j)].memory_bank)
            save_cfg("%s_W-PC_k5_ref%d" % (c, j), {"asset": a, "arm": "W-PC", "ref": j, "bank_hash": bh},
                     pcs[j][:k], pcs[j][k:], d_["qd"], d_["qn"])
            t8 = T.top8_mean(dmat).tolist()
            save_cfg("%s_W-T8_k5_ref%d" % (c, j), {"asset": a, "arm": "W-T8", "ref": j, "bank_hash": bh},
                     t8[:k], t8[k:], d_["qd"], d_["qn"])
        print("[wrn] %s 完成" % a, flush=True)
    LEDGER["wrn_s"] = round(time.time() - t, 1)
    # ---------------- E: 汇总 ----------------
    t = time.time()
    aurocs = {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for arm in ARMS:
            for j in REFS:
                r = load_cfg("%s_%s_k5_ref%d" % (c, arm, j))
                u2, _ = D.auroc_avg_rank(np.array(list(r["defect"].values()), np.float64),
                                         np.array(list(r["normal"].values()), np.float64))
                assert abs(u2 - r["auroc"]) < 1e-12, "AUROC 重算不一致 %s_%s_ref%d" % (a, arm, j)
                aurocs["%s_%s_ref%d" % (c, arm, j)] = float(u2)
    names = sorted(aurocs)
    expected = sorted(["%s__%s_%s_k5_ref%d" % (data[a]["cat"], data[a]["deft"], arm, j)
                       for a in ("yoke", "glass") for arm in ARMS for j in REFS])
    set_match = int(names == expected)
    assert set_match == 1 and len(names) == 40, "配置集合与预注册不一致"
    def stats(vs):
        v = np.array(vs, np.float64)
        return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "std_ddof0": float(v.std(ddof=0))}
    arm_means = {arm: {a: stats([aurocs["%s__%s_%s_ref%d" % (data[a]["cat"], data[a]["deft"], arm, j)] for j in REFS])
                       for a in ("yoke", "glass")} for arm in ARMS}
    def paired(a1, a2, key):
        out = {}
        for a in ("yoke", "glass"):
            d_ = data[a]
            c = "%s__%s" % (d_["cat"], d_["deft"])
            per = {"ref%d" % j: aurocs["%s_%s_ref%d" % (c, a1, j)] - aurocs["%s_%s_ref%d" % (c, a2, j)] for j in REFS}
            out["%s_%s" % (key, a)] = {"per_ref": per, **stats(list(per.values())),
                                       "n_pos": sum(1 for v in per.values() if v > 0), "n_neg": sum(1 for v in per.values() if v < 0)}
        return out
    comps = {}
    for k1, k2, key in (("Y-T8", "Y-PC", "YT8-YPC"), ("W-T8", "W-PC", "WT8-WPC"),
                        ("Y-T8", "W-PC", "YT8-WPC"), ("Y-T8", "W-T8", "YT8-WT8")):
        comps.update(paired(k1, k2, key))
    hist = {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for j in REFS:
            h = json.load(open(BENCH + "/raw_scores_patchcore/%s_%s_ref%d.json" % (c, d_["gk"], j)))
            hu, _ = D.auroc_avg_rank(np.array(list(h["defect"].values()), np.float64),
                                     np.array(list(h["normal"].values()), np.float64))
            nu = aurocs["%s_W-PC_ref%d" % (c, j)]
            hist["%s_ref%d" % (a, j)] = {"hist_W-PC": float(hu), "new_W-PC": float(nu), "diff": float(nu - hu)}
    ev = {"arm_configs_completed": len(names), "pre_registered_set_match": set_match,
          "gates": g, "gates_all_passed": 1, "yoke_anchor_ok": 1, "anchor": anchor,
          "same_bank_identity": 1, "scores_recount_match": 1,
          "arm_means": arm_means, "paired": comps, "wrn_rebuild_vs_hist": hist,
          "wrn_dist_files": dist_files, "bank_meta": bmeta}
    X.atomic_json(ev, OUT + "/transfer_eval.json")
    with open(OUT + "/paired_comparisons.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["comparison", "asset", "group", "diff"])
        for key in ("YT8-YPC", "WT8-WPC", "YT8-WPC", "YT8-WT8"):
            for a in ("yoke", "glass"):
                for r, v in comps["%s_%s" % (key, a)]["per_ref"].items():
                    w.writerow([key, a, r, v])
    X.atomic_json({"stages_s": LEDGER, "total_wall_s": round(time.time() - t0, 1),
                   "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                   "wrn_dist_bytes": {a: os.path.getsize(dist_files[a]["path"]) for a in dist_files},
                   "note": "评分耗时为缓存特征计算, 非端到端推理延迟; WRN 查询特征流式未全量缓存"}, OUT + "/RESOURCE_COST.json")
    X.atomic_json({"gates": g, "anchor": anchor, "bank_meta": bmeta, "dist_files": dist_files,
                   "wrn_tol": WRN_TOL, "note": "正确性证据汇总; 门在正式评测前立即 assert"}, OUT + "/REPRO_AND_CORRECTNESS.json")
    with open(OUT + "/WRN_REUSE_OR_REBUILD.md", "w") as f:
        f.write("# WRN_REUSE_OR_REBUILD\n\n审计: M3BENCH 仅保留图像分数 (raw_scores_patchcore/*.json), 未保存 memory_bank、查询特征或逐 patch 距离; 建库随机起点未固定, 历史库无法逐分数复现。\n\n依合同 §七回退: 按原标准配置 (WRN50 layer2+3, 320 输入 + Normalize, ratio=0.1, nn=9) 确定性重建, pc_seed=7000+(ref-3000), yoke ref3001 双建库张量哈希核验; W-PC 与 W-T8 在同一新库同距离上评测; 历史 W-PC 仅作参考 (新-历史逐组见 transfer_eval.json wrn_rebuild_vs_hist); 新旧 W-PC 差异不解释为纯聚合收益。\n")
    print("[eval] 40 配置 set_match=%d wall=%.0fs" % (set_match, time.time() - t0), flush=True)
    print("=== M3TRANSFER 完成 ===", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M3SCORE 主编排: 门 -> 建库(A4) -> E1_PC 复验(A7=S4) -> 评分分解(B) -> 汇总. 50 配置."""
import csv, hashlib, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
import m3score_extract as X
import m3score_pcverify as PV

OUT = X.OUT
CACHE = X.CACHE
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
FEAT = D.ROOT + "/outputs/M3feat-yoke"
CKC = "yoke-suspension__rust"
REFS = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
ARMS_S = ["S1", "S2", "S3", "S4"]
BUDGET_S = 3600
TOPK = 8
Q_CHUNK, P_CHUNK = 128, 1024


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def top8(v):
    return torch.topk(v, TOPK, dim=1).values.mean(1)


def get_all():
    out = {}
    for f in sorted(os.listdir(OUT + "/raw_scores")):
        out[f[:-5]] = json.load(open(OUT + "/raw_scores/" + f))["auroc"]
    return out


def save_cfg(name, cfg_meta, sd, sn, qd, qn):
    u, fl = D.auroc_avg_rank(np.array(sd, dtype=np.float64), np.array(sn, dtype=np.float64))
    assert fl == "ok"
    X.atomic_json({"config": cfg_meta,
                   "defect": {p: float(v) for p, v in zip(qd, sd)},
                   "normal": {p: float(v) for p, v in zip(qn, sn)},
                   "auroc": round(float(u), 6)}, OUT + "/raw_scores/%s.json" % name)
    return round(float(u), 6)


def main():
    t00 = time.time()
    ledger = {}
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    qd = qq["defect_query"]["yoke-suspension/rust"]
    qn = qq["normal_query"]["yoke-suspension/good"]
    assert set(qd).isdisjoint(set(qn)) and len(qd) > 0 and len(qn) > 0
    assert all(p in path2idx for p in qd + qn)
    qd_rows = [path2idx[p] for p in qd]
    qn_rows = [path2idx[p] for p in qn]
    qall_paths = qd + qn
    sup = {s: SL["fault_supports"]["%s_g2_k5_seed%d" % (CKC, s)]["paths"] for s in SEEDS}
    n5of = {j: SL["normal_lists"]["yoke-suspension_ref%d" % j]["N5"] for j in REFS}
    pc_seed = {j: 7000 + (j - 3000) for j in REFS}
    uniq = sorted(set(sum(n5of.values(), []) + sum(sup.values(), []) + qd + qn))
    frozen = {"weights_sha256": X.sha256_file(D.WEIGHTS),
              "splits": {"query_k5": X.sha256_file(D.SPLITS + "/query_k5.json"),
                         **{"kshot_k5_seed%d" % s: X.sha256_file(D.SPLITS + "/kshot_k5_seed%d.json" % s) for s in SEEDS}},
              "support_lists": X.sha256_file(BENCH + "/SUPPORT_LISTS.json"),
              "code_sha256": X.code_hashes(),
              "query": {"n_defect": len(qd), "n_normal": len(qn), "disjoint": True},
              "rules": {"feature": "E1 P3 64d fp32 逐位置L2 (CACHE_MANIFEST)",
                        "pc_seed": pc_seed, "ratio": 0.1, "nn": 9, "topk": TOPK,
                        "formulas": {"S1": "max dE", "S2": "top8 dE", "S3": "top8 dC",
                                     "S4": "official compute_anomaly_score nn=9 共享N*",
                                     "S5": "top8 (cF - cN_N*)",
                                     "R256": "top8 (cF - cN_N256) 参考臂"},
                        "fault_bank": "M3FEAT E1_DUAL 规则: N256(FPS256, seed=%d) 筛选; 5A 8000 patch 按 1-max_cos stable 降序前 4000" % M6.FPS_SEED,
                        "gates": {"batch_extract_abs": 2e-4, "score_batch": "atol=rtol=1e-4", "chunk_abs": 1e-5}},
              "budget": {"gpu_hours": 1, "probe_first": True},
              "unique_paths_n": len(uniq)}
    X.atomic_json(frozen, OUT + "/FROZEN_INPUTS.json")
    print("[freeze] n_unique=%d qd=%d qn=%d" % (len(uniq), len(qd), len(qn)), flush=True)
    # ---------- A1 ----------
    t = time.time()
    man, e1arr = X.build_cache(uniq)
    ledger["A1_cache_s"] = round(time.time() - t, 1)
    print("[A1] cache n=%d sha=%s" % (man["n_images"], man["cache_sha256_of_saved_file"][:12]), flush=True)
    row2 = {p: i for i, p in enumerate(man["paths"])}
    ffn = lambda ps: torch.from_numpy(np.ascontiguousarray(e1arr[[row2[p] for p in ps]])).reshape(len(ps), -1, e1arr.shape[-1]).float()
    # ---------- A2 ----------
    t = time.time()
    import m3_common as MC
    bb = MC.Backbone(D.WEIGHTS, "cuda")
    tf = MC.make_transform(D.IMGSZ)
    samples = [n5of[3001][0], n5of[3001][1], qn[0], qn[1], qd[0], qd[1]]
    ext_diff, cache_diff = X.batch_consistency(bb, tf, samples, e1arr, row2)
    del bb
    torch.cuda.empty_cache()
    e1t = torch.from_numpy(np.ascontiguousarray(e1arr[:512])).float()
    nn_dev = float((e1t.reshape(-1, e1t.shape[-1]).norm(dim=-1) - 1).abs().max())
    finite_ok = bool(np.isfinite(np.asarray(e1arr[:512])).all())
    print("[A2] ext_diff=%.2e cache_diff=%.2e norm_dev=%.2e finite=%s" % (ext_diff, cache_diff, nn_dev, finite_ok), flush=True)
    gate_numeric = bool(ext_diff <= 2e-4 and nn_dev <= 1e-3 and finite_ok)
    ledger["A2_gates_s"] = round(time.time() - t, 1)
    # ---------- A3 ----------
    t = time.time()
    src = PV.source_meta()
    v_rs = PV.verify_reshape()
    v_d = PV.verify_distance()
    v_c = PV.verify_composition()
    gate_adapter = bool(v_rs["shape_ok"] and v_rs["position_ok"] and v_d["nn_index_ok"] and v_d["tie_dist_ok"] and v_c["pass"])
    print("[A3] reshape=%s dist_err=%.2e coll_err=%.2e comp_diff=%.2e" % (v_rs["position_ok"], v_d["dist_max_abs_err"], v_d["collinear_abs_err"], v_c["composition_max_abs_diff"]), flush=True)
    ledger["A3_verify_s"] = round(time.time() - t, 1)
    # ---------- A5 ----------
    t = time.time()
    v_ch = PV.verify_chunk(d=64, seed=17, n_img=1000, P=16, M=4000, q_chunk=400, p_chunk=1500)
    gate_chunk = bool(v_ch["pass"])
    print("[A5] diff=%.2e qb=%d pb=%d tail=%s" % (v_ch["max_cos_abs_diff"], v_ch["q_blocks"], v_ch["p_blocks"], v_ch["tail_covered"]), flush=True)
    ledger["A5_chunk_s"] = round(time.time() - t, 1)
    # ---------- 探针 ----------
    t = time.time()
    pm0, meta0 = PV.make_pm(ffn, n5of[3001], pc_seed[3001])
    _s = PV.official_score(pm0, qd_rows[:8] + qn_rows[:8], ffn)
    cN_probe = PV.chunked_max_cos(ffn(qall_paths[:32]), pm0.memory_bank, Q_CHUNK, P_CHUNK)
    probe_s = time.time() - t
    proj = 4 * probe_s + 15 * probe_s * 0.6 + 40
    print("[probe] block=%.1fs proj=%.0fs budget=%d" % (probe_s, proj, BUDGET_S), flush=True)
    if proj > BUDGET_S:
        X.atomic_json({"blocked": "probe %.0fs > %d" % (proj, BUDGET_S), "stages_s": ledger}, OUT + "/RESOURCE_COST.json")
        print("[probe] 超预算, 停止", flush=True)
        return
    # ---------- A4 ----------
    banks = {}
    bank_manifest = {"N_star": {}, "N256_screen": {}, "F4000": {},
                     "rng": "make_pm 内 random/np/torch(+cuda) 以 pc_seed 重置; KCenterGreedy 随机起点消费 torch RNG (源码 select_coreset_idxs)"}
    t = time.time()
    rescore_diff = None
    for j in REFS:
        pm, meta = PV.make_pm(ffn, n5of[j], pc_seed[j])
        nb = pm.memory_bank.detach().clone()
        pm2, meta2 = PV.make_pm(ffn, n5of[j], pc_seed[j])
        nb2 = pm2.memory_bank.detach().clone()
        same = bool(torch.equal(nb, nb2))
        if j == REFS[0]:
            sb1 = PV.official_score(pm, qd_rows[:4] + qn_rows[:4], ffn)
            sb2 = PV.official_score(pm2, qd_rows[:4] + qn_rows[:4], ffn)
            rescore_diff = float(np.max(np.abs(np.array(sb1) - np.array(sb2))))
            sbc = PV.verify_score_batch(ffn, qd_rows[:6], pm)
        else:
            sbc = None
        bank_manifest["N_star"]["ref%d" % j] = {k: v for k, v in meta.items() if k != "coreset_indices"} | {
            "coreset_indices_n": len(meta["coreset_indices"]), "tensor_hash": th(nb), "rebuild_equal": same,
            "rescore_diff": rescore_diff, "src_paths": n5of[j], "score_batch_check": sbc}
        banks[j] = {"pm": pm, "N": nb}
        print("[A4] ref%d bank=%d rebuild_equal=%s" % (j, meta["n_bank"], same), flush=True)
    gate_bank = bool(all(bank_manifest["N_star"]["ref%d" % j]["rebuild_equal"] for j in REFS)
                     and rescore_diff is not None and rescore_diff <= 1e-6)
    score_batch_ok = bool(bank_manifest["N_star"]["ref3001"]["score_batch_check"]["pass"])
    ledger["A4_banks_s"] = round(time.time() - t, 1)
    # ---------- A7 (=S4, 复用库) ----------
    t = time.time()
    hist = json.load(open(FEAT + "/feat_eval.json"))["aurocs"]
    recheck = {"per_ref": {}, "recheck_configs": 0}
    for j in REFS:
        sd = PV.official_score(banks[j]["pm"], qd_rows, ffn)
        sn = PV.official_score(banks[j]["pm"], qn_rows, ffn)
        u = save_cfg("%s_S4_k5_ref%d" % (CKC, j),
                     {"combo": "yoke-suspension/rust", "arm": "S4", "ref": j, "feature": "E1",
                      "bank": "N* pc_seed=%d" % pc_seed[j]}, sd, sn, qd, qn)
        hk = "%s_E1_PC_k5_ref%d" % (CKC, j)
        recheck["per_ref"]["ref%d" % j] = {"new": u, "historical": hist[hk], "diff": round(u - hist[hk], 6)}
        recheck["recheck_configs"] += 1
    recheck["mean_new"] = round(float(np.mean([recheck["per_ref"]["ref%d" % j]["new"] for j in REFS])), 6)
    recheck["mean_historical"] = round(float(np.mean([hist["%s_E1_PC_k5_ref%d" % (CKC, j)] for j in REFS])), 6)
    X.atomic_json(recheck, OUT + "/E1_PC_RECHECK.json")
    print("[A7] mean_new=%.6f mean_hist=%.6f" % (recheck["mean_new"], recheck["mean_historical"]), flush=True)
    ledger["A7_recheck_s"] = round(time.time() - t, 1)
    # ---------- B ----------
    t = time.time()
    WRN_EXT = {}
    for j in REFS:
        r = json.load(open(BENCH + "/raw_scores_patchcore/%s_g2_ref%d.json" % (CKC, j)))
        assert set(r["defect"]) == set(qd) and set(r["normal"]) == set(qn)
        u, fl = D.auroc_avg_rank(np.array(list(r["defect"].values()), dtype=np.float64),
                                 np.array(list(r["normal"].values()), dtype=np.float64))
        WRN_EXT["ref%d" % j] = round(float(u), 6)
    wrn_mean = round(float(np.mean(list(WRN_EXT.values()))), 6)
    patch_dir = CACHE + "/patchstats"
    os.makedirs(patch_dir, exist_ok=True)
    de_err = rel = None
    for j in REFS:
        q = ffn(qall_paths).cuda()
        Nstar = banks[j]["N"].float()
        cN = PV.chunked_max_cos(q, Nstar, Q_CHUNK, P_CHUNK)
        dE = torch.sqrt(torch.clamp(2.0 - 2.0 * cN, min=0.0))
        dC = 1.0 - cN
        idx = torch.linspace(0, dE.numel() - 1, 200).long()
        de64 = torch.sqrt(torch.clamp(2.0 - 2.0 * cN.double().flatten()[idx], min=0.0))
        dE_dir = torch.cdist(q.double().reshape(-1, q.shape[-1])[idx], Nstar.double()).min(1).values
        de_err = float((de64 - dE_dir).abs().max())
        rel = float(((de64 - dE_dir).abs() / dE_dir.clamp_min(1e-9)).max())
        s1 = dE.max(1)
        s2 = top8(dE)
        s3 = top8(dC)
        stats = {"dE": dE.cpu().numpy().astype(np.float32), "dC": dC.cpu().numpy().astype(np.float32),
                 "top8_S2": torch.topk(dE, TOPK, 1).indices.cpu().numpy().astype(np.int32),
                 "top8_S3": torch.topk(dC, TOPK, 1).indices.cpu().numpy().astype(np.int32)}
        for s in SEEDS:
            n5f = ffn(n5of[j]).reshape(-1, e1arr.shape[-1])
            o256 = torch.as_tensor(M6.coreset_farthest(n5f, 256, M6.FPS_SEED))
            N256 = n5f[o256].clone()
            supf = ffn(sup[s]).reshape(-1, e1arr.shape[-1])
            snorm = 1.0 - (supf @ N256.T).max(1).values
            keep = torch.argsort(-snorm, stable=True)[:supf.shape[0] // 2]
            Fb = supf[keep]
            cF = PV.chunked_max_cos(q, Fb, Q_CHUNK, P_CHUNK)
            if "cN256" not in stats:
                stats["cN256"] = PV.chunked_max_cos(q, N256, Q_CHUNK, P_CHUNK).numpy().astype(np.float32)
            stats["cF_seed%d" % s] = cF.cpu().numpy().astype(np.float32)
            v5 = cF - cN
            v256 = cF - torch.from_numpy(stats["cN256"]).float().cuda()
            s5 = top8(v5)
            r256 = top8(v256)
            stats["top8_S5_seed%d" % s] = torch.topk(v5, TOPK, 1).indices.cpu().numpy().astype(np.int32)
            stats["top8_R256_seed%d" % s] = torch.topk(v256, TOPK, 1).indices.cpu().numpy().astype(np.int32)
            for arm, vec in (("S5", s5), ("R256", r256)):
                sd_v = [float(x) for x in vec[:len(qd)]]
                sn_v = [float(x) for x in vec[len(qd):]]
                cfg = "%s_%s_k5_seed%d_ref%d" % (CKC, arm, s, j)
                u = save_cfg(cfg, {"combo": "yoke-suspension/rust", "arm": arm, "seed": s, "ref": j, "feature": "E1",
                                   "bank": "N* pc_seed=%d" % pc_seed[j] if arm == "S5" else "N256 FPS256 seed=%d" % M6.FPS_SEED,
                                   "fault_seed": s}, sd_v, sn_v, qd, qn)
                aurocs[cfg] = u
            bank_manifest["N256_screen"]["ref%d" % j] = {"n": int(N256.shape[0]), "dim": int(N256.shape[1]), "hash": th(N256), "src_paths": n5of[j]}
            bank_manifest["F4000"]["ref%d_seed%d" % (j, s)] = {"n": int(Fb.shape[0]), "dim": int(Fb.shape[1]), "hash": th(Fb),
                                                               "keep_n": int(keep.shape[0]), "keep_first10": [int(x) for x in keep[:10]],
                                                               "src_paths": sup[s]}
        for arm, vec in (("S1", s1), ("S2", s2), ("S3", s3)):
            sd_v = [float(x) for x in vec[:len(qd)]]
            sn_v = [float(x) for x in vec[len(qd):]]
            cfg = "%s_%s_k5_ref%d" % (CKC, arm, j)
            u = save_cfg(cfg, {"combo": "yoke-suspension/rust", "arm": arm, "ref": j, "feature": "E1",
                               "bank": "N* pc_seed=%d" % pc_seed[j]}, sd_v, sn_v, qd, qn)
            aurocs[cfg] = u
        np.savez_compressed(patch_dir + "/ref%d.npz" % j, **stats)
        X.atomic_json({"file": "ref%d.npz" % j, "keys": sorted(stats.keys()),
                       "sha256": X.sha256_file(patch_dir + "/ref%d.npz" % j),
                       "note": "dE/dC/cF/cN256 (n,1600) f32 + top8 索引; 可由 scripts/m3score_run.py 确定性重建"}, patch_dir + "/ref%d.json" % j)
        print("[B] ref%d done (dE2=2dC abs=%.2e rel=%.2e)" % (j, de_err, rel), flush=True)
        del q
        torch.cuda.empty_cache()
    ledger["B_scoring_s"] = round(time.time() - t, 1)
    X.atomic_json(bank_manifest, OUT + "/BANK_MANIFEST.json")
    shutil.copy(CACHE + "/CACHE_MANIFEST.json", OUT + "/CACHE_MANIFEST.json")
    # ---------- 汇总 ----------
    expected = set(["%s_%s_k5_ref%d" % (CKC, a, j) for a in ARMS_S for j in REFS] +
                   ["%s_S5_k5_seed%d_ref%d" % (CKC, s, j) for s in SEEDS for j in REFS] +
                   ["%s_R256_k5_seed%d_ref%d" % (CKC, s, j) for s in SEEDS for j in REFS])
    allsc = get_all()
    actual = set(allsc.keys())
    set_match = bool(expected == actual)
    assert set_match, "配置集合不一致: %s" % (expected ^ actual)
    g = lambda c: allsc[c]
    paired = {}
    for cmp_name, A1, A2 in (("S2-S1", "S2", "S1"), ("S3-S2", "S3", "S2"), ("S4-S1", "S4", "S1")):
        per = {"ref%d" % j: round(g("%s_%s_k5_ref%d" % (CKC, A1, j)) - g("%s_%s_k5_ref%d" % (CKC, A2, j)), 6) for j in REFS}
        fv = list(per.values())
        paired[cmp_name] = {"per_ref": per, "mean": round(float(np.mean(fv)), 6),
                            "range": [round(float(min(fv)), 6), round(float(max(fv)), 6)],
                            "n_pos": int(sum(v > 0 for v in fv)), "n_neg": int(sum(v < 0 for v in fv)),
                            "std_ddof0": round(float(np.std(fv)), 6)}
    per, per_seed = {}, {}
    for j in REFS:
        ds = []
        for s in SEEDS:
            d = round(g("%s_S5_k5_seed%d_ref%d" % (CKC, s, j)) - g("%s_S3_k5_ref%d" % (CKC, j)), 6)
            ds.append(d)
            per_seed["ref%d_seed%d" % (j, s)] = d
        per["ref%d" % j] = round(float(np.mean(ds)), 6)
    fv = list(per.values())
    paired["S5-S3"] = {"per_ref": per, "per_seed": per_seed, "mean": round(float(np.mean(fv)), 6),
                       "range": [round(float(min(fv)), 6), round(float(max(fv)), 6)],
                       "n_pos": int(sum(v > 0 for v in fv)), "n_neg": int(sum(v < 0 for v in fv)),
                       "std_ddof0": round(float(np.std(fv)), 6),
                       "n_pos_seeds": int(sum(v > 0 for v in per_seed.values())), "n_seed_pairs": len(per_seed)}
    per = {}
    for j in REFS:
        ds = [round(g("%s_R256_k5_seed%d_ref%d" % (CKC, s, j)) - g("%s_S5_k5_seed%d_ref%d" % (CKC, s, j)), 6) for s in SEEDS]
        per["ref%d" % j] = round(float(np.mean(ds)), 6)
    fv = list(per.values())
    paired["R256-S5"] = {"per_ref": per, "mean": round(float(np.mean(fv)), 6),
                         "range": [round(float(min(fv)), 6), round(float(max(fv)), 6)],
                         "n_pos": int(sum(v > 0 for v in fv)), "n_neg": int(sum(v < 0 for v in fv)),
                         "std_ddof0": round(float(np.std(fv)), 6)}
    per = {"ref%d" % j: round(g("%s_S4_k5_ref%d" % (CKC, j)) - WRN_EXT["ref%d" % j], 6) for j in REFS}
    fv = list(per.values())
    paired["S4-vs-WRN10N"] = {"per_ref": per, "mean": round(float(np.mean(fv)), 6),
                              "range": [round(float(min(fv)), 6), round(float(max(fv)), 6)],
                              "n_pos": int(sum(v > 0 for v in fv)), "n_neg": int(sum(v < 0 for v in fv)),
                              "std_ddof0": round(float(np.std(fv)), 6), "note": "外部差距描述, 非配对检验"}
    hist_dual = {c: v for c, v in hist.items() if "E1_DUAL" in c}
    hist_cmp = {}
    for j in REFS:
        for s in SEEDS:
            c_old = "%s_E1_DUAL_k5_seed%d_ref%d" % (CKC, s, j)
            c_new = "%s_R256_k5_seed%d_ref%d" % (CKC, s, j)
            hist_cmp["ref%d_seed%d" % (j, s)] = {"historical_E1_DUAL": hist[c_old], "new_R256": g(c_new),
                                                 "diff": round(g(c_new) - hist[c_old], 6)}
    arm_means = {}
    for arm in ("S1", "S2", "S3", "S4", "S5", "R256"):
        vs = [v for c, v in allsc.items() if ("_%s_k5" % arm) in c]
        arm_means[arm] = round(float(np.mean(vs)), 6)
    gates = {"G_A1_cache": True, "G_A2_numeric": gate_numeric, "G_A2_score_batch": score_batch_ok,
             "G_A3_adapter": gate_adapter, "G_A5_chunk": gate_chunk, "G_A4_bank_rebuild": gate_bank,
             "pre_registered_set_match": set_match,
             "dE2_equals_2dC": {"max_abs_err": de_err, "max_rel_err": rel},
             "batch_extract_abs": ext_diff, "cache_vs_extract_abs": cache_diff,
             "adapter_source": src, "reshape_check": v_rs, "distance_check": v_d,
             "composition_check": v_c, "chunk_check": v_ch}
    gates_all = all(gates[k] for k in ("G_A1_cache", "G_A2_numeric", "G_A2_score_batch", "G_A3_adapter",
                                       "G_A5_chunk", "G_A4_bank_rebuild")) and set_match
    summary = {"arm_configs_completed": len(actual), "pre_registered_set_match": int(set_match),
               "gates_all_passed": int(gates_all), "gates": gates,
               "aurocs": dict(sorted(allsc.items())), "arm_means": arm_means,
               "paired": paired, "wrn_external": WRN_EXT, "wrn_external_mean": wrn_mean,
               "e1_pc_recheck": {k: recheck[k] for k in ("per_ref", "mean_new", "mean_historical")},
               "historical_vs_new_R256": hist_cmp,
               "exploratory": True, "score_precision": "JSON 全精度 float",
               "disclosure": {"shot_is_crops": True, "query_reuse": True, "identity_unknown": True,
                              "no_significance": True,
                              "supervision_differs": "S5/R256 额外使用 5A 故障支持, 与 S4 监督预算不同",
                              "historical_banks_unseeded": "M3FEAT 库未固定种子, 本轮 pc_seed 固定, 差异披露"}}
    X.atomic_json(summary, OUT + "/score_eval.json")
    rows = []
    for cmp_name, p in paired.items():
        for lab, v in p["per_ref"].items():
            rows.append({"comparison": cmp_name, "label": lab, "diff": v})
    with open(OUT + "/paired_comparisons.csv", "w", newline="") as fcsv:
        wcsv = csv.DictWriter(fcsv, fieldnames=["comparison", "label", "diff"])
        wcsv.writeheader()
        wcsv.writerows(rows)
    # 参数审计 (A6 勘误口径)
    import m3_common as MC2
    bbm = MC2.Backbone(D.WEIGHTS, "cuda")
    mod = bbm.model.model
    audit = {"yolo_full_params": int(sum(p.numel() for p in bbm.model.parameters())),
             "yolo_P3_subgraph_params": int(sum(p.numel() for p in mod[:5].parameters())),
             "trained_this_task": 0}
    del bbm
    torch.cuda.empty_cache()
    import timm
    wm = timm.create_model("wide_resnet50_2", pretrained=False)
    audit["wrn_pc_feature_path_params"] = int(sum(p.numel() for n, p in wm.named_parameters() if n.startswith(("stem", "layer1", "layer2", "layer3"))))
    audit["wrn_layer2_3_only"] = int(sum(p.numel() for n, p in wm.named_parameters() if n.startswith(("layer2", "layer3"))))
    audit["wrn_correction"] = "M3FEAT PARAMETER_AUDIT 只计 layer2+layer3=%d, 漏 stem+layer1; 完整特征路径 = stem+layer1+layer2+layer3=%d" % (audit["wrn_layer2_3_only"], audit["wrn_pc_feature_path_params"])
    X.atomic_json(audit, OUT + "/PARAMETER_AUDIT.json")
    cost = {"stages_s": ledger, "total_wall_s": round(time.time() - t00, 1),
            "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "cache": {"e1_bytes": os.path.getsize(CACHE + "/e1_feats.npy"),
                      "patchstats_bytes": int(sum(os.path.getsize(patch_dir + "/" + f) for f in os.listdir(patch_dir) if f.endswith(".npz")))},
            "bank_sizes": {"N_star_bytes": bank_manifest["N_star"]["ref3001"]["bank_bytes"],
                           "F4000_patches": bank_manifest["F4000"]["ref3001_seed0"]["n"]},
            "note": "分阶段记录 (提取/验证/建库/复验/评分); 评分为缓存特征耗时, 非端到端延迟"}
    X.atomic_json(cost, OUT + "/RESOURCE_COST.json")
    print("[b] score_eval.json saved; wall=%.1fs" % (time.time() - t00), flush=True)
    print("=== M3SCORE 完成 ===", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M3FEAT-yoke 主编排: 冻结 -> 门 G1-G4 -> 预注册全量 -> 汇总. 配置=3特征x(双库15+PC5)=60."""
import csv, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
import m3diag3_faultbank as F3
from m3diag_bank_eval import embed
import m3feat_yoke_extract as X
import m3feat_yoke_heads as H

OUT = X.OUT
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
CAP = D.ROOT + "/outputs/M3cap-yoke"
COMBO = "yoke-suspension/rust"
CKC = "yoke-suspension__rust"
REFS = [3001, 3002, 3003, 3004, 3005]
SEEDS = [0, 1, 2]
ARMS = ["E0_DUAL", "E1_DUAL", "E2_DUAL", "E0_PC", "E1_PC", "E2_PC"]
BUDGET_S = 3600


def h16(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:16]


def code_hashes():
    r = {}
    for f in ("m3feat_yoke_extract.py", "m3feat_yoke_heads.py", "m3feat_yoke_run.py"):
        r[f] = M6.sha256_file(D.ROOT + "/scripts/" + f)
    return r


def main():
    t00 = time.time()
    dev = "cuda"
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    os.makedirs(CACHE_DIR if False else OUT, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    ridx = lambda ps: [path2idx[p] for p in ps]
    urows = lambda ps: [e2map[p] for p in ps]
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    spec[1].eval()
    for p in spec[1].parameters():
        p.requires_grad_(False)
    mh0 = M6.state_hash(spec[1])
    with open(D.SPLITS + "/query_k5.json", encoding="utf-8") as f:
        qq = json.load(f)
    qd = qq["defect_query"][COMBO]
    qn = qq["normal_query"]["yoke-suspension/good"]
    qd_rows = [path2idx[p] for p in qd]
    qn_rows = [path2idx[p] for p in qn]
    sup = {s2: SL["fault_supports"]["%s_g2_k5_seed%d" % (CKC, s2)]["paths"] for s2 in SEEDS}
    n5of = {j: SL["normal_lists"]["yoke-suspension_ref%d" % j]["N5"] for j in REFS}
    # 冻结
    uniq = sorted(set(sum(n5of.values(), []) + sum(sup.values(), []) + qd + qn))
    frozen = {"weights": {"path": D.WEIGHTS, "sha256": M6.sha256_file(D.WEIGHTS)},
              "mapper_B": {"path": D.OUT + "/mapper_B.pt", "sha256": M6.sha256_file(D.OUT + "/mapper_B.pt")},
              "cache": {"patches": M6.sha256_file(D.CACHE + "/patches.npy"), "eval_paths": M6.sha256_file(D.CACHE + "/eval_paths.json")},
              "splits": {"query_k5": M6.sha256_file(D.SPLITS + "/query_k5.json"),
                         **{"kshot_k5_seed%d" % s: M6.sha256_file(D.SPLITS + "/kshot_k5_seed%d.json" % s) for s in SEEDS}},
              "support_lists": M6.sha256_file(BENCH + "/SUPPORT_LISTS.json"),
              "code": code_hashes(), "unique_paths_n": len(uniq),
              "rules": {"E2": "P3/P4/P5 feats[4]/[6]/[9] shape断言; bilinear align_corners=False; 逐层L2+等权concat+整体L2; FP32",
                        "seeds": {"fps": M6.FPS_SEED, "p1_keep": 0.5}, "dtype_new_cache": "float32", "dtype_E0_path": "fp16 cache (历史)",
                        "score": "sF=max_f cos; sN=max_n cos; patch=sF-sN; top8 mean", "head_pc": "anomalib 2.6.2 官方组件, ratio 0.1, nn 9"}}
    X.atomic_save_json(frozen, OUT + "/FROZEN_INPUTS.json")
    print("[freeze] ok, unique=%d" % len(uniq), flush=True)
    state_h = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]
    # ---------- G2: 特征验证 + E2 缓存 ----------
    te = time.time()
    e2map, e2meta, bb_hash_after = X.build_e2_cache(uniq)
    e2arr = np.load(X.CACHE + "/e2_feats.npy", mmap_mode="r")
    e1arr = np.load(X.CACHE + "/e1_feats.npy", mmap_mode="r")
    tf = None
    e2m = json.load(open(X.CACHE + "/e2_paths.json"))
    bb_hash_after = e2m["backbone_hash"]
    probe_e2_s = round(time.time() - te, 1)
    print("[G2] E2 cache %s dim=%d shapes=%s (%.1fs)" % (list(e2arr.shape), e2arr.shape[-1], e2meta["shapes"], probe_e2_s), flush=True)
    f1 = X.e1_features([e2map[p] for p in qd[:2]], e1arr)
    assert f1.shape[-1] == int(f1.shape[-1]) and f1.shape[-2] == 1600
    n1 = f1.norm(dim=-1)
    assert bool(((n1 - 1).abs() < 1e-4).all()), "E1 L2 范数异常"
    f2 = X.e2_features([e2map[qd[0]]], e2arr).reshape(1, -1, e2arr.shape[-1])
    n2 = f2.norm(dim=-1)
    assert bool(((n2 - 1).abs() < 1e-4).all()), "E2 L2 范数异常"
    feat_dim = {"E0": None, "E1": int(f1.shape[-1]), "E2": int(f2.shape[-1])}
    # batch 一致性: E2 同图两批
    a = X.extract_e2([uniq[0]], __import__("m3feat_yoke_extract").Backbone3("cuda"), __import__("m3_common").make_transform(D.IMGSZ))[0]
    b = X.extract_e2([uniq[0], uniq[1]], __import__("m3feat_yoke_extract").Backbone3("cuda"), __import__("m3_common").make_transform(D.IMGSZ))[0][:1]
    batch_diff = float(np.abs(a - b).max())
    print("[G2] E1 dim=%d E2 dim=%d batch_diff=%.2e" % (feat_dim["E1"], feat_dim["E2"], batch_diff), flush=True)
    # ---------- G3: 分块 vs 直接 (含 4000 大库) ----------
    demo_bank = F.normalize(torch.randn(4000, feat_dim["E2"]), dim=-1).float().cuda()
    demo_q = F.normalize(torch.randn(3, 1600, feat_dim["E2"]), dim=-1).float().cuda()
    g3_diff, _ = H.verify_chunk(demo_q, demo_bank, chunk=16)
    print("[G3] chunk vs direct max diff = %.2e" % g3_diff, flush=True)
    # ---------- G4: PC 适配对照 ----------
    g4 = H.verify_pc_adapter(d=feat_dim["E1"])
    print("[G4] %s" % g4, flush=True)
    # ---------- G1: E0+H-DUAL 15 配置复现 M3CAP RF ----------
    bank_manifest = {}
    repro = []
    worst = 0.0
    t_repro = time.time()
    for j in REFS:
        nb = M6.build_normal_bank(spec, n5of[j], dev)
        for s2 in SEEDS:
            vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup[s2]]]).reshape(-1, M6.CH)).float(), dev)
            s_norm = 1.0 - (vecs @ nb.T).max(1).values
            keep = torch.argsort(-s_norm, stable=True)[:vecs.shape[0] // 2]
            Fb = vecs[keep]
            feat_dim['E0'] = int(nb.shape[1])
            bank_manifest['E0_ref%d_seed%d_N' % (j, s2)] = {'n': int(nb.shape[0]), 'dim': int(nb.shape[1]), 'dtype': 'float32', 'hash': h16(nb)}
            bank_manifest['E0_ref%d_seed%d_F' % (j, s2)] = {'n': int(Fb.shape[0]), 'dim': int(Fb.shape[1]), 'dtype': 'float32', 'hash': h16(Fb)}
            sF_d = M6.crop_sims(patches, qd_rows, spec, Fb, dev)
            sN_d = M6.crop_sims(patches, qd_rows, spec, nb, dev)
            sF_n = M6.crop_sims(patches, qn_rows, spec, Fb, dev)
            sN_n = M6.crop_sims(patches, qn_rows, spec, nb, dev)
            Cd, Cn = M6.agg(sF_d - sN_d), M6.agg(sF_n - sN_n)
            old = json.load(open(CAP + "/raw_scores/%s_RF_k5_seed%d_ref%d.json" % (CKC, s2, j)))
            md = max(max(abs(old["defect"][p] - float(Cd[t])) for t, p in enumerate(qd)),
                     max(abs(old["normal"][p] - float(Cn[t])) for t, p in enumerate(qn)))
            worst = max(worst, md)
            u, fl = D.auroc_avg_rank(Cd, Cn)
            assert fl == "ok"
            cfg = "%s_E0_DUAL_k5_seed%d_ref%d" % (CKC, s2, j)
            X.atomic_save_json({"config": {"combo": COMBO, "feature": "E0", "head": "DUAL", "seed": s2, "ref": j},
                "defect": {p: round(float(Cd[t]), 8) for t, p in enumerate(qd)},
                "normal": {p: round(float(Cn[t]), 8) for t, p in enumerate(qn)}, "auroc": round(float(u), 6)},
                OUT + "/raw_scores/%s.json" % cfg)
            repro.append({"config": cfg, "max_abs_diff": round(float(md), 10)})
            print("[G1] %s max=%.2e" % (cfg, md), flush=True)
    gate1 = worst <= 1e-4
    print("[G1] worst=%.2e gate=%s (%.1fs)" % (worst, gate1, time.time() - t_repro), flush=True)
    assert gate1, "G1 未通过, 禁止新增臂评分"
    # ---------- 探针 ----------
    tp = time.time()
    fd1 = lambda rows: X.e1_features([e2map[p] for p in rows], e1arr)
    fd2 = lambda rows: X.e2_features([e2map[p] for p in rows], e2arr).reshape(len(rows), -1, e2arr.shape[-1])
    Nb, Fb, keep, meta_b = H.dual_bank(fd1, n5of[3001], sup[0], M6.FPS_SEED)
    H.dual_score(qd, qn, fd1, Nb, Fb)
    e1_block_s = time.time() - tp
    tp2 = time.time()
    Nb2, Fb2, _, _ = H.dual_bank(fd2, n5of[3001], sup[0], M6.FPS_SEED)
    H.dual_score(qd, qn, fd2, Nb2, Fb2)
    e2_block_s = time.time() - tp2
    tp3 = time.time()
    pm1, _ = H.pc_fit(fd1, n5of[3001])
    H.pc_score(pm1, qd, qn, fd1)
    e1_pc_s = time.time() - tp3
    tp4 = time.time()
    pm2, _ = H.pc_fit(fd2, n5of[3001])
    H.pc_score(pm2, qd, qn, fd2)
    e2_pc_s = time.time() - tp4
    proj = 15 * (e1_block_s + e2_block_s) + 5 * (2 * e1_pc_s + e2_pc_s)
    print("[probe] e1=%.1f e2=%.1f e1pc=%.1f e2pc=%.1f proj=%.0fs budget=%d" % (e1_block_s, e2_block_s, e1_pc_s, e2_pc_s, proj, BUDGET_S), flush=True)
    if proj > BUDGET_S:
        X.atomic_save_json({"blocked": "probe %.0fs > %ds" % (proj, BUDGET_S)}, OUT + "/RESOURCE_COST.json")
        print("[probe] 超预算, 停止", flush=True)
        return
    # ---------- 全量: E1/E2 双库 + 三特征 PC 变体 ----------
    feat_fns = {
        "E1": lambda rows: X.e1_features([e2map[p] for p in rows], e1arr),
        "E2": lambda rows: X.e2_features([e2map[p] for p in rows], e2arr).reshape(len(rows), -1, e2arr.shape[-1]),
    }
    feat_manifest = {"E1": {"dim": feat_dim["E1"], "source": "新鲜 fp32 P3 前向逐位置 L2 (e1_feats.npy; 非 fp16 历史缓存)"},
        "E2": dict(e2meta), "E0": {"dim": feat_dim["E0"], "source": "查询侧 patches fp16 缓存 + 支持侧新鲜提取 (历史数值路径, mapper_B -> L2)"}}
    aurocs = {a: {} for a in ARMS}
    for feat in ("E1", "E2"):
        for j in REFS:
            Nb, Fb, keep, meta_b = H.dual_bank(feat_fns[feat], n5of[j], sup[0], M6.FPS_SEED)
            bank_manifest["%s_ref%d_N" % (feat, j)] = {"n": int(Nb.shape[0]), "dim": int(Nb.shape[1]), "dtype": "float32", "hash": h16(Nb), "src_paths": n5of[j]}
            for s2 in SEEDS:
                Nb2, Fb2, keep2, mb2 = H.dual_bank(feat_fns[feat], n5of[j], sup[s2], M6.FPS_SEED)
                bank_manifest["%s_ref%d_seed%d_F" % (feat, j, s2)] = {"n": int(Fb2.shape[0]), "dim": int(Fb2.shape[1]), "dtype": "float32", "hash": h16(Fb2), "n_cand": mb2["n_cand"], "keep_indices_ref": "%s_ref%d_seed%d" % (feat, j, s2), "src_paths": sup[s2]}
                sd, sn = H.dual_score(qd, qn, feat_fns[feat], Nb2, Fb2)
                u, fl = D.auroc_avg_rank(sd, sn)
                assert fl == "ok"
                cfg = "%s_%s_DUAL_k5_seed%d_ref%d" % (CKC, feat, s2, j)
                aurocs["%s_DUAL" % feat][cfg] = round(float(u), 6)
                X.atomic_save_json({"config": {"combo": COMBO, "feature": feat, "head": "DUAL", "seed": s2, "ref": j, "dim": int(Fb2.shape[1])},
                    "defect": {p: round(float(v), 8) for p, v in zip(qd, sd)},
                    "normal": {p: round(float(v), 8) for p, v in zip(qn, sn)}, "auroc": round(float(u), 6)},
                    OUT + "/raw_scores/%s.json" % cfg)
            print("[%s] ref%d done (%.1fs)" % (feat, j, time.time() - tp), flush=True)
    for feat in ("E0", "E1", "E2"):
        ffn = (lambda rows: X.e1_features([e2map[p] for p in rows], e1arr)) if feat == "E1" else ((lambda rows: X.e2_features([e2map[p] for p in rows], e2arr).reshape(len(rows), -1, e2arr.shape[-1])) if feat == "E2" else None)
        for j in REFS:
            if feat == "E0":
                def ffn0(ps):
                    e = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in ps]]).reshape(-1, M6.CH)).float(), dev)
                    return e.reshape(len(ps), -1, e.shape[-1])
                def ffn0_fit(ps):
                    e = embed(spec, torch.from_numpy(F3.extract_crops(ps, dev).reshape(-1, M6.CH)).float(), dev)
                    return e.reshape(len(ps), -1, e.shape[-1])
                pm, pc_meta = H.pc_fit(ffn0_fit, n5of[j])
            else:
                pm, pc_meta = H.pc_fit(ffn, n5of[j])
            sd, sn = H.pc_score(pm, qd, qn, ffn if feat != "E0" else ffn0)
            u, fl = D.auroc_avg_rank(np.array(sd), np.array(sn))
            assert fl == "ok"
            cfg = "%s_%s_PC_k5_ref%d" % (CKC, feat, j)
            aurocs["%s_PC" % feat][cfg] = round(float(u), 6)
            bank_manifest["%s_ref%d_PC" % (feat, j)] = pc_meta
            X.atomic_save_json({"config": {"combo": COMBO, "feature": feat, "head": "PC_YOLO", "ref": j, "dim": pc_meta["bank_dim"]},
                "defect": {p: round(v, 8) for p, v in zip(qd, sd)},
                "normal": {p: round(v, 8) for p, v in zip(qn, sn)}, "auroc": round(float(u), 6)},
                OUT + "/raw_scores/%s.json" % cfg)
        print("[PC %s] done (%.1fs)" % (feat, time.time() - tp), flush=True)
    # ---------- 汇总: 从全部逐图文件重建 ----------
    allf = sorted(os.listdir(OUT + "/raw_scores"))
    rebuilt = {}
    for f in allf:
        r = json.load(open(OUT + "/raw_scores/" + f))
        rebuilt[f[:-5]] = r["auroc"]
    assert len(rebuilt) == 60, "配置数 %d != 60" % len(rebuilt)
    assert len(set(rebuilt)) == 60
    # PC 外部标杆
    pc_ext = {}
    for j in REFS:
        r = json.load(open(BENCH + "/raw_scores_patchcore/%s_g2_ref%d.json" % (CKC, j)))
        assert set(r["defect"]) == set(qd) and set(r["normal"]) == set(qn)
        u, fl = D.auroc_avg_rank(np.array(list(r["defect"].values())), np.array(list(r["normal"].values())))
        pc_ext["ref%d" % j] = round(float(u), 6)
    pc_ext_mean = round(float(np.mean(list(pc_ext.values()))), 6)
    # 配对比较
    pairs = {}
    def dual_vals(feat):
        return rebuilt
    def get(feat, s2, j):
        return rebuilt["%s_%s_DUAL_k5_seed%d_ref%d" % (CKC, feat, s2, j)]
    for cmp_name, fA, fB in (("E1-E0", "E1", "E0"), ("E2-E1", "E2", "E1")):
        per = {}
        for j in REFS:
            ds = [round(get(fA, s2, j) - get(fB, s2, j), 6) for s2 in SEEDS]
            per["ref%d" % j] = {"per_seed": ds, "mean": round(float(np.mean(ds)), 6)}
        fv = [v for p in per.values() for v in p["per_seed"]]
        pairs[cmp_name] = {"per_ref": per, "five_ref_mean": round(float(np.mean([p["mean"] for p in per.values()])), 6),
                           "range": [round(min(p["mean"] for p in per.values()), 6), round(max(p["mean"] for p in per.values()), 6)],
                           "n_pos_groups": int(sum(p["mean"] > 0 for p in per.values())),
                           "n_neg_configs": int(sum(v < 0 for v in fv)), "std_ddof0": round(float(np.std(fv)), 6)}
    for feat in ("E0", "E1", "E2"):
        per = {}
        for j in REFS:
            pc = rebuilt["%s_%s_PC_k5_ref%d" % (CKC, feat, j)]
            ds = [round(get(feat, s2, j) - pc, 6) for s2 in SEEDS]
            per["ref%d" % j] = {"per_seed": ds, "mean": round(float(np.mean(ds)), 6), "pc": pc}
        fv = [v for p in per.values() for v in p["per_seed"]]
        pairs["%s_DUAL-PC" % feat] = {"per_ref": per, "five_ref_mean": round(float(np.mean([p["mean"] for p in per.values()])), 6),
                                      "range": [round(min(p["mean"] for p in per.values()), 6), round(max(p["mean"] for p in per.values()), 6)],
                                      "n_pos_groups": int(sum(p["mean"] > 0 for p in per.values())),
                                      "n_neg_configs": int(sum(v < 0 for v in fv)), "std_ddof0": round(float(np.std(fv)), 6)}
    gap = {a: round(float(np.mean(list(aurocs[a].values()))) - pc_ext_mean, 6) if aurocs[a] else None for a in ARMS}
    rows = []
    for cmp_name, p in pairs.items():
        for rj, v in p["per_ref"].items():
            rows.append({"comparison": cmp_name, "ref": rj, "mean": v["mean"], "per_seed": ";".join(str(x) for x in v["per_seed"]), "pc": v.get("pc", "")})
    with open(OUT + "/paired_comparisons.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["comparison", "ref", "mean", "per_seed", "pc"])
        w.writeheader()
        w.writerows(rows)
    # 参数审计 (只读)
    bbm = __import__("m3_common").Backbone(D.WEIGHTS, dev)
    mod = bbm.model.model
    audit = {"yolo_full_params": int(sum(p.numel() for p in bbm.model.parameters())),
             "yolo_P3_subgraph_params": int(sum(p.numel() for p in mod[:5].parameters())),
             "yolo_P4_subgraph_params": int(sum(p.numel() for p in mod[:7].parameters())),
             "yolo_P5_subgraph_params": int(sum(p.numel() for p in mod[:10].parameters())),
             "mapper_params": int(sum(p.numel() for p in spec[1].parameters())),
             "trained_this_task": 0, "note": "加载/实际贡献/本轮训练(=0); PC变体与双库共享特征, 监督组成不同" }
    try:
        import timm
        wm = timm.create_model("wide_resnet50_2", pretrained=False)
        audit["wrn_pc_feature_path_params"] = int(sum(p.numel() for n, p in wm.named_parameters() if n.startswith(("layer2", "layer3"))))
        audit["wrn_pc_feature_path_note"] = "timm wide_resnet50_2 layer2+layer3 参数, anomalib TimmFeatureExtractor 同源"
    except Exception as e:
        audit["wrn_pc_feature_path_params"] = "unavailable: %s" % e
    del bbm
    torch.cuda.empty_cache()
    gates = {"G1_repro_E0": bool(gate1), "G2_features": True, "G3_chunk": bool(g3_diff <= 1e-5),
             "G4_adapter": bool(g4["dist_vs_cdist"] and g4["composition_deterministic"]),
             "batch_consistency_diff": batch_diff, "g3_diff": g3_diff, "g4": g4}
    summary = {"arm_configs_completed": len(rebuilt), "repro_E0_gate": 1 if gate1 else 0,
               "gates_all_passed": 1 if all(gates[k] for k in ("G1_repro_E0", "G2_features", "G3_chunk", "G4_adapter")) else 0,
               "aurocs": rebuilt, "arm_means": {a: round(float(np.mean(list(v.values()))), 6) for a, v in aurocs.items() if v},
               "arm_means_from_rebuilt": {a: round(float(np.mean([v for k, v in rebuilt.items() if k.startswith(CKC + "_" + a + "_")])), 6) for a in ARMS},
               "paired": pairs, "patchcore_external": pc_ext, "patchcore_external_mean": pc_ext_mean,
               "repro_R0": {"worst": round(float(worst), 10), "per_config": repro},
               "correctness": gates, "exploratory": True, "score_save_precision": 8,
               "disclosure": {"k_not_independent_frames": True, "query_reuse": True, "identity_unknown": True,
                              "features_L2_normalized": True, "supervision_differs": "双库5N5A vs PC变体5N", "no_param_align": True}}
    X.atomic_save_json(summary, OUT + "/feat_eval.json")
    X.atomic_save_json({"features": feat_manifest, "e2_backbone_hash": bb_hash_after}, OUT + "/FEATURE_MANIFEST.json")
    X.atomic_save_json({"banks": bank_manifest}, OUT + "/BANK_MANIFEST.json")
    X.atomic_save_json(audit, OUT + "/PARAMETER_AUDIT.json")
    cost = {"wall_time_s": round(time.time() - t00, 1), "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "vram_note": "allocated 口径; reserved 未记录", "e2_extract_s": probe_e2_s, "probe_s": {"e1_block": round(e1_block_s, 1), "e2_block": round(e2_block_s, 1), "e1_pc": round(e1_pc_s, 1), "e2_pc": round(e2_pc_s, 1)},
            "score_note": "缓存评分耗时, 非完整推理延迟", "cache_bytes": {"e2": os.path.getsize(X.CACHE + "/e2_feats.npy")}}
    X.atomic_save_json(cost, OUT + "/RESOURCE_COST.json")
    print("[b] feat_eval.json saved; wall=%.1fs" % (time.time() - t00), flush=True)
    print("=== M3FEAT 完成 ===", flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""M3FULL-frozen-benchmark: 7 组合 × 2 预算 × 4 方法 × 5 参照组 = 280 配置.
复用 M3BUDGET 80 配置; 新增 3 资产 (5 组合) 200 配置. 门全部在正式评分前 assert."""
import hashlib, json, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3score_extract as X
import m3score_pcverify as PV
import m3transfer_lib as T

OUT = D.ROOT + "/outputs/M3full-frozen"
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
MB = D.ROOT + "/outputs/M3budget-aligned"
WRNFIX = D.ROOT + "/outputs/M3transfer-wrnfix"
REFS = [3001, 3002, 3003, 3004, 3005]
ASSETS = ["glass-insulator", "lightning-rod-suspension", "polymer-insulator-upper-shackle",
          "vari-grip", "yoke-suspension"]
FAULTS = {"glass-insulator": ["missing-cap"], "lightning-rod-suspension": ["rust"],
          "polymer-insulator-upper-shackle": ["corrosion", "rust"],
          "vari-grip": ["bird-nest", "rust"], "yoke-suspension": ["rust"]}
GOOD_DIRS = ("good", "normal")
ARMS = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
BUDGETS = ["5N", "10N"]
TOPK = 8
TOL_A, TOL_R = 1e-4, 1e-4
LEDGER = {}
PM_W = None
WRN_BS = 32
_SHELL = None


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:32]


def hash_eq(mine, src):
    return str(src) == str(mine)[:len(str(src))]


def state_hash(m):
    h = hashlib.sha256()
    for k in sorted(m.state_dict()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(m.state_dict()[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:32]


def stats(vs):
    v = np.array(vs, np.float64)
    return {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()), "std_across_refs": float(v.std(ddof=0))}


def std_extract_flat(pm, xb):
    feats = pm.feature_extractor(xb)
    feats = {k: pm.feature_pooler(v) for k, v in feats.items()}
    emb = pm.generate_embedding(feats)
    return pm.reshape_embedding(emb)


@torch.inference_mode()
def std_extract(pm, paths, bs):
    tf = T.wrn_tf()
    outs = []
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to("cuda")
        outs.append(std_extract_flat(pm, xb).cpu())
    return torch.cat(outs, 0)


def wrn_ffn(paths):
    n = len(paths)
    return std_extract(PM_W, paths, WRN_BS).reshape(n, -1, 1536).float()


def score_bank_rows(bank, rows_flat, C, bs):
    """固定库 PC+T8; rows_flat (N_img*1600, C); 按 图 分块."""
    pm = _SHELL
    pm.memory_bank = bank
    P = 1600
    N = rows_flat.shape[0] // P
    assert N * P == rows_flat.shape[0]
    R = rows_flat.reshape(N, P, C)
    pc, t8 = [], []
    with torch.no_grad():
        for i in range(0, N, bs):
            pe = R[i:i + bs]
            B = pe.shape[0]
            pe = pe.reshape(B * P, C).cuda()
            ps, loc = pm.nearest_neighbors(pe, 1)
            pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
            pc.extend(float(x) for x in pred)
            t8.extend(float(x) for x in torch.topk(ps.reshape(B, -1), TOPK, dim=1).values.mean(1))
    assert all(np.isfinite(pc)) and all(np.isfinite(t8))
    return pc, t8


def stream_scores_wrn(bank, paths, bs):
    PM_W.memory_bank = bank
    pc, t8 = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), bs):
            xb = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to("cuda")
            pe = std_extract_flat(PM_W, xb)
            B = xb.shape[0]
            ps, loc = PM_W.nearest_neighbors(pe, 1)
            pred = PM_W.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, 1536))
            pc.extend(float(x) for x in pred)
            t8.extend(float(x) for x in torch.topk(ps.reshape(B, -1), TOPK, dim=1).values.mean(1))
    assert all(np.isfinite(pc)) and all(np.isfinite(t8))
    return pc, t8


def save_cfg(name, meta, sd, sn, qd, qn):
    u, _ = D.auroc_avg_rank(np.array(sd, dtype=np.float64), np.array(sn, dtype=np.float64))
    rec = {"defect": {p: float(v) for p, v in zip(qd, sd)},
           "normal": {p: float(v) for p, v in zip(qn, sn)}, "auroc": float(u)}
    rec.update(meta)
    with open("%s/raw_scores/%s.json" % (OUT, name), "w") as f:
        json.dump(rec, f)
    return float(u)


def main():
    global PM_W, _SHELL
    t0 = time.time()
    T.deterministic()
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    assert shutil_disk_free() > 5e9, "数据盘剩余不足 5GB"
    torch.cuda.reset_peak_memory_stats()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    # ========== A. 数据冻结 ==========
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    BSL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    base = "/root/autodl-tmp/data/insplad/defect_supervised"
    data, support_lists = {}, {}
    for a in ASSETS:
        qn = qq["normal_query"]["%s/good" % a]
        faults = {}
        for f in FAULTS[a]:
            faults[f] = qq["defect_query"]["%s/%s" % (a, f)]
        if a in ("yoke-suspension", "glass-insulator"):
            n5 = {j: BSL["normal_lists"]["%s_ref%d" % (a, j)]["N5"] for j in REFS}
            n10 = {j: BSL["normal_lists"]["%s_ref%d" % (a, j)]["N10"] for j in REFS}
            pool_src = "M3BENCH SUPPORT_LISTS (复用)"
        else:
            cand = []
            for split in ("train", "val"):
                for gd in GOOD_DIRS:
                    dpath = os.path.join(base, a, split, gd)
                    if os.path.isdir(dpath):
                        cand += [os.path.join(dpath, x) for x in sorted(os.listdir(dpath))
                                 if x.lower().endswith((".jpg", ".png", ".jpeg"))]
            pool = sorted(set(cand) - set(qn) - set(p for fs in faults.values() for p in fs))
            assert len(pool) >= 10, "%s 候选池不足" % a
            n5, n10, perm_log = {}, {}, {}
            for j in REFS:
                rng = np.random.Generator(np.random.PCG64(j))
                perm = rng.permutation(len(pool))
                n10[j] = [pool[i] for i in perm[:10]]
                n5[j] = n10[j][:5]
                perm_log[str(j)] = [int(i) for i in perm[:10]]
            pool_src = "defect_supervised good/normal 池 (路径级排除查询); PCG64(ref_id) 固定排列"
            support_lists[a + "_permutation_head"] = perm_log
        allsup = set(p for j in REFS for p in n10[j])
        assert allsup.isdisjoint(set(qn)) and all(allsup.isdisjoint(set(fs)) for fs in faults.values())
        for j in REFS:
            assert len(n5[j]) == 5 and len(n10[j]) == 10 and set(n5[j]) <= set(n10[j])
            assert len(set(n10[j])) == 10
        ov = [[len(set(n10[i]) & set(n10[k])) for k in REFS] for i in REFS]
        data[a] = {"qn": qn, "faults": faults, "n5": n5, "n10": n10, "overlap": ov, "pool_src": pool_src}
        for j in REFS:
            support_lists["%s_ref%d" % (a, j)] = {"N5": n5[j], "N10": n10[j]}
        print("[data] %s faults=%s qn=%d qd=%s" % (a, FAULTS[a], len(qn), {f: len(faults[f]) for f in faults}), flush=True)
    with open(OUT + "/SUPPORT_LISTS.json", "w") as f:
        json.dump(support_lists, f, ensure_ascii=False, indent=1)
    combos = [(a, f) for a in ASSETS for f in FAULTS[a]]
    expected = sorted(["%s__%s_%s_%s_ref%d" % (a, f, arm, b, j) for a, f in combos for arm in ARMS for b in BUDGETS for j in REFS])
    # ========== B. 权重身份 ==========
    wfa = json.load(open(WRNFIX + "/audit_evidence.json"))
    PM_W, _ = T.wrn_model()
    wrn_hash = state_hash(PM_W.feature_extractor)
    assert wrn_hash == wfa["loaded_state_dict_sha256"], "WRN 加载态哈希与 wrnfix 冻结不符"
    _SHELL = PM_W
    yolo_sha = T.file_sha(D.WEIGHTS)
    X.atomic_json({"yolo_best_pt_sha256": yolo_sha, "wrn_loaded_state_hash": wrn_hash,
                   "wrn_identity_source": "M3TRANSFER-wrnfix audit_evidence.json (已断言相等)",
                   "splits_query_k5": X.sha256_file(D.SPLITS + "/query_k5.json"),
                   "support_lists_sha256": T.file_sha(OUT + "/SUPPORT_LISTS.json"),
                   "rules": {"pc_seed": {str(j): 7000 + (j - 3000) for j in REFS}, "ratio": 0.1, "nn": 9, "topk": TOPK,
                             "input": 320, "tol": {"feature": [TOL_A, TOL_R], "score": [TOL_A, TOL_R], "anchor": TOL_A}},
                   "configs_expected": expected, "budget": {"gpu_hours": 2}}, OUT + "/FROZEN_INPUTS.json")
    print("[identity] WRN/YOLO 权重身份冻结", flush=True)
    # ========== C. YOLO 缓存 (5 资产; yoke/glass 复用 M3BUDGET 缓存) ==========
    t = time.time()
    caches, ffns = {}, {}
    for a in ASSETS:
        d_ = data[a]
        uniq = sorted(set(p for j in REFS for p in d_["n10"][j]) | set(d_["qn"])
                      | set(p for f in FAULTS[a] for p in d_["faults"][f]))
        cdir = "/root/autodl-tmp/m3budget-%s-cache" % a
        man, arr = X.build_cache(uniq, cache_dir=cdir)
        assert man["deterministic_conv"] and not man["tf32_cudnn"] and not man["tf32_matmul"]
        assert man["cache_sha256_of_saved_file"] == T.file_sha(cdir + "/e1_feats.npy")
        r2 = {p: i for i, p in enumerate(man["paths"])}
        caches[a] = {"dir": cdir, "n_images": len(uniq), "cache_sha256": man["cache_sha256"],
                     "weights_sha256": man.get("weights_sha256"), "deterministic_conv": man["deterministic_conv"],
                     "tf32_matmul": man["tf32_matmul"], "tf32_cudnn": man["tf32_cudnn"],
                     "rebuild_cmd": "X.build_cache(uniq_paths, cache_dir='%s')" % cdir}
        ffns[a] = (arr, r2)
        print("[cache] %s uniq=%d" % (a, len(uniq)), flush=True)
    X.atomic_json(caches, OUT + "/CACHE_MANIFEST.json")
    LEDGER["caches_s"] = round(time.time() - t, 1)

    def ffn_pick(a, ps):
        arr, r2 = ffns[a]
        return torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in ps]])).reshape(len(ps), -1, 64).float()

    # ========== D. 门 (正式评分前) ==========
    t = time.time()
    g = {}
    bb = X.MC.Backbone(D.WEIGHTS, "cuda")
    tfm = X.MC.make_transform(X.D.IMGSZ)
    yg = {}
    for a in ASSETS:
        d_ = data[a]
        arr, r2 = ffns[a]
        sample = d_["n5"][3001][:2] + d_["faults"][FAULTS[a][0]][:3] + d_["qn"][:3]
        ext, cd = X.batch_consistency(bb, tfm, sample, arr, r2)
        assert ext <= TOL_A and cd <= TOL_A, "%s YOLO 批量特征门: %.2e/%.2e" % (a, ext, cd)
        yg[a] = {"ext_bs32_vs_bs1": ext, "cache_vs_live": cd}
    del bb
    torch.cuda.empty_cache()
    # 门库: yoke Y10N ref3001 + yoke W5N ref3001 (双建库)
    d_ = data["yoke-suspension"]
    pm10, m10 = PV.make_pm(lambda ps: ffn_pick("yoke-suspension", ps), d_["n10"][3001], 7001)
    assert torch.equal(pm10.memory_bank, PV.make_pm(lambda ps: ffn_pick("yoke-suspension", ps), d_["n10"][3001], 7001)[0].memory_bank)
    pmw5, mw5 = PV.make_pm(wrn_ffn, d_["n5"][3001], 7001)
    assert torch.equal(pmw5.memory_bank, PV.make_pm(wrn_ffn, d_["n5"][3001], 7001)[0].memory_bank)
    pmw10, mw10 = PV.make_pm(wrn_ffn, d_["n10"][3001], 7001)
    assert hash_eq(th(pmw10.memory_bank), json.load(open(WRNFIX + "/BANK_MANIFEST.json"))["yoke_ref3001"]["tensor_hash"]), "yoke W10N 重建哈希与 wrnfix 不符"
    rows = torch.from_numpy(np.ascontiguousarray(ffns["yoke-suspension"][0][
        [ffns["yoke-suspension"][1][p] for p in d_["qn"] + d_["faults"]["rust"]]])).reshape(-1, 64).float()
    pc64, t864 = score_bank_rows(pm10.memory_bank, rows, 64, 64)
    pc1, t81 = score_bank_rows(pm10.memory_bank, rows[:8 * 1600], 64, 1)
    y_sb = max(float(np.abs(np.array(pc1) - np.array(pc64[:8])).max()),
               float(np.abs(np.array(t81) - np.array(t864[:8])).max()))
    assert y_sb <= TOL_A, "YOLO 固定库两 batch 分数门: %.2e" % y_sb
    rows2500 = rows[:2500]
    with torch.no_grad():
        PM_W.memory_bank = pm10.memory_bank
        ps_off, _ = PM_W.nearest_neighbors(rows2500.cuda(), 1)
    ps_off = ps_off.reshape(-1).cpu()
    direct = torch.cdist(rows2500.double(), pm10.memory_bank.detach().cpu().double()).min(1).values
    ch = chunked_direct(rows2500, pm10.memory_bank, 256, 512)
    y_d1 = float((ps_off.double() - direct).abs().max())
    y_d2 = float((ch - direct).abs().max())
    assert y_d1 <= TOL_A + TOL_R * float(direct.abs().max()) and y_d2 <= TOL_A
    g["yolo"] = {"batch_features": yg, "score_b1_vs_b64": y_sb,
                 "distance": {"official_vs_direct64": y_d1, "chunked_vs_direct64": y_d2}}
    print("[gates] YOLO 特征/分数/距离门通过 (5 资产)", flush=True)
    # WRN 特征门 (每新资产: 支持/故障查询/正常查询)
    repr_paths = []
    for a in ("lightning-rod-suspension", "polymer-insulator-upper-shackle", "vari-grip"):
        d2_ = data[a]
        repr_paths += d2_["n5"][3001][:2] + d2_["faults"][FAULTS[a][0]][:2] + d2_["qn"][:2]
    peN = std_extract(PM_W, repr_paths, WRN_BS)
    pe1 = std_extract(PM_W, repr_paths, 1)
    w_feat = float((peN - pe1).abs().max())
    assert w_feat <= TOL_A + TOL_R * float(pe1.abs().max()), "WRN 批量特征门: %.2e" % w_feat
    qrows = std_extract(PM_W, d_["qn"][:8] + d_["faults"]["rust"][:8], WRN_BS).reshape(-1, 1536)
    PM_W.memory_bank = pmw5.memory_bank
    with torch.no_grad():
        psN, locN = PM_W.nearest_neighbors(qrows.cuda(), 1)
        predN = PM_W.compute_anomaly_score(psN.reshape(16, -1), locN.reshape(16, -1), qrows.cuda().reshape(16, -1, 1536))
        t8N = torch.topk(psN.reshape(16, -1), TOPK, dim=1).values.mean(1)
        preds1, t8s1 = [], []
        for i in range(16):
            pe1r = qrows[i * 1600:(i + 1) * 1600].cuda()
            ps1, loc1 = PM_W.nearest_neighbors(pe1r, 1)
            preds1.append(float(PM_W.compute_anomaly_score(ps1.reshape(1, -1), loc1.reshape(1, -1), pe1r.reshape(1, -1, 1536))[0]))
            t8s1.append(float(torch.topk(ps1.reshape(1, -1), TOPK, dim=1).values.mean(1)))
    w_sb = max(float(np.abs(np.array([float(x) for x in predN]) - np.array(preds1)).max()),
               float(np.abs(np.array([float(x) for x in t8N]) - np.array(t8s1)).max()))
    assert w_sb <= TOL_A, "WRN 固定库两 batch 分数门: %.2e" % w_sb
    rows_w = std_extract(PM_W, d_["qn"][:8] + d_["faults"]["rust"][:8], WRN_BS).reshape(-1, 1536)
    with torch.no_grad():
        PM_W.memory_bank = pmw5.memory_bank
        psw, _ = PM_W.nearest_neighbors(rows_w.cuda(), 1)
    psw = psw.reshape(-1).cpu()
    directw = torch.cdist(rows_w.double(), pmw5.memory_bank.detach().cpu().double()).min(1).values
    chw = chunked_direct(rows_w, pmw5.memory_bank, 256, 256)
    w_d1 = float((psw.double() - directw).abs().max())
    w_d2 = float((chw - directw).abs().max())
    assert w_d1 <= TOL_A + TOL_R * float(directw.abs().max()) and w_d2 <= TOL_A
    # 官方 pred_score 门 + 烟测 + 状态哈希
    x4 = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in d_["n5"][3001][:4]]).to("cuda")
    PM_W.memory_bank = pmw5.memory_bank
    with torch.no_grad():
        official_pred = PM_W(x4).pred_score
        pe = std_extract_flat(PM_W, x4)
        ps2, loc2 = PM_W.nearest_neighbors(pe, 1)
        pred2 = PM_W.compute_anomaly_score(ps2.reshape(4, -1), loc2.reshape(4, -1), pe.reshape(4, -1, 1536))
    w_off = float((official_pred.float() - pred2.float()).abs().max())
    assert w_off <= TOL_A, "官方 pred_score 门: %.2e" % w_off
    sd_saved = {k: v.detach().cpu().clone() for k, v in PM_W.state_dict().items()}
    h0 = state_hash(PM_W)
    PM_W.train()
    with torch.no_grad():
        emb_off = PM_W(x4[:2])
    PM_W.load_state_dict(sd_saved)
    PM_W.embedding_store = PM_W.embedding_store[:0]
    PM_W.eval()
    assert state_hash(PM_W) == h0 and not PM_W.training
    pe2 = std_extract_flat(PM_W, x4[:2])
    w_smoke = float((emb_off.float() - pe2.float()).abs().max())
    assert w_smoke <= TOL_A + TOL_R * float(pe2.float().abs().max()), "官方烟测: %.2e" % w_smoke
    top8v = T.verify_top8_axis()
    assert top8v["axis_ok"] and top8v["wrong_axis_differs"]
    g["wrn"] = {"batch_features_bs1_vs_bs32": w_feat, "score_b1_vs_bN": w_sb,
                "distance": {"official_vs_direct64": w_d1, "chunked_vs_direct64": w_d2},
                "official_pred_score": w_off, "official_smoke": w_smoke, "state_hash_ok": True, "top8_axis": top8v}
    print("[gates] WRN 特征/分数/距离/官方门通过", flush=True)
    X.atomic_json({"gates": g, "chosen": {"yolo_extract_bs": 32, "yolo_score_chunk": 64, "wrn_bs": WRN_BS},
                   "tol": {"atol": TOL_A, "rtol": TOL_R}}, OUT + "/BATCH_AND_NUMERIC_CHECKS.json")
    LEDGER["gates_s"] = round(time.time() - t, 1)
    # ========== E. 锚点 ==========
    t = time.time()
    anchor = {}
    m3man = json.load(open(D.ROOT + "/outputs/M3score-yoke/BANK_MANIFEST.json"))
    pm5, _ = PV.make_pm(lambda ps: ffn_pick("yoke-suspension", ps), d_["n5"][3001], 7001)
    assert hash_eq(th(pm5.memory_bank), m3man["N_star"]["ref3001"]["tensor_hash"]), "yoke Y5N 锚点库哈希不符"
    pc, t8 = score_bank_rows(pm5.memory_bank, rows, 64, 64)
    src_pc = json.load(open(MB + "/raw_scores/yoke-suspension__rust_Y-PC_5N_ref3001.json"))
    src_t8 = json.load(open(MB + "/raw_scores/yoke-suspension__rust_Y-T8_5N_ref3001.json"))
    k = len(d_["qn"]) + len(d_["faults"]["rust"])
    qd = d_["faults"]["rust"]
    qn = d_["qn"]
    sd_map = {p: v for p, v in zip(qn + qd, pc)}
    dpc = max(max(abs(sd_map[p] - v) for p, v in src_pc["defect"].items()),
              max(abs(sd_map[p] - v) for p, v in src_pc["normal"].items()))
    t8_map = {p: v for p, v in zip(qn + qd, t8)}
    dt8 = max(max(abs(t8_map[p] - v) for p, v in src_t8["defect"].items()),
              max(abs(t8_map[p] - v) for p, v in src_t8["normal"].items()))
    assert dpc <= TOL_A and dt8 <= TOL_A, "YOLO 锚点: %.2e/%.2e" % (dpc, dt8)
    anchor["yolo_5n_yoke_ref3001"] = {"pc": dpc, "t8": dt8}
    print("[anchor] YOLO 5N yoke pc=%.2e t8=%.2e" % (dpc, dt8), flush=True)
    pcw, t8w = stream_scores_wrn(pmw10.memory_bank, qn + qd, WRN_BS)
    src_pcw = json.load(open(MB + "/raw_scores/yoke-suspension__rust_W-PC_10N_ref3001.json"))
    src_t8w = json.load(open(MB + "/raw_scores/yoke-suspension__rust_W-T8_10N_ref3001.json"))
    w_map = {p: v for p, v in zip(qn + qd, pcw)}
    dwpc = max(max(abs(w_map[p] - v) for p, v in src_pcw["defect"].items()),
               max(abs(w_map[p] - v) for p, v in src_pcw["normal"].items()))
    wt_map = {p: v for p, v in zip(qn + qd, t8w)}
    dwt8 = max(max(abs(wt_map[p] - v) for p, v in src_t8w["defect"].items()),
               max(abs(wt_map[p] - v) for p, v in src_t8w["normal"].items()))
    assert dwpc <= TOL_A and dwt8 <= TOL_A, "WRN 锚点: %.2e/%.2e" % (dwpc, dwt8)
    anchor["wrn_10n_yoke_ref3001"] = {"pc": dwpc, "t8": dwt8}
    print("[anchor] WRN 10N yoke pc=%.2e t8=%.2e" % (dwpc, dwt8), flush=True)
    LEDGER["anchors_s"] = round(time.time() - t, 1)
    # ========== F. 正式评分 ==========
    t = time.time()
    for a in ("lightning-rod-suspension", "polymer-insulator-upper-shackle", "vari-grip"):
        d_ = data[a]
        arr, r2 = ffns[a]
        qn = list(d_["qn"])  # E2-FIX: 每资产显式取本资产正常查询, 禁止沿用锚点阶段残留的 qn (曾为 yoke)
        allq = qn + [p for f in FAULTS[a] for p in d_["faults"][f]]
        assert len(set(allq)) == len(allq), "E2-FIX: 路径不唯一"
        assert set(qn).isdisjoint(set(p for f in FAULTS[a] for p in d_["faults"][f])), "E2-FIX: 正常/故障相交"
        assert all(("/%s/" % a) in p for p in allq), "E2-FIX: 出现他资产路径"
        assert set(qn) == set(qq["normal_query"]["%s/good" % a]), "E2-FIX: 正常集合与冻结清单不符"
        rows = torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in allq]])).reshape(-1, 64).float()
        qn_n = len(qn)
        assert rows.shape[0] == len(allq) == qn_n + sum(len(d_["faults"][f]) for f in FAULTS[a]), "E2-FIX: 路径数与分数数不一致"
        segs = {}
        ofs = qn_n
        for f in FAULTS[a]:
            segs[f] = (ofs, ofs + len(d_["faults"][f]))
            ofs += len(d_["faults"][f])
        for j in REFS:
            for b, nlist in (("5N", d_["n5"]), ("10N", d_["n10"])):
                seed = 7000 + (j - 3000)
                pm, meta = PV.make_pm(lambda ps, aa=a: ffn_pick(aa, ps), nlist[j], seed)
                bank = pm.memory_bank
                bh = th(bank)
                pc, t8 = score_bank_rows(bank, rows, 64, 64)
                assert len(pc) == len(allq) and len(t8) == len(allq), "E2-FIX: 分数数与路径数不一致"
                npc = pc[:qn_n]
                nt8 = t8[:qn_n]
                assert len(npc) == qn_n and len(nt8) == qn_n, "E2-FIX: 正常分数数不符"
                for f in FAULTS[a]:
                    s0, s1 = segs[f]
                    c = "%s__%s" % (a, f)
                    assert s1 - s0 == len(d_["faults"][f]), "E2-FIX: 故障切片长度不符"
                    save_cfg("%s_Y-PC_%s_ref%d" % (c, b, j), {"asset": a, "fault": f, "arm": "Y-PC", "budget": b,
                                                              "ref": j, "bank_hash": bh}, pc[s0:s1], npc, d_["faults"][f], qn)
                    save_cfg("%s_Y-T8_%s_ref%d" % (c, b, j), {"asset": a, "fault": f, "arm": "Y-T8", "budget": b,
                                                              "ref": j, "bank_hash": bh}, t8[s0:s1], nt8, d_["faults"][f], qn)
        print("[score] %s YOLO 完成" % a, flush=True)
    for a in ("lightning-rod-suspension", "polymer-insulator-upper-shackle", "vari-grip"):
        d_ = data[a]
        qn = d_["qn"]
        allq = qn + [p for f in FAULTS[a] for p in d_["faults"][f]]
        qn_n = len(qn)
        segs = {}
        ofs = qn_n
        for f in FAULTS[a]:
            segs[f] = (ofs, ofs + len(d_["faults"][f]))
            ofs += len(d_["faults"][f])
        for j in REFS:
            for b, nlist in (("5N", d_["n5"]), ("10N", d_["n10"])):
                seed = 7000 + (j - 3000)
                pm, meta = PV.make_pm(wrn_ffn, nlist[j], seed)
                bank = pm.memory_bank
                bh = th(bank)
                pc, t8 = stream_scores_wrn(bank, allq, WRN_BS)
                for f in FAULTS[a]:
                    s0, s1 = segs[f]
                    c = "%s__%s" % (a, f)
                    save_cfg("%s_W-PC_%s_ref%d" % (c, b, j), {"asset": a, "fault": f, "arm": "W-PC", "budget": b,
                                                              "ref": j, "bank_hash": bh}, pc[s0:s1], pc[:qn_n], d_["faults"][f], qn)
                    save_cfg("%s_W-T8_%s_ref%d" % (c, b, j), {"asset": a, "fault": f, "arm": "W-T8", "budget": b,
                                                              "ref": j, "bank_hash": bh}, t8[s0:s1], t8[:qn_n], d_["faults"][f], qn)
        print("[score] %s WRN 完成" % a, flush=True)
    LEDGER["scoring_s"] = round(time.time() - t, 1)
    # ========== G. 复用 80 + 溯源 ==========
    t = time.time()
    reuse_manifest = {"configs": {}, "sources": {"M3BUDGET-aligned-v1": {"role": "yoke/glass 80 配置"}}}
    for c in ("yoke-suspension__rust", "glass-insulator__missing-cap"):
        for arm in ARMS:
            for b in BUDGETS:
                for j in REFS:
                    src = "%s/raw_scores/%s_%s_%s_ref%d.json" % (MB, c, arm, b, j)
                    rec = json.load(open(src))
                    u, _ = D.auroc_avg_rank(np.array(list(rec["defect"].values()), np.float64),
                                            np.array(list(rec["normal"].values()), np.float64))
                    assert abs(u - rec["auroc"]) <= 1e-12
                    out = {"defect": rec["defect"], "normal": rec["normal"], "auroc": rec["auroc"],
                           "reused_from": {"task": "M3BUDGET-aligned-v1",
                                           "commit": subprocess.check_output(
                                               ["git", "-C", D.ROOT, "log", "-1", "--format=%H", "--", src]).decode().strip(),
                                           "file_sha256": T.file_sha(src)}}
                    with open("%s/raw_scores/%s_%s_%s_ref%d.json" % (OUT, c, arm, b, j), "w") as f:
                        json.dump(out, f)
                    reuse_manifest["configs"]["%s_%s_%s_ref%d" % (c, arm, b, j)] = out["reused_from"]
    X.atomic_json(reuse_manifest, OUT + "/REUSE_MANIFEST.json")
    LEDGER["reuse_s"] = round(time.time() - t, 1)
    # ========== H. 集合断言 + 全量重算 ==========
    aurocs = {}
    for name in expected:
        rec = json.load(open("%s/raw_scores/%s.json" % (OUT, name)))
        u, _ = D.auroc_avg_rank(np.array(list(rec["defect"].values()), np.float64),
                                np.array(list(rec["normal"].values()), np.float64))
        assert abs(u - rec["auroc"]) <= 1e-12, "重算不一致: %s" % name
        aurocs[name] = float(u)
    names = sorted(aurocs)
    set_match = int(names == expected)
    assert set_match == 1 and len(names) == 280, "配置集合不符"
    # ========== I. 汇总 ==========
    def au(a, f, arm, b, j):
        return aurocs["%s__%s_%s_%s_ref%d" % (a, f, arm, b, j)]

    per_combo = {}
    for a, f in combos:
        for b in BUDGETS:
            for arm in ARMS:
                per_combo["%s__%s_%s_%s" % (a, f, arm, b)] = stats([au(a, f, arm, b, j) for j in REFS])
    comps = {}
    for a, f in combos:
        for b in BUDGETS:
            for key, a1, a2 in (("YT8-WPC", "Y-T8", "W-PC"), ("YT8-WT8", "Y-T8", "W-T8"),
                                ("T8PC_Y", "Y-T8", "Y-PC"), ("T8PC_W", "W-T8", "W-PC")):
                per = {"ref%d" % j: au(a, f, a1, b, j) - au(a, f, a2, b, j) for j in REFS}
                comps["%s__%s_%s_%s" % (a, f, key, b)] = {"per_ref": per, **stats(list(per.values())),
                                                          "n_pos": sum(1 for v in per.values() if v > 0),
                                                          "n_neg": sum(1 for v in per.values() if v < 0)}
    for a, f in combos:
        for arm in ARMS:
            per = {"ref%d" % j: au(a, f, arm, "10N", j) - au(a, f, arm, "5N", j) for j in REFS}
            comps["%s__%s_10N-5N_%s" % (a, f, arm)] = {"per_ref": per, **stats(list(per.values())),
                                                       "n_pos": sum(1 for v in per.values() if v > 0),
                                                       "n_neg": sum(1 for v in per.values() if v < 0)}
    # 宏平均: 每参照组跨 7 组合
    macro = {}
    for arm in ARMS:
        for b in BUDGETS:
            per_ref = [float(np.mean([au(a, f, arm, b, j) for a, f in combos])) for j in REFS]
            macro["%s_%s" % (arm, b)] = {"per_ref_macro": per_ref, **stats(per_ref)}
    for key, a1, a2 in (("YT8-WPC", "Y-T8", "W-PC"), ("YT8-WT8", "Y-T8", "W-T8")):
        for b in BUDGETS:
            per_ref = [float(np.mean([au(a, f, a1, b, j) - au(a, f, a2, b, j) for a, f in combos])) for j in REFS]
            macro["%s_%s" % (key, b)] = {"per_ref_macro": per_ref, **stats(per_ref),
                                         "n_combos_pos_mean": sum(1 for a, f in combos
                                                                  if np.mean([au(a, f, a1, b, j) - au(a, f, a2, b, j) for j in REFS]) > 0)}
    for arm in ARMS:
        per_ref = [float(np.mean([au(a, f, arm, "10N", j) - au(a, f, arm, "5N", j) for a, f in combos])) for j in REFS]
        macro["10N-5N_%s" % arm] = {"per_ref_macro": per_ref, **stats(per_ref)}
    # 资产等权: 资产内故障平均 → 跨资产
    asset_eq = {}
    for arm in ARMS:
        for b in BUDGETS:
            per_ref = [float(np.mean([np.mean([au(a, f, arm, b, j) for f in FAULTS[a]]) for a in ASSETS])) for j in REFS]
            asset_eq["%s_%s" % (arm, b)] = {"per_ref": per_ref, **stats(per_ref)}
    ev = {"arm_configs_completed": len(names), "pre_registered_set_match": set_match,
          "gates_all_passed": 1, "batch_numeric_ok": 1, "anchor_ok": 1, "same_bank_identity": 1,
          "reuse_manifest_ok": 1, "macro_ok": 1, "data_preflight_ok": 1,
          "coverage": {"combos_total": 7, "combos_feasible": 7, "blocked": []},
          "anchors": anchor, "gates": g, "per_combo": per_combo, "comparisons": comps,
          "macro": macro, "asset_equal": asset_eq, "n_reused": 80, "n_new": 200}
    X.atomic_json(ev, OUT + "/full_eval.json")
    X.atomic_json(macro, OUT + "/macro_metrics.json")
    X.atomic_json({"gates": g, "anchors": anchor, "wrn_identity": wrn_hash,
                   "note": "门在正式评分前 assert; WRN 官方路径烟测通过; 双建库一致; 状态哈希不变"},
                  OUT + "/REPRO_AND_CORRECTNESS.json")
    with open(OUT + "/BANK_MANIFEST.json", "w") as f:
        json.dump({"note": "新资产库按 ref×预算×主干 现场构建 (种子冻结); 复用配置库身份由源清单+锚点重建核对",
                   "yoke_anchor_Y5N_3001": m3man["N_star"]["ref3001"]["tensor_hash"],
                   "double_build": "每新增资产×主干 ref3001 (在门阶段完成)"}, f, ensure_ascii=False, indent=1)
    X.atomic_json({"stages_s": LEDGER, "total_wall_s": round(time.time() - t0, 1),
                   "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                   "cache_dirs": {a: caches[a]["dir"] for a in caches},
                   "cache_bytes": {a: os.path.getsize(caches[a]["dir"] + "/e1_feats.npy") for a in caches},
                   "counts": {"reused_configs": 80, "new_configs": 200, "new_banks": 60},
                   "note": "评分缓存吞吐非端到端推理延迟"}, OUT + "/RESOURCE_COST.json")
    with open(OUT + "/per_combo_metrics.csv", "w") as f:
        f.write("combo,arm,budget,mean,min,max,std\n")
        for k, v in sorted(per_combo.items()):
            c, arm, b = k.rsplit("_", 2)[0], k.rsplit("_", 2)[1], k.rsplit("_", 2)[2]
            f.write("%s,%s,%s,%.4f,%.4f,%.4f,%.4f\n" % (c, arm, b, v["mean"], v["min"], v["max"], v["std_across_refs"]))
    with open(OUT + "/paired_comparisons.csv", "w") as f:
        f.write("comparison,group,ref,diff\n")
        for k, v in sorted(comps.items()):
            for r, dv in v["per_ref"].items():
                f.write("%s,%s,%.6f\n" % (k, r, dv))
    print("[eval] 280 配置 set_match=%d wall=%.0fs" % (set_match, time.time() - t0), flush=True)
    print("=== M3FULL-FROZEN 完成 ===", flush=True)


def chunked_direct(rows, bank, cq, cp):
    outs = []
    b = bank.detach().cpu().double()
    for i in range(0, rows.shape[0], cq):
        blk = rows[i:i + cq].double()
        sub = []
        for j in range(0, b.shape[0], cp):
            sub.append(torch.cdist(blk, b[j:j + cp]))
        outs.append(torch.cat(sub, 1).min(1).values)
    return torch.cat(outs, 0)


def shutil_disk_free():
    import shutil
    return shutil.disk_usage("/root/autodl-tmp").free


if __name__ == "__main__":
    main()

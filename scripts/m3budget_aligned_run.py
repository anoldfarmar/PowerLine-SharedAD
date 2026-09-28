#!/usr/bin/env python3
"""M3BUDGET-aligned: 同支持名单/查询集/预算下 YOLO-P3 vs 标准WRN(官方池化) PC/T8 系统比较. 80 配置.
复用: YOLO-5N(M3TRANSFER) + WRN-10N(m3wrnfix); 新增: YOLO-10N + WRN-5N. 门全部在正式评分前 assert."""
import hashlib, json, os, subprocess, sys, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3score_extract as X
import m3score_pcverify as PV
import m3transfer_lib as T

OUT = D.ROOT + "/outputs/M3budget-aligned"
BENCH = D.ROOT + "/outputs/M3bench-dualbank-patchcore"
M3T = D.ROOT + "/outputs/M3transfer-top8"
WRNFIX = D.ROOT + "/outputs/M3transfer-wrnfix"
REFS = [3001, 3002, 3003, 3004, 3005]
ASSETS = {"yoke": ("yoke-suspension", "rust"), "glass": ("glass-insulator", "missing-cap")}
ARMS = ["Y-PC", "Y-T8", "W-PC", "W-T8"]
BUDGETS = ["5N", "10N"]
TOPK = 8
TOL_A, TOL_R = 1e-4, 1e-4
LEDGER, BATCHLOG = {}, {}
PM_W = None
WRN_BS = None


def th(t):
    return hashlib.sha256(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes()).hexdigest()[:32]


def state_hash(m):
    h = hashlib.sha256()
    for k in sorted(m.state_dict()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(m.state_dict()[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:32]


def save_cfg(name, meta, sd, sn, qd, qn):
    u, _ = D.auroc_avg_rank(np.array(sd, dtype=np.float64), np.array(sn, dtype=np.float64))
    rec = {"defect": {p: float(v) for p, v in zip(qd, sd)},
           "normal": {p: float(v) for p, v in zip(qn, sn)}, "auroc": float(u)}
    rec.update(meta)
    with open("%s/raw_scores/%s.json" % (OUT, name), "w") as f:
        json.dump(rec, f)
    return float(u)


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


def stream_scores_wrn(bank, paths, bs):
    """流式: 逐 batch 提取(共享) -> 对固定库 PC+T8. 不物化全量查询特征."""
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


def score_bank_rows(bank, rows_flat, C, bs):
    """固定库: 官方 NN(1)+compute_anomaly_score -> PC; 同距离 patch 轴 top8 -> T8. rows_flat: (N, C) cpu fp32."""
    pm = _SHELL
    pm.memory_bank = bank
    pc, t8 = [], []
    N = rows_flat.shape[0]
    with torch.no_grad():
        for i in range(0, N, bs):
            pe = rows_flat[i:i + bs].cuda()
            B = pe.shape[0]
            ps, loc = pm.nearest_neighbors(pe, 1)
            pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
            pc.extend(float(x) for x in pred)
            t8.extend(float(x) for x in torch.topk(ps.reshape(B, -1), TOPK, dim=1).values.mean(1))
    assert all(np.isfinite(pc)) and all(np.isfinite(t8))
    return pc, t8


def chunked_direct(rows, bank, cq, cp):
    """查询/原型分块 fp64 距离 -> 每行最小值."""
    outs = []
    b = bank.double()
    for i in range(0, rows.shape[0], cq):
        blk = rows[i:i + cq].double()
        sub = []
        for j in range(0, b.shape[0], cp):
            sub.append(torch.cdist(blk, b[j:j + cp]))
        outs.append(torch.cat(sub, 1).min(1).values)
    return torch.cat(outs, 0)


_SHELL = None


def main():
    global PM_W, WRN_BS, _SHELL
    t0 = time.time()
    T.deterministic()
    os.makedirs(OUT + "/raw_scores", exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    # ============ A. 数据身份 ============
    SL = json.load(open(BENCH + "/SUPPORT_LISTS.json"))
    qq = json.load(open(D.SPLITS + "/query_k5.json"))
    data = {}
    for a in ("yoke", "glass"):
        cat, deft = ASSETS[a]
        qd = qq["defect_query"]["%s/%s" % (cat, deft)]
        qn = qq["normal_query"]["%s/good" % cat]
        n5 = {j: SL["normal_lists"]["%s_ref%d" % (cat, j)]["N5"] for j in REFS}
        n10 = {j: SL["normal_lists"]["%s_ref%d" % (cat, j)]["N10"] for j in REFS}
        for j in REFS:
            assert len(n5[j]) == 5 and len(n10[j]) == 10 and set(n5[j]) <= set(n10[j]), "%s ref%d N5 不是 N10 子集" % (a, j)
            assert set(n10[j]).isdisjoint(set(qd)) and set(n10[j]).isdisjoint(set(qn)), "%s ref%d 支持/查询相交" % (a, j)
        data[a] = {"cat": cat, "deft": deft, "qd": qd, "qn": qn, "n5": n5, "n10": n10}
        print("[data] %s qd=%d qn=%d (N5⊂N10 断言通过)" % (a, len(qd), len(qn)), flush=True)
    # ============ B. WRN 身份 ============
    t = time.time()
    wfa = json.load(open(WRNFIX + "/audit_evidence.json"))
    PM_W, _ = T.wrn_model()
    wrn_hash = state_hash(PM_W.feature_extractor)
    assert wrn_hash == wfa["loaded_state_dict_sha256"], "WRN 加载态哈希与 wrnfix 冻结不符"
    yolo_sha = T.file_sha(D.WEIGHTS)
    print("[identity] WRN 加载态哈希=wrnfix冻结值; YOLO best.pt sha=%s" % yolo_sha[:12], flush=True)
    expected = sorted(["%s__%s_%s_%s_ref%d" % (data[a]["cat"], data[a]["deft"], arm, b, j)
                       for a in data for arm in ARMS for b in BUDGETS for j in REFS])
    X.atomic_json({"yolo_best_pt_sha256": yolo_sha, "wrn_loaded_state_hash": wrn_hash,
                   "wrn_identity_source": "M3TRANSFER-wrnfix audit_evidence.json (须相等, 已断言)",
                   "wrn_timm_checkpoint": wfa["hf_checkpoint"], "wrn_timm_pretrained_cfg": wfa["timm_pretrained_cfg"],
                   "torchvision_file_role": "仅审计对照, 非加载权重",
                   "splits_query_k5": X.sha256_file(D.SPLITS + "/query_k5.json"),
                   "support_lists": X.sha256_file(BENCH + "/SUPPORT_LISTS.json"),
                   "rules": {"pc_seed": {str(j): 7000 + (j - 3000) for j in REFS}, "ratio": 0.1, "nn": 9, "topk": TOPK,
                             "input": 320, "wrn_l2": "无", "yolo_l2": "逐位置",
                             "tol": {"feature": [TOL_A, TOL_R], "score": [TOL_A, TOL_R], "anchor": TOL_A}},
                   "batch_rule": "YOLO 提取32(缓存固定)/评分chunk起步64; WRN 16起步2倍递增, 预留20%显存, 不按AUROC",
                   "configs_expected": expected, "budget": {"gpu_hours": 1}}, OUT + "/FROZEN_INPUTS.json")
    # ============ C. YOLO 缓存 (uniq = N10∪查询, N5⊂N10) ============
    t = time.time()
    caches, ffns = {}, {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        uniq = sorted(set(d_["n10"][3001] + d_["n10"][3002] + d_["n10"][3003] + d_["n10"][3004] + d_["n10"][3005]
                          + d_["qd"] + d_["qn"]))
        cdir = "/root/autodl-tmp/m3budget-%s-cache" % a
        man, arr = X.build_cache(uniq, cache_dir=cdir)
        assert man["deterministic_conv"] and not man["tf32_cudnn"] and not man["tf32_matmul"]
        assert man["cache_sha256_of_saved_file"] == T.file_sha(cdir + "/e1_feats.npy")
        assert set(man["paths"]) == set(uniq)
        r2 = {p: i for i, p in enumerate(man["paths"])}
        caches[a] = {"dir": cdir, "n_images": len(uniq), "weights_sha256": man.get("weights_sha256"),
                     "cache_sha256": man["cache_sha256"], "deterministic_conv": man["deterministic_conv"],
                     "tf32_matmul": man["tf32_matmul"], "tf32_cudnn": man["tf32_cudnn"],
                     "norm": man.get("norm"), "rebuild_cmd": "X.build_cache(uniq_paths, cache_dir='%s')" % cdir}
        ffns[a] = (arr, r2)
        print("[cache] %s uniq=%d" % (a, len(uniq)), flush=True)
    X.atomic_json(caches, OUT + "/CACHE_MANIFEST.json")
    LEDGER["caches_s"] = round(time.time() - t, 1)
    # ============ D. 门用建库 (双建库 + WRN10N 重建核对) ============
    t = time.time()
    arr_y, r2y = ffns["yoke"]
    arr_g, r2g = ffns["glass"]

    def ffn_y(ps):
        return torch.from_numpy(np.ascontiguousarray(arr_y[[r2y[p] for p in ps]])).reshape(len(ps), -1, 64).float()

    def ffn_g(ps):
        return torch.from_numpy(np.ascontiguousarray(arr_g[[r2g[p] for p in ps]])).reshape(len(ps), -1, 64).float()

    def ffn_pick(a, ps):
        return ffn_y(ps) if a == "yoke" else ffn_g(ps)

    banks, bmeta = {}, {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        for j in (3001,):
            seed = 7000 + (j - 3000)
            pm, meta = PV.make_pm(ffn_pick(a, None) if False else (ffn_y if a == "yoke" else ffn_g), d_["n5"][j], seed)
            assert torch.equal(pm.memory_bank, PV.make_pm((ffn_y if a == "yoke" else ffn_g), d_["n5"][j], seed)[0].memory_bank), \
                "%s Y5N ref3001 双建库不一致" % a
            banks[(a, "Y", j, "5N")] = pm.memory_bank
            bmeta["%s_Y_5N_ref%d" % (a, j)] = {"n_bank": meta["n_bank"], "dtype": str(meta["bank_dtype"]),
                                               "bytes": meta["bank_bytes"], "tensor_hash": th(pm.memory_bank),
                                               "pc_seed": seed, "n_support": 5, "branch": "new"}
            pm10, meta10 = PV.make_pm((ffn_y if a == "yoke" else ffn_g), d_["n10"][j], seed)
            assert torch.equal(pm10.memory_bank, PV.make_pm((ffn_y if a == "yoke" else ffn_g), d_["n10"][j], seed)[0].memory_bank), \
                "%s Y10N ref3001 双建库不一致" % a
            banks[(a, "Y", j, "10N")] = pm10.memory_bank
            bmeta["%s_Y_10N_ref%d" % (a, j)] = {"n_bank": meta10["n_bank"], "dtype": str(meta10["bank_dtype"]),
                                                "bytes": meta10["bank_bytes"], "tensor_hash": th(pm10.memory_bank),
                                                "pc_seed": seed, "n_support": 10, "branch": "new"}
            pmw, metaw = PV.make_pm(wrn_ffn, d_["n5"][j], seed)
            assert torch.equal(pmw.memory_bank, PV.make_pm(wrn_ffn, d_["n5"][j], seed)[0].memory_bank), \
                "%s W5N ref3001 双建库不一致" % a
            banks[(a, "W", j, "5N")] = pmw.memory_bank
            bmeta["%s_W_5N_ref%d" % (a, j)] = {"n_bank": metaw["n_bank"], "dtype": str(metaw["bank_dtype"]),
                                               "bytes": metaw["bank_bytes"], "tensor_hash": th(pmw.memory_bank),
                                               "pc_seed": seed, "n_support": 5, "branch": "new"}
            # WRN 10N 重建核对 (复用锚点)
            pmw10, metaw10 = PV.make_pm(wrn_ffn, d_["n10"][j], seed)
            ref_hash = json.load(open(WRNFIX + "/BANK_MANIFEST.json"))["%s_ref%d" % (a, j)]["tensor_hash"]
            assert th(pmw10.memory_bank) == ref_hash, "%s W10N ref%d 重建库哈希与 wrnfix 不符" % (a, j)
            bmeta["%s_W_10N_ref%d" % (a, j)] = {"n_bank": metaw10["n_bank"], "tensor_hash": ref_hash,
                                                "pc_seed": seed, "n_support": 10, "branch": "reused(wrnfix), 重建哈希核对"}
        print("[banks] %s ref3001 门库+双建库完成" % a, flush=True)
    _SHELL = PM_W  # WRN shell
    LEDGER["gate_banks_s"] = round(time.time() - t, 1)
    # ============ E. 批量精度门 (正式评分前 assert) ============
    t = time.time()
    g = {}
    bb = X.MC.Backbone(D.WEIGHTS, "cuda")
    tfm = X.MC.make_transform(X.D.IMGSZ)
    yolo_gate = {}
    for a in ("yoke", "glass"):
        d_ = data[a]
        arr, r2 = ffns[a]
        ext, cd = X.batch_consistency(bb, tfm, d_["n5"][3001][:2] + d_["qd"][:3] + d_["qn"][:3], arr, r2)
        assert ext <= TOL_A and cd <= TOL_A, "%s YOLO 批量特征门超限: ext=%.2e cache=%.2e" % (a, ext, cd)
        yolo_gate[a] = {"ext_bs32_vs_bs1": ext, "cache_vs_live": cd}
    del bb
    torch.cuda.empty_cache()
    # 行级 batch (缓存行 bs1 vs bs64 精确一致) + 固定库两 batch PC/T8
    d_ = data["yoke"]
    rows = torch.from_numpy(np.ascontiguousarray(arr_y[[r2y[p] for p in d_["qd"] + d_["qn"]]])).reshape(-1, 64).float()
    r1 = rows[:1]
    assert torch.equal(rows[:64], rows[:64].clone())
    bank_y10 = banks[("yoke", "Y", 3001, "10N")]
    pc64, t864 = score_bank_rows(bank_y10, rows, 64, 64)
    pc1, t81 = score_bank_rows(bank_y10, rows[:8], 64, 1)
    y_pc_b = float(np.abs(np.array(pc1) - np.array(pc64[:8])).max())
    y_t8_b = float(np.abs(np.array(t81) - np.array(t864[:8])).max())
    assert y_pc_b <= TOL_A and y_t8_b <= TOL_A, "YOLO 固定库两 batch 分数门超限: %.2e/%.2e" % (y_pc_b, y_t8_b)
    # YOLO 距离门: 官方 vs 直接 fp64 vs 分块 (跨块+尾块)
    rows2500 = rows[:2500] if rows.shape[0] >= 2500 else rows
    with torch.no_grad():
        PM_W.memory_bank = bank_y10
        ps_off, _ = PM_W.nearest_neighbors(rows2500.cuda(), 1)
    ps_off = ps_off.reshape(-1).cpu()
    direct = torch.cdist(rows2500.double(), bank_y10.double()).min(1).values
    chunked = chunked_direct(rows2500, bank_y10, 256, 512)
    y_d1 = float((ps_off.double() - direct).abs().max())
    y_d2 = float((chunked - direct).abs().max())
    assert y_d1 <= TOL_A + TOL_R * float(direct.abs().max()), "YOLO 官方 vs 直接距离 %.2e" % y_d1
    assert y_d2 <= TOL_A, "YOLO 分块距离 %.2e" % y_d2
    g["yolo"] = {"batch_features": yolo_gate, "score_b1_vs_b64": {"pc": y_pc_b, "t8": y_t8_b},
                 "distance": {"official_vs_direct64": y_d1, "chunked_vs_direct64": y_d2,
                              "n_rows": int(rows2500.shape[0]), "chunks": "q=256(含尾块), p=512"}}
    print("[gates] YOLO 特征/分数/距离门通过", flush=True)
    # WRN 探针与 batch 选择
    d_ = data["glass"]
    probe_paths = d_["qd"][:32]
    t = time.time()
    std_extract(PM_W, probe_paths, 16)
    t16 = time.time() - t
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    v32_ok = True
    try:
        std_extract(PM_W, probe_paths, 32)
        t32 = time.time() - t
    except RuntimeError:
        v32_ok = False
        t32 = None
        torch.cuda.empty_cache()
    vram32 = torch.cuda.max_memory_allocated()
    if v32_ok and vram32 <= 0.8 * total_vram:
        WRN_BS = 32
    else:
        WRN_BS = 16
    BATCHLOG["wrn_probe"] = {"t_bs16_s": round(t16, 2), "t_bs32_s": round(t32, 2) if t32 else None,
                             "vram_bs32_pct": round(100 * vram32 / total_vram, 1), "chosen_bs": WRN_BS,
                             "rule": "16起步2倍递增; 预留20%%显存; 不按AUROC"}
    print("[probe] WRN bs16=%.2fs bs32=%s -> chosen=%d" % (t16, ("%.2fs" % t32) if t32 else "OOM", WRN_BS), flush=True)
    # WRN 特征门: bs1 vs 正式批 (正常支持/正常查询/故障查询)
    repr_paths = data["yoke"]["n5"][3001][:4] + data["yoke"]["qd"][:4] + data["yoke"]["qn"][:4] \
        + data["glass"]["n5"][3001][:2] + data["glass"]["qd"][:2] + data["glass"]["qn"][:2]
    peN = std_extract(PM_W, repr_paths, WRN_BS)
    pe1 = std_extract(PM_W, repr_paths, 1)
    w_feat = float((peN - pe1).abs().max())
    assert w_feat <= TOL_A + TOL_R * float(pe1.abs().max()), "WRN 批量特征门超限: %.2e" % w_feat
    # WRN 固定库两 batch PC/T8 (yoke W5N ref3001, 800 原型)
    bank_w5 = banks[("yoke", "W", 3001, "5N")]
    qrows = std_extract(PM_W, data["yoke"]["qd"][:8] + data["yoke"]["qn"][:8], WRN_BS).reshape(-1, 1536)
    PM_W.memory_bank = bank_w5
    with torch.no_grad():
        psN, locN = PM_W.nearest_neighbors(qrows.cuda(), 1)
        predN = PM_W.compute_anomaly_score(psN.reshape(16, -1), locN.reshape(16, -1), qrows.cuda().reshape(16, -1, 1536))
        t8N = torch.topk(psN.reshape(16, -1), TOPK, dim=1).values.mean(1)
        preds1, t8s1 = [], []
        for i in range(16):
            pe1r = qrows[i:i + 1].cuda()
            ps1, loc1 = PM_W.nearest_neighbors(pe1r, 1)
            preds1.append(float(PM_W.compute_anomaly_score(ps1.reshape(1, -1), loc1.reshape(1, -1), pe1r.reshape(1, -1, 1536))[0]))
            t8s1.append(float(torch.topk(ps1.reshape(1, -1), TOPK, dim=1).values.mean(1)))
    w_pc_b = float(np.abs(np.array([float(x) for x in predN]) - np.array(preds1)).max())
    w_t8_b = float(np.abs(np.array([float(x) for x in t8N]) - np.array(t8s1)).max())
    assert w_pc_b <= TOL_A and w_t8_b <= TOL_A, "WRN 固定库两 batch 分数门超限: %.2e/%.2e" % (w_pc_b, w_t8_b)
    # WRN 距离门 (5N 库 800 原型, 1536d; 16 图=25600 行, q=256 分块含尾块)
    rows_w = std_extract(PM_W, data["yoke"]["qd"][:8] + data["yoke"]["qn"][:8], WRN_BS).reshape(-1, 1536)
    with torch.no_grad():
        PM_W.memory_bank = bank_w5
        psw, _ = PM_W.nearest_neighbors(rows_w.cuda(), 1)
    psw = psw.reshape(-1).cpu()
    directw = torch.cdist(rows_w.double(), bank_w5.double()).min(1).values
    chunkedw = chunked_direct(rows_w, bank_w5, 256, 256)
    w_d1 = float((psw.double() - directw).abs().max())
    w_d2 = float((chunkedw - directw).abs().max())
    assert w_d1 <= TOL_A + TOL_R * float(directw.abs().max()), "WRN 官方 vs 直接 %.2e" % w_d1
    assert w_d2 <= TOL_A, "WRN 分块距离 %.2e" % w_d2
    g["wrn"] = {"batch_features_bs1_vs_bs%d" % WRN_BS: w_feat, "score_b1_vs_bN": {"pc": w_pc_b, "t8": w_t8_b},
                "distance": {"official_vs_direct64": w_d1, "chunked_vs_direct64": w_d2,
                             "n_rows": int(rows_w.shape[0]), "chunks": "q=256(含尾块), p=256"},
                "repr_coverage": "正常支持4+故障查询4+正常查询4(yoke); 各2(glass)"}
    # 官方一致烟测 (train 分支 embedding vs std_extract, 2 图) + 状态还原
    x2 = torch.stack([T.wrn_tf()(Image.open(p).convert("RGB")) for p in data["yoke"]["n5"][3001][:2]]).to("cuda")
    sd_saved = {k: v.detach().cpu().clone() for k, v in PM_W.state_dict().items()}
    h0 = state_hash(PM_W)
    PM_W.train()
    with torch.no_grad():
        emb_off = PM_W(x2)
    PM_W.load_state_dict(sd_saved)
    PM_W.embedding_store = PM_W.embedding_store[:0]
    PM_W.eval()
    assert state_hash(PM_W) == h0 and not PM_W.training
    pe2 = std_extract_flat(PM_W, x2)
    w_smoke = float((emb_off.float() - pe2.float()).abs().max())
    assert w_smoke <= TOL_A + TOL_R * float(pe2.float().abs().max()), "WRN 官方烟测超限: %.2e" % w_smoke
    g["wrn"]["official_smoke_train_branch_vs_std_extract"] = w_smoke
    g["wrn"]["state_hash_before_after"] = h0 == state_hash(PM_W)
    assert g["wrn"]["state_hash_before_after"]
    print("[gates] WRN 特征/分数/距离/烟测门通过 (bs=%d)" % WRN_BS, flush=True)
    X.atomic_json({"gates": g, "batch_log": BATCHLOG, "chosen": {"yolo_extract_bs": 32, "yolo_score_chunk": 64, "wrn_bs": WRN_BS},
                   "tol": {"atol": TOL_A, "rtol": TOL_R}, "vram_total_GiB": round(total_vram / 2**30, 1)},
                  OUT + "/BATCH_AND_NUMERIC_CHECKS.json")
    LEDGER["gates_s"] = round(time.time() - t, 1)
    # ============ F. 锚点 (每方法逐图重评分) ============
    t = time.time()
    anchor = {}
    m3man = json.load(open(D.ROOT + "/outputs/M3score-yoke/BANK_MANIFEST.json"))
    # YOLO 5N: 全 refs 库哈希核对 + ref3001 逐图重评分
    for a in ("yoke", "glass"):
        d_ = data[a]
        for j in REFS:
            seed = 7000 + (j - 3000)
            pm5, meta5 = PV.make_pm((ffn_y if a == "yoke" else ffn_g), d_["n5"][j], seed)
            if a == "yoke":
                src_hash = m3man["N_star"]["ref%d" % j]["tensor_hash"]
            else:
                src_hash = json.load(open(M3T + "/raw_scores/glass-insulator__missing-cap_Y-PC_k5_ref%d.json" % j))["bank_hash"]
            assert th(pm5.memory_bank) == src_hash, "%s Y5N ref%d 库哈希与源不符" % (a, j)
            banks[(a, "Y", j, "5N")] = pm5.memory_bank
            bmeta["%s_Y_5N_ref%d" % (a, j)] = {"n_bank": meta5["n_bank"], "dtype": str(meta5["bank_dtype"]),
                                               "bytes": meta5["bank_bytes"], "tensor_hash": src_hash,
                                               "pc_seed": seed, "n_support": 5, "branch": "reused(M3TRANSFER/M3SCORE), 重建哈希核对"}
        j = 3001
        pc, t8 = score_bank_rows(banks[(a, "Y", j, "5N")], rows if a == "yoke" else
                                 torch.from_numpy(np.ascontiguousarray((arr_g if a == "glass" else arr_y)[
                                     [(r2g if a == "glass" else r2y)[p] for p in d_["qd"] + d_["qn"]]])).reshape(-1, 64).float(), 64, 64)
        src_pc = json.load(open(M3T + "/raw_scores/%s__%s_Y-PC_k5_ref%d.json" % (d_["cat"], d_["deft"], j)))
        src_t8 = json.load(open(M3T + "/raw_scores/%s__%s_Y-T8_k5_ref%d.json" % (d_["cat"], d_["deft"], j)))
        k = len(d_["qd"])
        dpc = max(max(abs(x - v) for x, v in zip(pc[:k], src_pc["defect"].values())),
                  max(abs(x - v) for x, v in zip(pc[k:], src_pc["normal"].values())))
        dt8 = max(max(abs(x - v) for x, v in zip(t8[:k], src_t8["defect"].values())),
                  max(abs(x - v) for x, v in zip(t8[k:], src_t8["normal"].values())))
        assert dpc <= TOL_A and dt8 <= TOL_A, "%s YOLO 5N 锚点超门限 ref%d: %.2e/%.2e" % (a, j, dpc, dt8)
        anchor["yolo_5n_%s_ref3001" % a] = {"pc_max_diff": dpc, "t8_max_diff": dt8}
        print("[anchor] YOLO 5N %s ref3001 pc=%.2e t8=%.2e" % (a, dpc, dt8), flush=True)
    # WRN 10N: yoke ref3001 逐图重评分 (D 阶段已重建库并哈希核对; 流式评分)
    d_ = data["yoke"]
    pcw, t8w = stream_scores_wrn(banks[("yoke_w10n_ref3001",)], d_["qd"] + d_["qn"], WRN_BS) if False else (None, None)
    pcw, t8w = stream_scores_wrn(PV.make_pm(wrn_ffn, d_["n10"][3001], 7001)[0].memory_bank, d_["qd"] + d_["qn"], WRN_BS)
    src_pcw = json.load(open(WRNFIX + "/raw_scores/yoke-suspension__rust_W-PC-FIX_ref3001.json"))
    src_t8w = json.load(open(WRNFIX + "/raw_scores/yoke-suspension__rust_W-T8-FIX_ref3001.json"))
    k = len(d_["qd"])
    dwpc = max(max(abs(x - v) for x, v in zip(pcw[:k], src_pcw["defect"].values())),
               max(abs(x - v) for x, v in zip(pcw[k:], src_pcw["normal"].values())))
    dwt8 = max(max(abs(x - v) for x, v in zip(t8w[:k], src_t8w["defect"].values())),
               max(abs(x - v) for x, v in zip(t8w[k:], src_t8w["normal"].values())))
    assert dwpc <= TOL_A and dwt8 <= TOL_A, "WRN 10N 锚点超门限: %.2e/%.2e" % (dwpc, dwt8)
    anchor["wrn_10n_yoke_ref3001"] = {"pc_max_diff": dwpc, "t8_max_diff": dwt8}
    print("[anchor] WRN 10N yoke ref3001 pc=%.2e t8=%.2e" % (dwpc, dwt8), flush=True)
    anchor_ok = 1
    LEDGER["anchors_s"] = round(time.time() - t, 1)
    # ============ G. 正式评分: 新增分支 ============
    t = time.time()
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        arr, r2 = ffns[a]
        rows = torch.from_numpy(np.ascontiguousarray(arr[[r2[p] for p in d_["qd"] + d_["qn"]]])).reshape(-1, 64).float()
        k = len(d_["qd"])
        for j in REFS:
            if (a, "Y", j, "10N") not in banks:
                pm10, meta10 = PV.make_pm((ffn_y if a == "yoke" else ffn_g), d_["n10"][j], 7000 + (j - 3000))
                banks[(a, "Y", j, "10N")] = pm10.memory_bank
                bmeta["%s_Y_10N_ref%d" % (a, j)] = {"n_bank": meta10["n_bank"], "dtype": str(meta10["bank_dtype"]),
                                                    "bytes": meta10["bank_bytes"], "tensor_hash": th(pm10.memory_bank),
                                                    "pc_seed": 7000 + (j - 3000), "n_support": 10, "branch": "new"}
            bank = banks[(a, "Y", j, "10N")]
            bh = th(bank)
            pc, t8 = score_bank_rows(bank, rows, 64, 64)
            save_cfg("%s_Y-PC_10N_ref%d" % (c, j), {"asset": a, "arm": "Y-PC", "budget": "10N", "ref": j, "bank_hash": bh}, pc[:k], pc[k:], d_["qd"], d_["qn"])
            save_cfg("%s_Y-T8_10N_ref%d" % (c, j), {"asset": a, "arm": "Y-T8", "budget": "10N", "ref": j, "bank_hash": bh}, t8[:k], t8[k:], d_["qd"], d_["qn"])
        print("[score] %s YOLO 10N 完成" % a, flush=True)
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        qpaths = d_["qd"] + d_["qn"]
        k = len(d_["qd"])
        for j in REFS:
            if (a, "W", j, "5N") not in banks:
                pmw5, metaw5 = PV.make_pm(wrn_ffn, d_["n5"][j], 7000 + (j - 3000))
                banks[(a, "W", j, "5N")] = pmw5.memory_bank
                bmeta["%s_W_5N_ref%d" % (a, j)] = {"n_bank": metaw5["n_bank"], "dtype": str(metaw5["bank_dtype"]),
                                                   "bytes": metaw5["bank_bytes"], "tensor_hash": th(pmw5.memory_bank),
                                                   "pc_seed": 7000 + (j - 3000), "n_support": 5, "branch": "new"}
            bank = banks[(a, "W", j, "5N")]
            bh = th(bank)
            pc, t8 = stream_scores_wrn(bank, qpaths, WRN_BS)
            save_cfg("%s_W-PC_5N_ref%d" % (c, j), {"asset": a, "arm": "W-PC", "budget": "5N", "ref": j, "bank_hash": bh}, pc[:k], pc[k:], d_["qd"], d_["qn"])
            save_cfg("%s_W-T8_5N_ref%d" % (c, j), {"asset": a, "arm": "W-T8", "budget": "5N", "ref": j, "bank_hash": bh}, t8[:k], t8[k:], d_["qd"], d_["qn"])
        print("[score] %s WRN 5N 完成" % a, flush=True)
    LEDGER["scoring_s"] = round(time.time() - t, 1)
    # ============ H. 复用 40 配置 (逐图原样复制 + 溯源 + 重算) ============
    t = time.time()
    reuse_manifest = {"configs": {}, "sources": {"M3TRANSFER-top8-v1": {"role": "YOLO 5N"},
                                                  "M3TRANSFER-wrnfix-v1": {"role": "WRN 10N (官方池化)"}}}
    for a in ("yoke", "glass"):
        d_ = data[a]
        c = "%s__%s" % (d_["cat"], d_["deft"])
        for j in REFS:
            pairs = [("Y-PC", "5N", M3T + "/raw_scores/%s_Y-PC_k5_ref%d.json" % (c, j), "M3TRANSFER-top8-v1"),
                     ("Y-T8", "5N", M3T + "/raw_scores/%s_Y-T8_k5_ref%d.json" % (c, j), "M3TRANSFER-top8-v1"),
                     ("W-PC", "10N", WRNFIX + "/raw_scores/%s_W-PC-FIX_ref%d.json" % (c, j), "M3TRANSFER-wrnfix-v1"),
                     ("W-T8", "10N", WRNFIX + "/raw_scores/%s_W-T8-FIX_ref%d.json" % (c, j), "M3TRANSFER-wrnfix-v1")]
            for arm, bud, src, task in pairs:
                rec = json.load(open(src))
                u, _ = D.auroc_avg_rank(np.array(list(rec["defect"].values()), np.float64),
                                        np.array(list(rec["normal"].values()), np.float64))
                assert abs(u - rec["auroc"]) <= 1e-12, "复用配置 AUROC 重算不一致: %s" % src
                out = {"defect": rec["defect"], "normal": rec["normal"], "auroc": rec["auroc"],
                       "asset": a, "arm": arm, "budget": bud, "ref": j,
                       "reused_from": {"task": task, "commit": subprocess.check_output(
                           ["git", "-C", D.ROOT, "log", "-1", "--format=%H", "--", src]).decode().strip(),
                           "file_sha256": T.file_sha(src)}}
                name = "%s_%s_%s_ref%d" % (c, arm, bud, j)
                with open("%s/raw_scores/%s.json" % (OUT, name), "w") as f:
                    json.dump(out, f)
                reuse_manifest["configs"][name] = out["reused_from"] | {"bank_hash_source": rec.get("bank_hash", "")}
    X.atomic_json(reuse_manifest, OUT + "/REUSE_MANIFEST.json")
    LEDGER["reuse_s"] = round(time.time() - t, 1)
    # ============ I. 80 配置集合断言 + 全量重算 ============
    aurocs = {}
    for name in expected:
        rec = json.load(open("%s/raw_scores/%s.json" % (OUT, name)))
        u, _ = D.auroc_avg_rank(np.array(list(rec["defect"].values()), np.float64),
                                np.array(list(rec["normal"].values()), np.float64))
        assert abs(u - rec["auroc"]) <= 1e-12, "AUROC 重算不一致: %s" % name
        aurocs[name] = float(u)
    names = sorted(aurocs)
    set_match = int(names == expected)
    assert set_match == 1 and len(names) == 80, "配置集合与预注册不一致"
    # ============ J. 汇总 ============
    arm_means = {}
    for arm in ARMS:
        for b in BUDGETS:
            for a in data:
                d_ = data[a]
                c = "%s__%s" % (d_["cat"], d_["deft"])
                arm_means["%s_%s_%s" % (arm, b, a)] = stats([aurocs["%s_%s_%s_ref%d" % (c, arm, b, j)] for j in REFS])
    comps = {}
    pair_defs = [("5N_YT8-WT8", ("Y-T8", "5N"), ("W-T8", "5N")), ("5N_YT8-WPC", ("Y-T8", "5N"), ("W-PC", "5N")),
                 ("10N_YT8-WT8", ("Y-T8", "10N"), ("W-T8", "10N")), ("10N_YT8-WPC", ("Y-T8", "10N"), ("W-PC", "10N")),
                 ("YOLO_10N-5N_PC", ("Y-PC", "10N"), ("Y-PC", "5N")), ("YOLO_10N-5N_T8", ("Y-T8", "10N"), ("Y-T8", "5N")),
                 ("WRN_10N-5N_PC", ("W-PC", "10N"), ("W-PC", "5N")), ("WRN_10N-5N_T8", ("W-T8", "10N"), ("W-T8", "5N")),
                 ("T8-PC_Y_5N", ("Y-T8", "5N"), ("Y-PC", "5N")), ("T8-PC_Y_10N", ("Y-T8", "10N"), ("Y-PC", "10N")),
                 ("T8-PC_W_5N", ("W-T8", "5N"), ("W-PC", "5N")), ("T8-PC_W_10N", ("W-T8", "10N"), ("W-PC", "10N"))]
    for key, (a1, b1), (a2, b2) in pair_defs:
        comps[key] = {}
        for a in data:
            d_ = data[a]
            c = "%s__%s" % (d_["cat"], d_["deft"])
            per = {"ref%d" % j: aurocs["%s_%s_%s_ref%d" % (c, a1, b1, j)] - aurocs["%s_%s_%s_ref%d" % (c, a2, b2, j)] for j in REFS}
            comps[key][a] = {"per_ref": per, **stats(list(per.values())),
                             "n_pos": sum(1 for v in per.values() if v > 0), "n_neg": sum(1 for v in per.values() if v < 0)}
    ev = {"arm_configs_completed": len(names), "pre_registered_set_match": set_match,
          "gates_all_passed": 1, "batch_numeric_ok": 1, "anchor_ok": anchor_ok, "same_bank_identity": 1,
          "reuse_manifest_ok": 1, "scores_recount_match": 1,
          "gates": g, "batch": BATCHLOG, "chosen_batches": {"yolo_extract": 32, "yolo_score_chunk": 64, "wrn_bs": WRN_BS},
          "anchors": anchor, "arm_means": arm_means, "comparisons": comps,
          "bank_meta": bmeta, "n_reused_configs": 40, "n_new_configs": 40}
    X.atomic_json(ev, OUT + "/budget_eval.json")
    X.atomic_json(bmeta, OUT + "/BANK_MANIFEST.json")
    X.atomic_json({"gates": g, "anchors": anchor, "wrn_identity": {"loaded_state_hash": wrn_hash, "equals_wrnfix": True},
                   "double_build": {k: v for k, v in bmeta.items() if "ref3001" in k},
                   "note": "门在依赖的正式评分前立即 assert; 状态哈希前后一致; embedding_store 清理核查; WRN 欧氏距离非 cos 替代"},
                  OUT + "/REPRO_AND_CORRECTNESS.json")
    X.atomic_json({"stages_s": LEDGER, "total_wall_s": round(time.time() - t0, 1),
                   "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                   "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                   "cache_dirs": {a: caches[a]["dir"] for a in caches},
                   "cache_bytes": {a: os.path.getsize(caches[a]["dir"] + "/e1_feats.npy") for a in caches},
                   "bank_bytes_total": sum(v.get("bytes", 0) for v in bmeta.values()),
                   "counts": {"reused_configs": 40, "new_configs": 40, "new_banks": 20, "rebuild_verified_banks": 12},
                   "note": "评分缓存吞吐非端到端推理延迟; 不外推边缘设备"}, OUT + "/RESOURCE_COST.json")
    with open(OUT + "/paired_comparisons.csv", "w") as f:
        f.write("comparison,asset,group,diff\n")
        for key in comps:
            for a in data:
                for r, v in comps[key][a]["per_ref"].items():
                    f.write("%s,%s,%s,%.6f\n" % (key, a, r, v))
    print("[eval] 80 配置 set_match=%d wall=%.0fs" % (set_match, time.time() - t0), flush=True)
    print("=== M3BUDGET-ALIGNED 完成 ===", flush=True)


if __name__ == "__main__":
    main()

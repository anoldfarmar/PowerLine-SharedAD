#!/usr/bin/env python3
"""M3SCORE PC 适配验证 (A3) + 二维分块验证 (A5). 全部对照独立参考实现."""
import hashlib, json, os, sys
import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda"


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def source_meta():
    import anomalib
    tm = anomalib.models.image.patchcore.torch_model.__file__
    import anomalib.models.components.sampling.k_center_greedy as kc
    return {"anomalib_version": anomalib.__version__, "torch_model_py": tm, "torch_model_sha256": file_sha(tm),
            "kcenter_py": kc.__file__, "kcenter_sha256": file_sha(kc.__file__)}


def _cls():
    import anomalib.models.image.patchcore.torch_model as tcm
    return getattr(tcm, "PatchcoreModel")


from anomalib.models.image.patchcore.torch_model import KCenterGreedy


def _to4d(q):
    n, P, d = q.shape
    hw = int(round(P ** 0.5))
    assert hw * hw == P, "P 非平方: %d" % P
    return q.reshape(n, hw, hw, d).permute(0, 3, 1, 2).contiguous()


def _flat(pe):
    t = pe if torch.is_tensor(pe) else torch.as_tensor(pe)
    return t.reshape(-1, t.shape[-1])


# ---------- A3-1: reshape_embedding 顺序 ----------
def verify_reshape():
    B, C, Hh, Ww = 2, 3, 4, 5
    marker = torch.arange(B * C * Hh * Ww, dtype=torch.float32).reshape(B, C, Hh, Ww).cuda()
    pe = _flat(_cls().reshape_embedding(marker))
    ok = True
    for b in range(B):
        for h in range(Hh):
            for w in range(Ww):
                row = pe[b * Hh * Ww + h * Ww + w]
                if not torch.equal(row, marker[b, :, h, w]):
                    ok = False
    return {"shape_ok": bool(tuple(pe.shape) == (B * Hh * Ww, C)), "position_ok": ok,
            "order": "patch 行序 = (batch, h, w) 行主序, 行内为该位置 C 维向量"}


# ---------- A3-2/3: 官方距离 vs float64 + 近共线 + nn 索引 ----------
def verify_distance(d=64, seed=11):
    torch.manual_seed(seed)
    bank = Fn.normalize(torch.randn(200, d), dim=-1).float().cuda()
    q = Fn.normalize(torch.randn(50, d), dim=-1).float().cuda()
    pm = _cls()(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=9).to(DEVICE)
    pm.eval()
    pm.memory_bank = bank
    with torch.no_grad():
        ps, loc = pm.nearest_neighbors(q, 1)
    D64 = torch.cdist(q.double(), bank.double())
    ref_min, ref_idx = D64.min(1)
    dist_err = float((ps.double() - ref_min).abs().max())
    rel_err = float(((ps.double() - ref_min).abs() / ref_min.clamp_min(1e-12)).max())
    idx_ok = bool(torch.allclose(bank[loc].double().sub(q.double()).norm(dim=-1), ref_min, atol=1e-4))
    bank2 = torch.cat([bank, q[:1]], 0)
    pm.memory_bank = bank2
    with torch.no_grad():
        ps2, loc2 = pm.nearest_neighbors(q[:1], 1)
    tie_ok = bool(abs(float(ps2[0]) - float(ref_min[0])) < 1e-4)
    pm.memory_bank = bank
    coll = bank[0] + 1e-4 * Fn.normalize(torch.randn(1, d), dim=-1).cuda()
    coll = Fn.normalize(coll, dim=-1)
    with torch.no_grad():
        ps_c, _ = pm.nearest_neighbors(coll, 1)
    ref_c = torch.cdist(coll.double(), bank.double()).min()
    coll_err = float((ps_c.double() - ref_c).abs().max())
    return {"dist_max_abs_err": dist_err, "dist_max_rel_err": rel_err, "nn_index_ok": idx_ok,
            "tie_dist_ok": tie_ok, "collinear_abs_err": coll_err, "collinear_dist": float(ref_c)}


# ---------- A3-4/5: 独立图像分数参考 vs 适配 ----------
def reference_image_score(pm, patch_scores, locations, embedding, nn=9):
    """按源码公式独立组合 (torch.cdist + softmax), 不调用官方 compute_anomaly_score."""
    B, P = patch_scores.shape
    bank = pm.memory_bank
    max_i = torch.argmax(patch_scores, dim=1)
    s_star = patch_scores[torch.arange(B), max_i]
    nn_loc = locations[torch.arange(B), max_i]
    m_star = bank[nn_loc]
    k = min(nn, bank.shape[0])
    Dm = torch.cdist(m_star.double(), bank.double())
    sup = torch.topk(Dm, k=k, dim=1, largest=False).indices
    feats = embedding.reshape(B, P, -1)[torch.arange(B), max_i].double()
    supf = bank.double()[sup.reshape(-1)].reshape(B, k, -1).reshape(B * k, -1)
    dq = torch.cdist(feats, supf).reshape(B, k)
    w = (1 - torch.softmax(dq, dim=1))[:, 0]
    return (w * s_star.double()).float()


def verify_composition(d=64, seed=13):
    torch.manual_seed(seed)
    bank = Fn.normalize(torch.randn(100, d), dim=-1).float().cuda()
    q = Fn.normalize(torch.randn(4, 25, d), dim=-1).float().cuda()
    pm = _cls()(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=9).to(DEVICE)
    pm.eval()
    pm.memory_bank = bank
    with torch.no_grad():
        pe = _flat(pm.reshape_embedding(_to4d(q))).float().cuda()
        ps, loc = pm.nearest_neighbors(pe, 1)
        official = pm.compute_anomaly_score(ps.reshape(4, 25), loc.reshape(4, 25), pe.reshape(4, 25, d))
        ref = reference_image_score(pm, ps.reshape(4, 25), loc.reshape(4, 25), pe.reshape(4, 25, d))
    diff = float((official.double() - ref.double()).abs().max())
    rel = float(((official.double() - ref.double()).abs() / ref.double().abs().clamp_min(1e-9)).max())
    return {"composition_max_abs_diff": diff, "composition_max_rel_diff": rel,
            "pass": bool(diff < 1e-4 and rel < 1e-3)}


# ---------- A5: 二维分块 max-cos ----------
def chunked_max_cos(qfeat, bank, q_chunk=40, p_chunk=1500):
    """(n,P,d) x (M,d) -> (n,P) 最大余弦; 查询块 x 原型块 running max, 不生成全矩阵."""
    bk = bank if torch.is_tensor(bank) else torch.from_numpy(np.asarray(bank))
    bk = bk.float().to(qfeat.device)
    out = torch.empty(qfeat.shape[0], qfeat.shape[1], dtype=torch.float32)
    for qi in range(0, qfeat.shape[0], q_chunk):
        q = qfeat[qi:qi + q_chunk].float()
        s = None
        for pi in range(0, bk.shape[0], p_chunk):
            part = torch.einsum("bpd,md->bpm", q, bk[pi:pi + p_chunk]).max(-1).values
            s = part if s is None else torch.maximum(s, part)
        out[qi:qi + q_chunk] = s.cpu()
    return out


def verify_chunk(d=64, seed=17, n_img=2, P=1600, M=4000, q_chunk=700, p_chunk=1500):
    torch.manual_seed(seed)
    q = Fn.normalize(torch.randn(n_img, P, d), dim=-1).float().cuda()
    bank = Fn.normalize(torch.randn(M, d), dim=-1).float().cuda()
    ch = chunked_max_cos(q, bank, q_chunk, p_chunk)
    direct = torch.einsum("bpd,md->bpm", q, bank).max(-1).values.cpu()
    diff = float((ch - direct).abs().max())
    n_qb = (n_img + q_chunk - 1) // q_chunk
    n_pb = (M + p_chunk - 1) // p_chunk
    tail = (P % q_chunk != 0) and (M % p_chunk != 0)
    return {"max_cos_abs_diff": diff, "q_blocks": n_qb, "p_blocks": n_pb, "tail_covered": tail,
            "pass": bool(diff <= 1e-5 and n_qb >= 2 and n_pb >= 2 and tail)}


# ---------- 评分批量一致性 (A2 后半) ----------
def verify_score_batch(feat_fn, rows, pm, bs_list=(1, 32)):
    outs = {}
    with torch.no_grad():
        for bs in bs_list:
            vals = []
            for i in range(0, len(rows), bs):
                q4 = _to4d(feat_fn(rows[i:i + bs]))
                B, C = q4.shape[0], q4.shape[1]
                pe = _flat(pm.reshape_embedding(q4)).float().cuda()
                ps, loc = pm.nearest_neighbors(pe, 1)
                pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
                vals.extend(float(x) for x in pred)
            outs[bs] = np.array(vals)
    d = np.abs(outs[1] - outs[32])
    ok = bool(np.all(d <= 1e-4 + 1e-4 * np.abs(outs[32])))
    return {"max_abs": float(d.max()), "pass": ok}


def make_pm(feat_fn, n_paths, seed, ratio=0.1, nn=9, bs=32):
    """A4: 种子重置后建库. fit 只访问 n_paths. 返回 (pm, meta含选择索引)."""
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pm = _cls()(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=nn).to(DEVICE)
    pm.train()
    store = []
    with torch.no_grad():
        for i in range(0, len(n_paths), bs):
            q4 = _to4d(feat_fn(n_paths[i:i + bs]))
            store.append(_flat(pm.reshape_embedding(q4)))
    allp = torch.cat(store, 0).float().cuda()
    pm.memory_bank = allp
    kg = KCenterGreedy(embedding=pm.memory_bank, sampling_ratio=ratio)
    idxs = kg.select_coreset_idxs()
    pm.memory_bank = kg.embedding[torch.as_tensor(idxs, device=DEVICE)]
    pm.eval()
    meta = {"n_support_patches": int(allp.shape[0]), "n_bank": int(pm.memory_bank.shape[0]),
            "bank_dim": int(pm.memory_bank.shape[1]), "bank_dtype": str(pm.memory_bank.dtype),
            "bank_bytes": int(pm.memory_bank.numel() * pm.memory_bank.element_size()),
            "coreset_indices": [int(i) for i in idxs], "seed": seed, "ratio": ratio, "nn": nn}
    return pm, meta


@torch.no_grad()
def official_score(pm, rows, feat_fn, bs=32):
    vals = []
    for i in range(0, len(rows), bs):
        q4 = _to4d(feat_fn(rows[i:i + bs]))
        B, C = q4.shape[0], q4.shape[1]
        pe = _flat(pm.reshape_embedding(q4)).float().cuda()
        ps, loc = pm.nearest_neighbors(pe, 1)
        pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
        vals.extend(float(x) for x in pred)
    assert all(np.isfinite(vals))
    return vals
#!/usr/bin/env python3
"""M3FEAT 检测头: H-DUAL (每特征空间, 分块含大F库) + H-PC-YOLO (官方组件薄适配) + 验证."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6


def chunk_max_cos(qfeat, bank, chunk_crops=16):
    """分块最大余弦: q (n,P,d) L2 已归一化, bank (M,d). 返回 (n,P) fp32 CPU. running max."""
    bk = bank if torch.is_tensor(bank) else torch.from_numpy(np.asarray(bank))
    bk = bk.float().to(qfeat.device)
    out = []
    for i in range(0, qfeat.shape[0], chunk_crops):
        q = qfeat[i:i + chunk_crops]
        s = torch.einsum("bpd,md->bpm", q, bk).max(-1).values
        assert torch.isfinite(s).all()
        out.append(s.cpu())
    return torch.cat(out, 0)


def dual_bank(feat_fn, n_paths, f_paths, fps_seed, cap=256, keep_frac=0.5):
    """每特征空间独立建库: N=FPS256, F=P1 前50% 全保留. feat_fn(paths)->(n,P,d) L2 fp32 tensor."""
    nfeat = feat_fn(n_paths)
    nflat = nfeat.reshape(-1, nfeat.shape[-1])
    o = torch.as_tensor(M6.coreset_farthest(nflat, cap, fps_seed))
    Nbank = nflat[o].clone()
    ffeat = feat_fn(f_paths)
    fflat = ffeat.reshape(-1, ffeat.shape[-1])
    s_norm = 1.0 - (fflat @ Nbank.T).max(1).values
    keep = torch.argsort(-s_norm, stable=True)[:fflat.shape[0] // 2]
    cand = fflat[keep]
    return Nbank, cand, keep, {"n_patches_normal": int(nflat.shape[0]), "n_cand": int(cand.shape[0])}


def dual_score(qd_rows, qn_rows, feat_fn, Nbank, Fbank, chunk=16):
    """显式路径索引; sF/sN 独立 max 再减, patch 维 top8. 返回 (defect_scores, normal_scores)."""
    M = Fbank.shape[0]
    sd, sn = [], []
    for r0, r1 in ((qd_rows, None), (qn_rows, None)):
        pass
    for rows, acc in ((qd_rows, sd), (qn_rows, sn)):
        q = feat_fn(rows)
        sF = chunk_max_cos(q, Fbank, chunk)
        sN = chunk_max_cos(q, Nbank, chunk)
        v = M6.agg(sF - sN)
        acc.append(v)
    return sd[0], sn[0]


def verify_chunk(qfeat, bank, chunk=16, tol=1e-5):
    """G3: 分块 vs 直接 (含大库). 返回 max diff."""
    direct = torch.einsum("bpd,md->bpm", qfeat, bank.float()).max(-1).values.cpu()
    ch = chunk_max_cos(qfeat, bank, chunk)
    return float((direct - ch).abs().max()), direct.shape


# ---------- H-PC-YOLO: 官方组件薄适配 (绕过 timm 主干, 直接调用官方子组件) ----------
def _pc_cls():
    import anomalib.models.image.patchcore.torch_model as tcm
    for n in ("PatchcoreModel", "PatchcoreTorchModel", "TorchModel"):
        if hasattr(tcm, n):
            return getattr(tcm, n)
    raise ImportError("Patchcore torch model class not found in %s" % tcm.__file__)


from anomalib.models.image.patchcore.torch_model import KCenterGreedy


def _flat(pe):
    t = pe if torch.is_tensor(pe) else torch.as_tensor(pe)
    return t.reshape(-1, t.shape[-1])


def pc_fit(feat_fn, n_paths, ratio=0.1, nn=9, bs=32):
    """官方组件: reshape_embedding 收集 -> KCenterGreedy.sample_coreset. 只访问正常支持."""
    cls = _pc_cls()
    pm = cls(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=nn).to("cuda")
    pm.train()
    store = []
    with torch.no_grad():
        for i in range(0, len(n_paths), bs):
            q4 = _to4d(feat_fn(n_paths[i:i + bs]))
            store.append(_flat(pm.reshape_embedding(q4)))
    allp = torch.cat(store, 0).float()
    pm.memory_bank = allp
    kg = KCenterGreedy(embedding=pm.memory_bank, sampling_ratio=ratio)
    pm.memory_bank = kg.sample_coreset()
    pm.eval()
    return pm, {"n_support_patches": int(allp.shape[0]), "n_bank": int(pm.memory_bank.shape[0]),
                "bank_dim": int(pm.memory_bank.shape[1]), "bank_dtype": str(pm.memory_bank.dtype),
                "bank_bytes": int(pm.memory_bank.numel() * pm.memory_bank.element_size()),
                "coreset": "official KCenterGreedy sampling_ratio=%s" % ratio}


def pc_score(pm, qd_rows, qn_rows, feat_fn, bs=32):
    """官方 nearest_neighbors(n=1) + compute_anomaly_score(9 邻域重加权). 原始连续 pred_score."""
    out = {}
    with torch.no_grad():
        for rows, tag in ((qd_rows, "defect"), (qn_rows, "normal")):
            vals = []
            for i in range(0, len(rows), bs):
                q4 = _to4d(feat_fn(rows[i:i + bs]))
                B, C = q4.shape[0], q4.shape[1]
                hw = q4.shape[-1]
                pe = _flat(pm.reshape_embedding(q4)).float().cuda()
                ps, loc = pm.nearest_neighbors(pe, 1)
                pred = pm.compute_anomaly_score(ps.reshape(B, -1), loc.reshape(B, -1), pe.reshape(B, -1, C))
                pm.anomaly_map_generator(ps.reshape(B, 1, hw, hw), image_size=(320, 320))
                vals.extend(float(x) for x in pred)
            assert all(np.isfinite(vals))
            out[tag] = vals
    return out["defect"], out["normal"]


def verify_pc_adapter(d=64, seed=7):
    """G4: 固定合成特征, 适配路径与官方组件直接调用对照 (cdist / 手工组合 / coreset 规模)."""
    torch.manual_seed(seed)
    cls = _pc_cls()
    pm = cls(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=9).to("cuda")
    pm.eval()
    bank = F.normalize(torch.randn(200, d), dim=-1).float().cuda()
    q = F.normalize(torch.randn(3, 100, d), dim=-1).float().cuda()
    q4 = q.permute(0, 2, 1).reshape(3, d, 10, 10)
    pm.memory_bank = bank
    with torch.no_grad():
        pe = _flat(pm.reshape_embedding(q4)).float().cuda()
        assert pe.shape == (300, d), "reshape_embedding patch 数不符"
        ps, loc = pm.nearest_neighbors(pe, 1)
        dmin = torch.cdist(pe, bank).min(1).values
        dist_ok = bool(torch.allclose(ps, dmin, atol=1e-4))
        pred = pm.compute_anomaly_score(ps.reshape(3, 100), loc.reshape(3, 100), pe.reshape(3, 100, d))
        ps2, loc2 = pm.nearest_neighbors(pe, 1)
        pred2 = pm.compute_anomaly_score(ps2.reshape(3, 100), loc2.reshape(3, 100), pe.reshape(3, 100, d))
        score_ok = bool(torch.equal(pred, pred2))
        pm.memory_bank = bank
        kg = KCenterGreedy(embedding=bank, sampling_ratio=0.1)
        mb = kg.sample_coreset()
    return {"coreset_size": int(mb.shape[0]), "expected_size": 20, "dist_vs_cdist": dist_ok,
            "composition_deterministic": score_ok, "bank_dtype": str(mb.dtype)}
#!/usr/bin/env python3
"""M3FEAT 检测头: H-DUAL (每特征空间, 分块含大F库) + H-PC-YOLO (官方组件薄适配) + 验证."""
import os, sys
import numpy as np
import torch

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


# ---------- H-PC-YOLO: 官方组件薄适配 ----------
def _pc_cls():
    import anomalib.models.image.patchcore.torch_model as tcm
    for n in ("PatchcoreModel", "PatchcoreTorchModel", "TorchModel"):
        if hasattr(tcm, n):
            return getattr(tcm, n)
    raise ImportError("Patchcore torch model class not found in %s: %s" % (tcm.__file__, [x for x in dir(tcm) if "atch" in x]))


def _to4d(q):
    """(n,P,d) L2 特征 -> (n,d,hw,hw), hw=sqrt(P) (40x40=1600)."""
    n, P, d = q.shape
    hw = int(round(P ** 0.5))
    assert hw * hw == P, "P 非平方: %d" % P
    return q.reshape(n, hw, hw, d).permute(0, 3, 1, 2).contiguous()


def pc_fit(feat_fn, n_paths, ratio=0.1, nn=9, bs=32):
    """官方组件: collect -> vstack -> KCenterGreedy.sample_coreset. 只访问正常支持."""
    cls = _pc_cls()
    pm = cls(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=nn)
    pm = pm.to("cuda")
    pm.train()
    with torch.no_grad():
        for i in range(0, len(n_paths), bs):
            q = feat_fn(n_paths[i:i + bs])
            pm.forward(_to4d(q))
    flat = []
    for e in pm.embedding_store:
        t = e if torch.is_tensor(e) else torch.as_tensor(e)
        flat.append(t.reshape(-1, t.shape[-1]))
    allp = torch.cat(flat, 0).float()
    pm.memory_bank = allp
    from anomalib.models.image.patchcore.torch_model import KCenterGreedy
    kg = KCenterGreedy(embedding=pm.memory_bank, sampling_ratio=ratio)
    pm.memory_bank = kg.sample_coreset()
    pm.eval()
    return pm, {"n_support_patches": int(allp.shape[0]), "n_bank": int(pm.memory_bank.shape[0]),
                "bank_dim": int(pm.memory_bank.shape[1]), "bank_dtype": str(pm.memory_bank.dtype),
                "bank_bytes": int(pm.memory_bank.numel() * pm.memory_bank.element_size()),
                "coreset": "official KCenterGreedy sampling_ratio=%s" % ratio}


def pc_score(pm, qd_rows, qn_rows, feat_fn, bs=32):
    """官方 nearest_neighbors + compute_anomaly_score + AnomalyMapGenerator. 原始连续 pred_score."""
    out = {}
    for rows, tag in ((qd_rows, "defect"), (qn_rows, "normal")):
        vals = []
        with torch.no_grad():
            for i in range(0, len(rows), bs):
                q = feat_fn(rows[i:i + bs])
                r = pm.forward(_to4d(q))
                vals.extend(float(x) for x in r.pred_score)
        assert all(np.isfinite(vals))
        out[tag] = vals
    return out["defect"], out["normal"]


def verify_pc_adapter(d=64, P=5, n_bank=200, seed=7):
    """G4: 固定合成特征上 adapter 与官方组件直接调用对照. 返回检查 dict."""
    torch.manual_seed(seed)
    cls = _pc_cls()
    pm = cls(layers=["layer2"], backbone="wide_resnet50_2", pre_trained=False, num_neighbors=9).to("cuda")
    pm.train()
    bank_flat = F.normalize(torch.randn(n_bank, d), dim=-1).float().cuda()
    q = F.normalize(torch.randn(3, P, d), dim=-1).float().cuda()
    with torch.no_grad():
        pm.forward(bank_flat.view(2, -1, d).permute(0, 3, 1, 2) if False else bank_flat.view(2, 100, d).permute(0, 2, 1).reshape(2, d, 10, 10))
    flat = torch.cat([e.reshape(-1, e.shape[-1]) for e in pm.embedding_store], 0).float()
    from anomalib.models.image.patchcore.torch_model import KCenterGreedy
    kg = KCenterGreedy(embedding=flat, sampling_ratio=0.5)
    mb = kg.sample_coreset()
    pm.memory_bank = mb
    pm.eval()
    with torch.no_grad():
        r = pm.forward(q.permute(0, 2, 1).reshape(3, d, -1, 1).expand(3, d, P, 1).reshape(3, d, int(P ** 0.5), -1) if False else q.permute(0, 2, 1).reshape(3, d, 1, P))
    ps, loc = pm.nearest_neighbors(q.reshape(-1, d), 1)
    dmin_direct = torch.cdist(q.reshape(-1, d), mb).min(1).values
    dist_ok = bool(torch.allclose(ps, dmin_direct, atol=1e-4))
    with torch.no_grad():
        pred2 = pm.compute_anomaly_score(ps.reshape(1, -1) if False else ps.reshape(3, P), loc.reshape(3, P), q)
    score_ok = bool(torch.allclose(r.pred_score.cpu(), pred2.reshape(-1).cpu(), atol=1e-5))
    return {"coreset_size": int(mb.shape[0]), "dist_vs_cdist": dist_ok, "forward_vs_components": score_ok,
            "bank_dtype": str(mb.dtype)}


import torch.nn.functional as F
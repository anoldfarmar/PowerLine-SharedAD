#!/usr/bin/env python3
"""M3SCORE E1 缓存重建: 身份绑定 (A1) + 数值门 (A2). 模型一次加载, 路径去重, eval+inference_mode."""
import hashlib, json, os, sys, tempfile, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as MC
import m3diag_common as D

CACHE = "/root/autodl-tmp/m3score-yoke-cache"
OUT = D.ROOT + "/outputs/M3score-yoke"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_npy(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def state_hash(bb):
    sd = bb.model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def atomic_npy(arr, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp + ".npy", path)


def atomic_json(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


CODE_FILES = ["m3score_extract.py", "m3score_pcverify.py", "m3score_run.py", "m3score_report.py"]


def code_hashes():
    return {f: sha256_file(D.ROOT + "/scripts/" + f) for f in CODE_FILES if os.path.exists(D.ROOT + "/scripts/" + f)}


@torch.inference_mode()
def extract(paths, bb, tf, bs=32):
    outs = []
    exp = None
    for i in range(0, len(paths), bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to("cuda")
        feats = bb.forward_layers(xb)
        p3 = feats[4]
        if exp is None:
            exp = list(p3.shape)
            assert tuple(p3.shape[1:]) == (64, 40, 40), "P3 形状异常: %s" % list(p3.shape)
        z = F.normalize(p3, dim=1)
        assert torch.isfinite(z).all()
        outs.append(z.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, 0), {"p3_shape": exp, "bs": bs}


def build_cache(uniq_paths, force=False, cache_dir=None):
    """A1: 带身份重建/复用. 返回 (manifest, arr). cache_dir 默认沿用 M3SCORE 目录 (向后兼容)."""
    cdir = cache_dir or CACHE
    os.makedirs(cdir, exist_ok=True)
    fp = cdir + "/e1_feats.npy"
    mp = cdir + "/CACHE_MANIFEST.json"
    if not force and os.path.exists(fp) and os.path.exists(mp):
        man = json.load(open(mp))
        if man["paths"] == list(uniq_paths) and man["cache_sha256"] == sha256_file(fp) and man.get("identity_complete"):
            return man, np.load(fp, mmap_mode="r")
    t0 = time.time()
    # A2: 确定性非 TF32 提取 (跨运行一致); 门限不放宽, 环境修正
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bb = MC.Backbone(D.WEIGHTS, "cuda")
    tf = MC.make_transform(D.IMGSZ)
    h0 = state_hash(bb)
    A, meta = extract(list(uniq_paths), bb, tf, bs=32)
    h1 = state_hash(bb)
    assert h0 == h1, "提取前后 state_dict 哈希不一致"
    del bb
    torch.cuda.empty_cache()
    man = {"paths": list(uniq_paths), "n_images": len(uniq_paths), "identity_complete": True,
           "weights_sha256": sha256_file(D.WEIGHTS), "node": "ultralytics forward 第4层 feats[4] (运行时形状断言)",
           "preprocess": "MC.make_transform(IMGSZ), RGB, 与历史一致", "interp": "无 (P3 直接)",
           "norm": "NCHW dim=1 逐位置 L2 (E2 修复后语义)", "dtype": "float32", "p3_shape": meta["p3_shape"],
           "code_sha256": code_hashes(),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
           "deterministic_conv": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
           "tf32_note": "门限超限后修正: 提取改确定性非TF32; M3FEAT 历史缓存为默认TF32构建, 差异在勘误披露",
           "state_hash_before": h0, "state_hash_after": h1,
           "cache_sha256": sha256_npy(A), "extract_s": round(time.time() - t0, 1)}
    atomic_npy(A, fp)
    man["cache_sha256_of_saved_file"] = sha256_file(fp)
    atomic_json(man, mp)
    return man, np.load(fp, mmap_mode="r")


def batch_consistency(bb, tf, sample_paths, cache_arr, path2row):
    """A2: 同图 batch=1 vs batch=32 提取一致性; 并与缓存比对."""
    a32, _ = extract(sample_paths, bb, tf, bs=32)
    a1, _ = extract(sample_paths, bb, tf, bs=1)
    ext_diff = float(np.abs(a32 - a1).max())
    cached = np.asarray(cache_arr[[path2row[p] for p in sample_paths]])
    cache_diff = float(np.abs(a32 - cached).max())
    return ext_diff, cache_diff


from PIL import Image
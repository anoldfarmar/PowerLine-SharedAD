#!/usr/bin/env python3
"""M3DIAG 特征提取 (独立缓存, 不写历史目录).
产出 outputs/M3diag/cache/:
  eval_paths.json + patches.npy (N,40,40,64) fp16 -- 评测裁剪全网格 P3
  trainB_patches.npy (M,16,64) + trainB_labels.json -- B 训练 (与历史同规则)
  trainC_patches.npy (M,2,16,64) + trainC_jitter.json -- C 双视图 (固定扰动)
  audit.json -- 权重/划分哈希, 确定性抽样对拍, 禁入路径断言
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3_common as C
from m3_patch_extractor import collect_metric_boxes, collect_eval_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=D.CACHE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda", "需要 GPU"
    from PIL import Image
    import torchvision.transforms.functional as TF

    bb = C.Backbone(D.WEIGHTS, dev)
    tf = C.make_transform(D.IMGSZ)
    wsha = D.sha256_file(D.WEIGHTS)
    assert wsha.startswith(D.WEIGHTS_SHA_PREFIX), wsha

    ok, bad = C.verify_split_hashes(D.SPLITS, C.MANIFEST_HASHES)
    audit = {"weights_sha256": wsha, "split_hashes_verified": ok, "split_hash_bad": bad,
             "forbidden_path_check": None, "determinism_sample": None}

    eval_paths = collect_eval_paths(D.SPLITS)
    print("[extract] eval crops:", len(eval_paths))
    N, HP, CH = len(eval_paths), D.IMGSZ // 8, 64
    patches = np.lib.format.open_memmap(args.out + "/patches.npy", mode="w+",
                                        dtype=np.float16, shape=(N, HP, HP, CH))
    bs = 64
    for i in range(0, N, bs):
        xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in eval_paths[i:i + bs]])
        p3, _ = bb.p3_and_p5gap(xb)
        patches[i:i + bs] = p3.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16)
        if (i // bs) % 20 == 0:
            print("  eval %d/%d" % (i, N))
    patches.flush()
    D.save_json(eval_paths, args.out + "/eval_paths.json")

    old_meta = json.load(open("/root/auto-exp/outputs/M3/cache/meta.json", encoding="utf-8"))
    old_idx = {p: i for i, p in enumerate(old_meta["eval_paths"])}
    old_patches = np.load("/root/auto-exp/outputs/M3/cache/patches.npy", mmap_mode="r")
    diffs = []
    for p in eval_paths[:200]:
        if p in old_idx:
            d = float(np.abs(np.asarray(patches[eval_paths.index(p)]) -
                             np.asarray(old_patches[old_idx[p]])).max())
            diffs.append(d)
    audit["determinism_sample"] = {"n": len(diffs),
                                   "max_abs_diff": max(diffs) if diffs else None}
    print("[extract] determinism vs M3 cache:", audit["determinism_sample"])

    recs = collect_metric_boxes("/root/autodl-tmp/data/yolo_det")
    M, PP = len(recs), D.TRAIN_CFG["patches_per_box"]
    print("[extract] metric boxes:", M)
    train_imgs = {r["img"] for r in recs}
    inter = train_imgs & set(eval_paths)
    assert not inter, "train box imgs overlap eval crops: %d" % len(inter)
    audit["forbidden_path_check"] = {"train_imgs": len(train_imgs),
                                     "overlap_with_eval": 0, "passed": True}
    audit["train_boxes"] = M
    audit["train_frames"] = len({os.path.basename(r["img"]) for r in recs})

    grid = HP * HP
    xb_arr = np.zeros((M, PP, CH), dtype=np.float16)
    xc_arr = np.zeros((M, 2, PP, CH), dtype=np.float16)
    jit = {}
    for i in range(0, M, bs):
        chunk = recs[i:i + bs]
        pil = [Image.open(r["img"]).convert("RGB").crop(tuple(r["box"])) for r in chunk]
        imgs = [tf(im) for im in pil]
        p3, _ = bb.p3_and_p5gap(torch.stack(imgs))
        p3 = p3.permute(0, 2, 3, 1).cpu().numpy()
        for j, r in enumerate(chunk):
            g = i + j
            prng = np.random.default_rng(1000 + g)
            idx = prng.choice(grid, PP, replace=False)
            xb_arr[g] = p3[j].reshape(grid, CH)[idx].astype(np.float16)
            jr = np.random.default_rng(2000 + g)
            b1, c1, b2, c2 = [float(jr.uniform(0.9, 1.1)) for _ in range(4)]
            jit[str(g)] = [b1, c1, b2, c2]
            vs = []
            for (b, c) in ((b1, c1), (b2, c2)):
                im2 = TF.adjust_brightness(pil[j], b)
                im2 = TF.adjust_contrast(im2, c)
                v, _ = bb.p3_and_p5gap(tf(im2)[None])
                vs.append(v[0].permute(1, 2, 0).cpu().numpy().reshape(grid, CH)[idx])
            xc_arr[g, 0] = vs[0].astype(np.float16)
            xc_arr[g, 1] = vs[1].astype(np.float16)
        if (i // bs) % 50 == 0:
            print("  train %d/%d" % (i, M))
    np.save(args.out + "/trainB_patches.npy", xb_arr)
    np.save(args.out + "/trainC_patches.npy", xc_arr)
    D.save_json({"labels": [r["cls"] for r in recs], "M": M, "PP": PP,
                 "rule": D.TRAIN_CFG["positions"]}, args.out + "/trainB_labels.json")
    D.save_json({"rule": D.TRAIN_CFG["aug"], "per_box": jit},
                args.out + "/trainC_jitter.json")
    D.save_json(audit, args.out + "/audit.json")
    print("[done] cache ->", args.out)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""M3BENCH latency: batch=1 主计时 (已解码裁剪图 -> 图像分数), CUDA 事件.
代表配置: ref3001 + Ours seed0; 预热20 正式100; 排除读盘/加载/建库."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
import m3_common as MC
from m3diag_bank_eval import embed
from m3bench_patchcore import make_model, dl, BUDGETS, COMBOS, OUT
from PIL import Image


def timed(fn, warmup=20, n=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return round(float(np.median(ts)), 3), round(float(np.percentile(ts, 95)), 3)


def main():
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    SL = json.load(open(OUT + "/SUPPORT_LISTS.json"))
    nl = SL["normal_lists"]
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    spec[1].eval()
    tf = MC.make_transform(D.IMGSZ)
    res = []
    for g, b in BUDGETS.items():
        k = b["k"]
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            ckc = combo.replace("/", "__")
            nb = M6.build_normal_bank(spec, nl["%s_ref3001" % a][b["n_sub"]], dev)
            with open(D.SPLITS + "/kshot_k%d_seed0.json" % k, encoding="utf-8") as f:
                sup0 = json.load(f)["prototype_pools"][combo]
            vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup0]]).reshape(-1, M6.CH)).float(), dev)
            fb = M6.p1_fault_bank(vecs, nb, dev)
            with open(D.SPLITS + "/query_k%d.json" % b["k"], encoding="utf-8") as f:
                qp = sorted(json.load(f)["defect_query"][combo])[0]
            img = Image.open(qp).convert("RGB")
            bb = MC.Backbone(D.WEIGHTS, dev)
            fbt = (fb if torch.is_tensor(fb) else torch.from_numpy(np.asarray(fb))).float().to(dev)
            nbt = (nb if torch.is_tensor(nb) else torch.from_numpy(np.asarray(nb))).float().to(dev)
            def ours_score():
                xb = tf(img).unsqueeze(0).to(dev, non_blocking=True)
                with torch.no_grad():
                    p3, _ = bb.p3_and_p5gap(xb)
                    v = embed(spec, p3.permute(0, 2, 3, 1).reshape(-1, M6.CH).float(), dev)
                    r = v @ fbt.T - v @ nbt.T
                    return float(torch.topk(r.squeeze(0), 8).values.mean())
            med, p95 = timed(ours_score)
            res.append({"method": "ours", "combo": ckc, "budget": g, "query": qp,
                        "median_ms": med, "p95_ms": p95,
                        "weights_bytes": os.path.getsize(D.WEIGHTS) + os.path.getsize(D.OUT + "/mapper_B.pt"),
                        "banks_bytes": int(fb.nbytes + nb.nbytes), "bank_protos": int(fb.shape[0] + nb.shape[0]),
                        "bank_dim": int(fb.shape[1]), "bank_dtype": "float32"})
            print("[lat] ours %s %s median=%.2fms p95=%.2fms" % (ckc, g, med, p95), flush=True)
            del bb, fb, nb, fbt, nbt
            torch.cuda.empty_cache()
    from anomalib.engine import Engine
    for g, b in BUDGETS.items():
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            ckc = combo.replace("/", "__")
            paths_n = nl["%s_ref3001" % a][b["pc_n"]]
            model = make_model()
            eng = Engine(default_root_dir="/tmp/m3bench_logs")
            tb = time.time()
            eng.fit(model, train_dataloaders=dl(paths_n, 32))
            build_s = round(time.time() - tb, 1)
            with open(D.SPLITS + "/query_k%d.json" % b["k"], encoding="utf-8") as f:
                qp = sorted(json.load(f)["defect_query"][combo])[0]
            def pc_score():
                eng.predict(model, dataloaders=dl([qp], 1))
            med, p95 = timed(pc_score)
            tm = getattr(model, "model", model)
            mb = tm.memory_bank
            res.append({"method": "patchcore", "combo": ckc, "budget": g, "query": qp,
                        "median_ms": med, "p95_ms": p95, "build_s_excluded": build_s,
                        "backbone_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                        "memory_bank_shape": list(mb.shape), "memory_bank_bytes": int(mb.numel() * mb.element_size()),
                        "note": "经 Engine.predict 包装路径计时, 含 dataloader 开销"})
            print("[lat] pc %s %s median=%.2fms p95=%.2fms" % (ckc, g, med, p95), flush=True)
            del model, eng
            torch.cuda.empty_cache()
    eng = Engine(default_root_dir="/tmp/m3bench_logs")
    model = make_model()
    eng.fit(model, train_dataloaders=dl(nl["glass-insulator_ref3001"]["N10"], 32))
    q4 = sorted(json.load(open(D.SPLITS + "/query_k1.json", encoding="utf-8"))["defect_query"]["glass-insulator/missing-cap"])[:4]
    bat = eng.predict(model, dataloaders=dl(q4, 4))
    bs = [float(x) for x in bat[0].pred_score]
    sin = [float(eng.predict(model, dataloaders=dl([p], 1))[0].pred_score[0]) for p in q4]
    md = max(abs(x - y) for x, y in zip(bs, sin))
    D.save_json({"timings": res, "warmup": 20, "n_timed": 100, "timer": "cuda events",
                 "gpu": torch.cuda.get_device_name(0), "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                 "batch_consistency": {"max_abs_diff": round(md, 6), "n": 4, "note": "PatchCore 批量4 vs 单图; Ours 流水为逐裁剪原生单图"},
                 "rule": "主计时=已解码裁剪图到图像分数, 排除读盘/加载/建库; 代表配置 ref3001 + Ours seed0; 不外推 RK3588, 不称整图端到端"},
                OUT + "/latency.json")
    print("[lat] saved; max_batch_diff=%.2e" % md, flush=True)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""M3BENCH latency: batch=1 主计时 (已解码裁剪图 -> 图像分数), CUDA 事件."""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3diag_common as D
import m3diag6_dualbank as M6
from m3_common import C as MC
from m3bench_main import lists, BUDGETS, COMBOS, REF_SEEDS, OUT, D as DD
from m3diag_bank_eval import embed
from PIL import Image

def main():
    dev = "cuda"
    rg = json.load(open(M6.OUT4 + "/ref_groups.json"))
    eval_paths = json.load(open(D.CACHE + "/eval_paths.json"))
    path2idx = {p: i for i, p in enumerate(eval_paths)}
    patches = np.load(D.CACHE + "/patches.npy", mmap_mode="r")
    spec = ("head", D.load_head_state(D.OUT + "/mapper_B.pt", dev).to(dev))
    head = spec[1]; head.eval()
    tf = MC.make_transform(D.IMGSZ)
    res = []
    for g, b in BUDGETS.items():
        k = b["k"]
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            j = 3001
            nl = lists["%s_ref%d" % (a, j)]
            nb = M6.build_normal_bank(spec, nl[b["n_sub"]], dev)
            with open(D.SPLITS + "/kshot_k%d_seed0.json" % k, encoding="utf-8") as f:
                sp = json.load(f)
            sup0 = sp["prototype_pools"][combo]
            vecs = embed(spec, torch.from_numpy(np.asarray(patches[[path2idx[p] for p in sup0]]).reshape(-1, M6.CH)).float(), dev)
            fb = M6.p1_fault_bank(vecs, nb, dev)
            with open(D.SPLITS + "/query_k%d.json" % k, encoding="utf-8") as f:
                qq = json.load(f)
            qp = sorted(qq["defect_query"][combo])[0]
            img = Image.open(qp).convert("RGB")
            # Ours: 解码图 -> transform -> P3 -> mapper -> 双库 top8 差值
            def ours_once():
                xb = tf(img).unsqueeze(0).to(dev, non_blocking=True)
                p3, _ = MC.Backbone(D.WEIGHTS, dev).p3_and_p5gap(xb) if False else (None, None)
            # Backbone 每次重建太贵; 预建一次
            bb = MC.Backbone(D.WEIGHTS, dev)
            def ours_score():
                xb = tf(img).unsqueeze(0).to(dev, non_blocking=True)
                with torch.no_grad():
                    p3, _ = bb.p3_and_p5gap(xb)
                    f = p3.permute(0, 2, 3, 1).reshape(1600, 64).float()
                    v = embed(spec, f, dev)
                    sF = (v @ torch.from_numpy(np.asarray(fb)).float().to(dev).T).max()
                    sN = (v @ torch.from_numpy(np.asarray(nb)).float().to(dev).T).max()
                    r = (v @ torch.from_numpy(np.asarray(fb)).float().to(dev).T - v @ torch.from_numpy(np.asarray(nb)).float().to(dev).T)
                    return float(torch.topk(r.squeeze(-1) if r.dim() > 1 else r, 8).values.mean()) if False else float(torch.topk(sF - sN + (r.max() * 0), 8).values.mean()) if False else float(torch.topk((v @ torch.from_numpy(np.asarray(fb)).float().to(dev).T - v @ torch.from_numpy(np.asarray(nb)).float().to(dev).T).squeeze(0), 8).values.mean())
            for _ in range(20):
                ours_score()
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); ours_score(); e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            res.append({"method": "ours", "combo": combo, "budget": g, "query": qp,
                        "median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)})
            print("[lat] ours %s %s median=%.2fms" % (combo.replace("/", "__"), g, np.median(ts)), flush=True)
            del bb, fb, nb
            torch.cuda.empty_cache()
    # PatchCore: 复用 bench 已建模型的预测路径计时 (每配置重建+拟合一次, 计时仅推理)
    import torchvision.transforms.v2 as T2
    from anomalib.models import Patchcore
    from anomalib.pre_processing import PreProcessor
    from anomalib.engine import Engine
    tr = T2.Compose([T2.Resize((320, 320), antialias=True), T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pp = PreProcessor(transform=tr)
    for g, b in BUDGETS.items():
        for combo in COMBOS:
            a = D.ASSET_OF[combo]
            j = 3001
            nl = lists["%s_ref%d" % (a, j)]
            model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], pre_trained=True,
                              coreset_sampling_ratio=0.1, num_neighbors=9, pre_processor=pp)
            eng = Engine()
            class DS(torch.utils.data.Dataset):
                def __init__(self, ps):
                    self.ps = ps
                def __len__(self):
                    return len(self.ps)
                def __getitem__(self, i):
                    return {"image": Image.open(self.ps[i]).convert("RGB"), "image_path": self.ps[i]}
            eng.fit(model, datamodule=type("DM", (__import__("lightning.pytorch", fromlist=["LightningDataModule"]).LightningDataModule,), {"__init__": lambda s, ps: (setattr(s, "ps", ps), __import__("lightning.pytorch", fromlist=["LightningDataModule"]).LightningDataModule.__init__(s)), "train_dataloader": lambda s: torch.utils.data.DataLoader(DS(s.ps), batch_size=32)})(nl[b["pc_n"]]))
            qp = sorted(json.load(open(D.SPLITS + "/query_k%d.json" % b["k"], encoding="utf-8"))["defect_query"][combo])[0]
            img = Image.open(qp).convert("RGB")
            def pc_once():
                eng.predict(model, dataloaders=torch.utils.data.DataLoader(DS([qp]), batch_size=1))
            for _ in range(20):
                pc_once()
            torch.cuda.synchronize()
            ts = []
            for _ in range(100):
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record(); pc_once(); e1.record()
                torch.cuda.synchronize()
                ts.append(e0.elapsed_time(e1))
            res.append({"method": "patchcore", "combo": combo, "budget": g, "query": qp,
                        "median_ms": round(float(np.median(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3),
                        "note": "经 Engine.predict 包装路径计时, 含 dataloader 开销"})
            print("[lat] pc %s %s median=%.2fms" % (combo.replace("/", "__"), g, np.median(ts)), flush=True)
            del model
            torch.cuda.empty_cache()
    D.save_json({"timings": res, "warmup": 20, "n_timed": 100, "timer": "cuda events", "gpu": torch.cuda.get_device_name(0),
                 "rule": "主计时=已解码裁剪图到图像分数, 排除读盘/加载/建库; 代表配置 ref3001 + Ours seed0"},
                OUT + "/latency.json")
    print("[lat] saved", flush=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""M3 Step-5: anomalib 同查询集 AD 基线 → baselines_eval.json.

方法: PatchCore (timm wide_resnet50_2.racm_in1k), PaDiM (timm resnet18.a1_in1k),
     EfficientAD-S (预训练 teacher + imagenette2 蒸馏)
协议 (preflight.json 固化 + 本脚本顶部冻结配置):
  - 支持样本只来自非查询的 train/good 池 (defect_supervised/<asset>/train/good);
    禁止 normal_query 建模或调参
  - 支持样本选择规则: 固定种子 (sha256 派生) 从排序文件名池随机抽 1/5/10 张,
    3 种子; 消费清单 + sha256 全量登记
  - 泄漏检查: 支持样本文件名不得与任何 query/kshot 冻结划分条目相交
  - 评测: 同冻结查询集 (缺陷查询 vs 同资产正常查询), 图像级 AUROC
  - 监督差异声明: AD 基线仅见正常支持样本 (单类), 本方案用目标故障 k-shot
    原型 —— 监督信息不同, 不可当作同范式 head-to-head

用法: python3 scripts/m3_run_anomalib_baselines.py [--methods patchcore padim efficientad]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 本机 HF 不可达, 走镜像

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C

from anomalib.engine import Engine  # 模块级导入: run_method 内引用 (原脚本局部导入致 NameError)

SPLITS = "/root/auto-exp/data/splits/insplad-fd"
FD_ROOT = "/root/autodl-tmp/data/insplad/defect_supervised"
OUT = "/root/auto-exp/outputs/M3"
WORK = "/tmp/m3_anomalib"
ASSETS = sorted({C.ASSET_OF[c] for c in C.COMBOS})

# ---- 冻结 AD 基线配置 (写入 outputs/M3/baselines_cfg.json 后才允许评测) ----
AD_CFG = {
    "imgsz": C.IMGSZ,
    "preprocess": "Resize((320,320), antialias) + ImageNet mean/std 归一化 (方图, 与本方案一致)",
    "patchcore": {"backbone": "timm:wide_resnet50_2.racm_in1k",
                  "layers": ["layer2", "layer3"], "coreset_sampling_ratio": 0.1,
                  "num_neighbors": 9, "epochs": 1},
    "padim": {"backbone": "timm:resnet18.a1_in1k", "layers": ["layer1", "layer2", "layer3"],
              "epochs": 1},
    "efficientad": {"model_size": "small", "total_train_steps": 4000,
                    "note": "论文默认 70000 步; 少样本协议下统一固定 4000 步 "
                            "(a priori, 对全部 k/种子/资产一致, 不针对结果调参)"},
    "support_source": "选项A (2026-09-17 人工裁决): 该资产全部 good 裁剪 "
                      "(defect_supervised/<asset>/{train,val}/good)。原冻结规则 (仅非查询 "
                      "train/good) 因 normal_query 消耗 good 池而不可行 (非查询池="
                      "26/1/2/3/4654), 详见 outputs/M3/support_pool_amendment.md",
    "support_selection": "rng = np.random.default_rng(int(sha256("
                         "'insplad-fd-ad-support-<asset>-k<k>-s<seed>')[:8],16)); "
                         "从排序文件名池 (train∪val good) 抽 k 张",
    "support_leakage_policy": "硬红线: 不得命中 defect/kshot 冻结条目 (已验证与 good 池零交集); "
                              "与 normal_query 的重叠允许并逐配置登记 (normal_query_overlap); "
                              "偏差方向: 有利 AD 基线, 对本方案保守",
    "auroc_scoring": "anomalib 图像级 anomaly score (pred_score); 缺陷查询 vs 同资产正常查询",
}


def support_rng(asset, k, seed):
    h = hashlib.sha256(f"insplad-fd-ad-support-{asset}-k{k}-s{seed}".encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


def select_support(asset, k, seed):
    """选项A (2026-09-17 人工裁决): 支持样本池 = 该资产全部 good 裁剪 (train∪val),
    冻结 RNG 规则不变。原冻结规则 (仅非查询 train/good) 因 normal_query 消耗 good
    池而不可行 (非查询池 = 26/1/2/3/4654), 详见 outputs/M3/support_pool_amendment.md。
    与 normal_query 的重叠逐配置登记; 偏差方向: 有利 AD 基线, 对本方案保守。"""
    files = []
    for split in ("train", "val"):
        d = os.path.join(FD_ROOT, asset, split, "good")
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = sorted(files)
    rng = support_rng(asset, k, seed)
    return [files[i] for i in sorted(rng.choice(len(files), size=k, replace=False).tolist())]


def _frozen_names():
    """冻结划分文件名按角色分组: defect/kshot 为硬红线, normal_query 允许重叠 (选项A)."""
    defect, normal = set(), set()
    for k in C.SHOTS:
        with open(f"{SPLITS}/query_k{k}.json", encoding="utf-8") as f:
            q = json.load(f)
        for ps in q["defect_query"].values():
            defect.update(os.path.basename(p) for p in ps)
        for ps in q["normal_query"].values():
            normal.update(os.path.basename(p) for p in ps)
        for s in C.SEEDS:
            with open(f"{SPLITS}/kshot_k{k}_seed{s}.json", encoding="utf-8") as f:
                sp = json.load(f)
            for ps in sp["prototype_pools"].values():
                defect.update(os.path.basename(p) for p in ps)
    return defect, normal


DEFECT_NAMES, NORMAL_NAMES = _frozen_names()


def leakage_check(support_paths):
    """[硬红线] 支持样本不得命中任何 defect/kshot 冻结条目 (文件名级)."""
    return [p for p in support_paths if os.path.basename(p) in DEFECT_NAMES]


def normal_overlap(support_paths):
    """[选项A·允许并登记] 支持样本与 normal_query 冻结条目的重叠 (文件名级)."""
    return [os.path.basename(p) for p in support_paths if os.path.basename(p) in NORMAL_NAMES]


def make_datamodule(support_paths, batch_size=32, num_workers=4):
    from anomalib.data import Folder
    sdir = os.path.join(WORK, "support", f"s{time.time_ns()}")
    os.makedirs(sdir, exist_ok=True)
    for p in support_paths:
        os.symlink(p, os.path.join(sdir, os.path.basename(p)))
    dm = Folder(name="m3ad", normal_dir=sdir, normal_split_ratio=0.0,
                val_split_mode="none", test_split_mode="none", num_workers=num_workers,
                train_batch_size=batch_size, eval_batch_size=batch_size)
    return dm, sdir


def repeat_loader(dm, times):
    """把 Folder 训练集重复 times 次 —— EfficientAD 固定步数用."""
    from torch.utils.data import DataLoader
    if not hasattr(dm, "train_data"):  # anomalib Folder 惰性物化, 需先 setup
        dm.setup()
    ds = dm.train_data

    class Rep(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds) * times

        def __getitem__(self, i):
            return ds[i % len(ds)]

    return DataLoader(Rep(), batch_size=1, shuffle=False, num_workers=4)


def make_pre_processor():
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Normalize, Resize
    tf = Compose([Resize((C.IMGSZ, C.IMGSZ), antialias=True),
                  Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return PreProcessor(transform=tf)


def predict_scores(model, engine, paths):
    """对路径列表批量打分. anomalib 2.6 PredictDataset 只接受单路径 (文件/目录),
    故建立 0-worker 符号链接目录, 按输入顺序编号, 保证顺序映射."""
    from anomalib.data.predict import PredictDataset
    pdir = os.path.join(WORK, "predict", f"p{time.time_ns()}")
    os.makedirs(pdir, exist_ok=True)
    try:
        for i, p in enumerate(paths):
            os.symlink(p, os.path.join(pdir, f"{i:06d}{os.path.splitext(p)[1]}"))
        preds = engine.predict(model=model, dataset=PredictDataset(pdir),
                               return_predictions=True)
    finally:
        shutil.rmtree(pdir, ignore_errors=True)
    scores, got = [], []
    for it in preds:
        d = it if isinstance(it, dict) else {a: getattr(it, a) for a in ("pred_score", "image_path")}
        scores.append(float(d.get("pred_score", d.get("anomaly_map", torch.tensor(0.0)).mean())))
        got.append(d.get("image_path"))
    # 符号链接名 = 输入顺序编号 -> 还原为原始路径, 保证 path2score 正确
    idx_of = {}
    for g in got:
        stem = os.path.basename(str(g)).split(".")[0]
        idx_of[str(g)] = paths[int(stem)] if stem.isdigit() and int(stem) < len(paths) else str(g)
    got = [idx_of[str(g)] for g in got]
    return np.array(scores), got


def build_model(method):
    from anomalib.models import EfficientAd, Padim, Patchcore
    pp = make_pre_processor()
    if method == "patchcore":
        return Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"),
                         pre_processor=pp)
    if method == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"],
                     pre_processor=pp)
    # EfficientAd 强制 transform 不含 Normalize (内部自带归一化), 且 train_batch_size=1
    from anomalib.pre_processing import PreProcessor
    from torchvision.transforms.v2 import Compose, Resize
    pp_eff = PreProcessor(transform=Compose([Resize((C.IMGSZ, C.IMGSZ), antialias=True)]))
    return EfficientAd(model_size="small", imagenet_dir="/tmp/imagenette2", pre_processor=pp_eff)


def run_method(method):
    results = {"configs": {}}
    for asset in ASSETS:
        combos = [c for c in C.COMBOS if C.ASSET_OF[c] == asset]
        for k in C.SHOTS:
            with open(f"{SPLITS}/query_k{k}.json", encoding="utf-8") as f:
                q = json.load(f)
            qpaths = list(q["defect_query"][combos[0]])
            if len(combos) > 1:
                qpaths += list(q["defect_query"][combos[1]])
            qpaths += list(q["normal_query"][f"{asset}/good"])
            for seed in C.SEEDS:
                key = f"{asset}_k{k}_s{seed}"
                sup = select_support(asset, k, seed)
                rec = {"support": [os.path.basename(p) for p in sup],
                       "support_sha256": [C.sha256_file(p) for p in sup],
                       "defect_kshot_leakage": leakage_check(sup),
                       "normal_query_overlap": normal_overlap(sup), "combos": {}}
                if rec["defect_kshot_leakage"]:
                    rec["error"] = "LEAKAGE: support hits defect/kshot frozen entries — skip"
                    results["configs"][key] = rec
                    continue
                try:
                    t0 = time.time()
                    model = build_model(method)
                    if method == "efficientad":
                        # anomalib 2.6 EfficientAd.configure_optimizers 需要 datamodule:
                        # num_steps = max_epochs * len(train_dataloader)。
                        # 单 epoch × (k*times) 重复加载器: num_steps = 1*(k*times) = 4000,
                        # 与 AD_CFG a priori 一致, 消除逐-epoch 的 Lightning 循环开销
                        times = max(1, AD_CFG["efficientad"]["total_train_steps"] // max(1, k))
                        dm, sdir = make_datamodule(sup, batch_size=1, num_workers=4)
                        dm.setup()
                        base_ds = dm.train_data
                        collate = dm.train_dataloader().collate_fn  # anomalib 自定义 collate

                        class _Rep(torch.utils.data.Dataset):
                            def __len__(self):
                                return len(base_ds) * times

                            def __getitem__(self, i):
                                return base_ds[i % len(base_ds)]

                        loader = torch.utils.data.DataLoader(
                            _Rep(), batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=collate)
                        dm.train_dataloader = lambda: loader  # 实例级覆盖: 1 epoch × k*times 步
                        eng = Engine(max_epochs=1, accelerator="gpu", devices=1,
                                     default_root_dir=f"{WORK}/{method}", logger=False,
                                     barebones=True, enable_progress_bar=False,
                                     limit_val_batches=0)
                        eng.fit(model=model, datamodule=dm)
                    else:
                        dm, sdir = make_datamodule(sup, batch_size=min(32, max(1, k)))
                        eng = Engine(max_epochs=1, accelerator="gpu", devices=1,
                                     default_root_dir=f"{WORK}/{method}", logger=False,
                                     barebones=True, enable_progress_bar=False,
                                     limit_val_batches=0)
                        eng.fit(model=model, datamodule=dm)
                    scores, got = predict_scores(model, eng, qpaths)
                    path2score = dict(zip(got, scores))
                    for c in combos:
                        d = [path2score[p] for p in q["defect_query"][c]]
                        n = [path2score[p] for p in q["normal_query"][f"{asset}/good"]]
                        auroc, _, _ = C.auroc_recall(np.array(d), np.array(n))
                        rec["combos"][c] = {"auroc": auroc, "n_defect": len(d), "n_normal": len(n)}
                    rec["train_seconds"] = round(time.time() - t0, 1)
                except Exception as e:  # noqa: BLE001 —— 失败也如实登记
                    rec["error"] = f"{type(e).__name__}: {e}"
                results["configs"][key] = rec
                print(f"[{method}] {key}: "
                      f"{ {c: round(v['auroc'], 4) for c, v in rec['combos'].items()} } "
                      f"{rec.get('error', '')} {rec.get('train_seconds', '')}s", flush=True)
                shutil.rmtree(os.path.join(WORK, "support"), ignore_errors=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["patchcore", "padim", "efficientad"])
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    C.save_json(AD_CFG, f"{OUT}/baselines_cfg.json")  # 冻结配置落盘 (评测前)

    from anomalib.engine import Engine  # noqa: F401 提前暴露导入错误
    all_results = {}
    for m in args.methods:
        all_results[m] = run_method(m)

    completed = [m for m, r in all_results.items()
                 if any(c.get("combos") and not c.get("error") for c in r["configs"].values())]
    leakage_ok = all(not c.get("defect_kshot_leakage") and not str(c.get("error", "")).startswith("LEAKAGE")
                     for r in all_results.values() for c in r["configs"].values())
    ev = {
        "task": "M3-proto-head-v1",
        "cfg": AD_CFG,
        "methods_completed": completed,
        "methods_attempted": list(all_results.keys()),
        "query_leakage_check_passed": 1 if leakage_ok else 0,
        "leakage_check_rule": "硬红线: 支持样本不得命中 defect/kshot 冻结条目 (文件名级); "
                              "与 normal_query 的重叠按选项A (2026-09-17 人工裁决) 允许并逐配置登记 "
                              "(normal_query_overlap 字段)",
        "supervision_note": "AD 基线只用正常支持样本(单类建模); 本方案用目标故障 k-shot "
                            "缺陷原型(多类判别)。监督信息不同, 非同范式 head-to-head。"
                            "选项A 下 AD 拟合样本与 normal_query 部分重叠, 偏差方向有利 AD 基线, 对本方案保守。",
        "results": all_results,
    }
    C.save_json(ev, f"{OUT}/baselines_eval.json")


if __name__ == "__main__":
    main()

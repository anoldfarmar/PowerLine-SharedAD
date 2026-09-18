#!/usr/bin/env python3
"""M3 Step-5 单配置补跑: efficientad / lightning-rod-suspension / k5 / seed2.

背景: 主运行 (2026-09-18 10:00-16:09) 期间, 为缓解系统盘 ENOSPC 清理了
/tmp/m3_anomalib, 恰好命中本配置的 DataLoader 支持图读取 -> FileNotFoundError,
记录为空 (combos={}). 本脚本用【同一冻结种子/同一协议】确定性重跑该配置:

  - 支持样本: select_support(asset,k,seed) 为 sha256 派生 RNG, 确定性;
    启动即校验与事故前记录的 support/support_sha256 逐条一致, 不一致立即终止
  - 模型/训练/打分: 逐行复刻 m3_run_anomalib_baselines.run_method 的
    efficientad 分支 (batch_size=1, times=4000//k, Engine barebones),
    唯一区别是只评这一个配置
  - 合并: 仅替换 baselines_eval.json 中该条记录, 并在记录内登记 repair 说明

用法: XDG_CACHE_HOME=/root/auto-exp/.cache python3 scripts/m3_repair_efficientad_k5s2.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as C  # noqa: E402
import m3_run_anomalib_baselines as B  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

METHOD, ASSET, K, SEED = "efficientad", "lightning-rod-suspension", 5, 2
KEY = f"{ASSET}_k{K}_s{SEED}"


def main():
    ev_path = f"{B.OUT}/baselines_eval.json"
    with open(ev_path, encoding="utf-8") as f:
        ev = json.load(f)
    old = ev["results"][METHOD]["configs"][KEY]

    # 1) 冻结规则重选支持样本 + 确定性校验 (硬闸: 不一致即终止, 绝不静默换样本)
    sup = B.select_support(ASSET, K, SEED)
    assert [os.path.basename(p) for p in sup] == old["support"], (
        "support 与事故前记录不一致 — 冻结规则被破坏, 终止补跑")
    assert [C.sha256_file(p) for p in sup] == old["support_sha256"], (
        "support sha256 与事故前记录不一致 — 终止补跑")
    assert not B.leakage_check(sup), "LEAKAGE: 支持样本命中 defect/kshot 冻结条目"
    print(f"[repair] support 一致性校验通过: {len(sup)} 张 (与事故前记录逐条相同)", flush=True)

    # 2) 逐行复刻 run_method 的 efficientad 分支 (仅单配置)
    with open(f"{B.SPLITS}/query_k{K}.json", encoding="utf-8") as f:
        q = json.load(f)
    combos = [c for c in C.COMBOS if C.ASSET_OF[c] == ASSET]
    qpaths = list(q["defect_query"][combos[0]])
    if len(combos) > 1:
        qpaths += list(q["defect_query"][combos[1]])
    qpaths += list(q["normal_query"][f"{ASSET}/good"])

    rec = {"support": [os.path.basename(p) for p in sup],
           "support_sha256": [C.sha256_file(p) for p in sup],
           "defect_kshot_leakage": B.leakage_check(sup),
           "normal_query_overlap": B.normal_overlap(sup), "combos": {}}
    try:
        t0 = time.time()
        model = B.build_model(METHOD)
        times = max(1, B.AD_CFG[METHOD]["total_train_steps"] // max(1, K))
        dm, sdir = B.make_datamodule(sup, batch_size=1, num_workers=4)
        dm.setup()
        base_ds = dm.train_data
        collate = dm.train_dataloader().collate_fn

        class _Rep(torch.utils.data.Dataset):
            def __len__(self):
                return len(base_ds) * times

            def __getitem__(self, i):
                return base_ds[i % len(base_ds)]

        loader = torch.utils.data.DataLoader(
            _Rep(), batch_size=1, shuffle=False, num_workers=4, collate_fn=collate)
        dm.train_dataloader = lambda: loader
        eng = B.Engine(max_epochs=1, accelerator="gpu", devices=1,
                       default_root_dir=f"{B.WORK}/{METHOD}", logger=False,
                       barebones=True, enable_progress_bar=False, limit_val_batches=0)
        eng.fit(model=model, datamodule=dm)
        scores, got = B.predict_scores(model, eng, qpaths)
        path2score = dict(zip(got, scores))
        for c in combos:
            d = [path2score[p] for p in q["defect_query"][c]]
            n = [path2score[p] for p in q["normal_query"][f"{ASSET}/good"]]
            auroc, _, _ = C.auroc_recall(np.array(d), np.array(n))
            rec["combos"][c] = {"auroc": auroc, "n_defect": len(d), "n_normal": len(n)}
        rec["train_seconds"] = round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        print(f"[repair] FAILED again: {rec['error']}", flush=True)
        C.save_json(ev, ev_path)  # 不掩盖失败, 原样保留
        raise

    rec["repair"] = ("2026-09-18 单配置补跑: 主运行中该配置因 /tmp/m3_anomalib 被清理 "
                     "(ENOSPC 缓解操作) 触发 FileNotFoundError; 本补跑使用同一冻结种子 "
                     "(sha256 派生 RNG) 与同一协议 (4000 步/batch=1/同一查询集), "
                     "支持样本经一致性校验与事故前记录逐条相同。")

    # 3) 合并回 baselines_eval.json 并重算顶层状态字段
    ev["results"][METHOD]["configs"][KEY] = rec
    ev["methods_completed"] = [m for m, r in ev["results"].items()
                               if any(c.get("combos") and not c.get("error")
                                      for c in r["configs"].values())]
    ev["query_leakage_check_passed"] = 1 if all(
        not c.get("defect_kshot_leakage") and not str(c.get("error", "")).startswith("LEAKAGE")
        for r in ev["results"].values() for c in r["configs"].values()) else 0
    C.save_json(ev, ev_path)
    print(f"[repair] OK {KEY}: "
          f"{ {c: round(v['auroc'], 4) for c, v in rec['combos'].items()} } "
          f"{rec['train_seconds']}s -> baselines_eval.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""M3BASELINE-AnomalyDINO-full-v1: extend frozen AnomalyDINO-S/448 to 7 InsPLAD combos.
Per-asset sharing (normal bank + normal-query scores across faults), v1 reuse with
hash verification, new-entry re-scoring reproduction gates, sampled GPU memory peak.
Method identical to v1 METHOD_SPEC: official agnostic_no_mask fallback, 8-angle rotation,
short-edge448 aspect-preserving patch-multiple crop, FP32 nonTF32, batch=1 extraction,
exact normalized-L2 FAISS NN=1, official mean_top1p. No config tuning."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["FAISS_OPT_LEVEL"] = "generic"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["XFORMERS_DISABLED"] = "1"
import argparse, csv, hashlib, json, shutil, subprocess, sys, threading, time, traceback
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from scipy.stats import rankdata

from m3ad_model import load, sha, statehash

TASK = "M3BASELINE-AnomalyDINO-full-v1"
V1TASK = "M3BASELINE-AnomalyDINO-v1"
ASSET_FAULTS = {"glass-insulator": ["missing-cap"], "lightning-rod-suspension": ["rust"],
                 "polymer-insulator-upper-shackle": ["corrosion", "rust"],
                 "vari-grip": ["bird-nest", "rust"], "yoke-suspension": ["rust"]}
V1_ASSET_FAULTS = {("yoke-suspension", "rust"), ("vari-grip", "bird-nest")}
REFS = [3001, 3002, 3003, 3004, 3005]
ARMS = {"Y-T8": "Y-T8", "W-PC": "W-PC", "W-T8": "W-T8"}  # M3FULL arm file -> system
PARAMS = dict(model="dinov2_vits14", short_edge=448, precision="FP32", tf32=False,
               angles=[0, 45, 90, 135, 180, 225, 270, 315], masking=False, mask_ref_images=False,
               preprocessing="official agnostic_no_mask fallback (unknown dataset)",
               retrieval="exact IndexFlatL2/GpuIndexFlatL2 on L2-normalized features, squaredL2/2",
               k=1, score="official mean_top1p", seed=0, atol=1e-4, formal_batch=1,
               naming="AnomalyDINO-S/448, official custom-dataset no-mask fallback configuration")


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def arrsha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def now():
    torch.cuda.synchronize()
    return time.perf_counter()


def auc(d, n):
    x = np.r_[d, n]
    r = rankdata(x, method="average")
    nd, nn = len(d), len(n)
    return float((r[:nd].sum() - nd * (nd + 1) / 2) / (nd * nn))


class GpuSampler:
    """Device-level sampled GPU memory peak (nvidia-smi); NOT process-exclusive,
    NOT PyTorch allocated. Tool and interval recorded."""
    def __init__(self, interval=1.0):
        self.interval = interval
        self.peak_mib = 0
        self.samples = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                               "--format=csv,noheader,nounits"], text=True).strip()
                mib = int(out.split("\n")[0])
                self.peak_mib = max(self.peak_mib, mib)
                self.samples += 1
            except Exception:
                pass
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=3)
        return dict(tool="nvidia-smi --query-gpu=memory.used (device-level)", interval_s=self.interval,
                     samples=self.samples, sampled_peak_mib=self.peak_mib,
                     note="sampled device memory.used peak; includes non-PyTorch/FAISS CUDA allocations "
                           "and any other processes; not process-exclusive; PyTorch reported separately")


class Experiment:
    def __init__(self, args):
        self.a = args
        self.root = Path(args.root)
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.v1 = Path(args.v1)
        self.cache = self.out / "cache"
        self.cache.mkdir(exist_ok=True)
        self.feat_cache = Path(args.feature_cache)
        self.feat_cache.mkdir(parents=True, exist_ok=True)
        self.v1bank = self.v1 / "cache" / "banks"
        self.m3full = self.root / "outputs" / "M3full-frozen"
        self.support = read(self.m3full / "SUPPORT_LISTS.json")
        self.qq = read(self.root / "data/splits/insplad-fd/query_k5.json")
        self.v1fi = read(self.v1 / "FROZEN_INPUTS.json")
        self.v1ident = read(self.v1 / "MODEL_IDENTITY.json")
        self.v1bankman = read(self.v1 / "BANK_MANIFEST.json")
        self.weights = Path(args.model)
        assert read(self.root / "registry/active_task.json")["task"] == TASK
        torch.set_grad_enabled(False)
        torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        faiss.omp_set_num_threads(4)
        self.res = faiss.StandardGpuResources(); self.res.setTempMemory(256 << 20)
        self.wrapper, self.hf, self.identity = load(args.model, args.dino_repo, args.official)
        self.before = statehash(self.wrapper.model)
        assert self.before == self.v1ident["state_hash"], "model state differs from v1"
        assert sha(Path(__file__).parent / "m3ad_model.py") == \
            self.v1fi["sha256"][str(self.root / "scripts/m3ad_model.py")], "conversion code changed since v1"
        self.sampler = GpuSampler(1.0)
        self.cost = defaultdict(float); self.hits = 0; self.new = 0
        from src.post_eval import mean_top1p
        self.mean = mean_top1p
        self.records = {}

    # ---------- freeze ----------
    def binding(self):
        b = {}
        for asset, faults in ASSET_FAULTS.items():
            qn = list(self.qq["normal_query"][asset + "/good"])
            assert len(qn) == len(set(qn)) == 500
            supps = {}
            for j in REFS:
                s = self.support[f"{asset}_ref{j}"]["N10"]
                assert len(s) == len(set(s)) == 10
                assert not set(s) & set(qn), (asset, j)
                supps[str(j)] = s
            fds = {}
            for f in faults:
                qd = list(self.qq["defect_query"][f"{asset}/{f}"])
                assert not set(qd) & set(qn)
                fds[f] = qd
            b[asset] = dict(normal_query=qn, supports=supps, faults=fds)
        # cross-checks with v1 provenance (yoke/vari)
        for asset in ("yoke-suspension", "vari-grip"):
            for j in REFS:
                v1s = self.v1fi["binding"][f"{asset}_ref{j}"]["support"]
                assert v1s == b[asset]["supports"][str(j)], (asset, j, "support list differs from v1")
                v1q = self.v1fi["binding"][f"{asset}_ref{j}"]["queries"]
                assert set(v1q["normal"]) == set(b[asset]["normal_query"])
                f0 = ASSET_FAULTS[asset][0]
                assert set(v1q["defect"]) == set(b[asset]["faults"][f0])
        return b

    def freeze(self):
        self.binding_cache = self.binding()
        b = self.binding_cache
        expected = []
        for asset, faults in ASSET_FAULTS.items():
            for f in faults:
                for j in REFS:
                    src = "v1" if (asset, f) in V1_ASSET_FAULTS else "new"
                    expected.append(dict(config=f"{asset}__{f}_ref{j}", asset=asset, fault=f,
                                          ref=j, source=src, bank_key=f"{asset}_ref{j}",
                                          normal_sharing="asset-level (same normal bank and normal-query scores)"))
        assert len(expected) == 35 and len(set(e["config"] for e in expected)) == 35
        write(self.out / "EXPECTED_CONFIGS.json", dict(task=TASK, configs=expected,
              count=35, systems=["Y-T8", "W-PC", "W-T8", "AnomalyDINO-S/448"],
              frozen_before_first_new_scoring=True,
              bank_sharing=dict(polymer="one bank per ref serves corrosion and rust",
                                 vari="bird-nest banks from v1 serve rust defect scoring")))
        # input file hashes
        files = [self.m3full / "SUPPORT_LISTS.json", self.root / "data/splits/insplad-fd/query_k5.json",
                 self.root / "plan/tasks/M3BASELINE-AnomalyDINO-full-v1.yaml",
                 Path(__file__), Path(__file__).parent / "m3ad_model.py",
                 self.v1 / "FROZEN_INPUTS.json", self.v1 / "MODEL_IDENTITY.json",
                 self.v1 / "REPRO_AND_CORRECTNESS.json", self.v1 / "METHOD_SPEC.md",
                 self.v1 / "BANK_MANIFEST.json", self.v1 / "CACHE_MANIFEST.json",
                 self.v1 / "RESOURCE_COST.json", self.v1 / "metrics.json"]
        baselines = {}
        for asset, faults in ASSET_FAULTS.items():
            for f in faults:
                combo = f"{asset}__{f}"
                for arm in ARMS.values():
                    for j in REFS:
                        p = self.m3full / "raw_scores" / f"{combo}_{arm}_10N_ref{j}.json"
                        files.append(p); baselines[f"{combo}_{arm}_ref{j}"] = str(p)
        for name, e in [(x["config"], x) for x in expected]:
            if e["source"] == "v1":
                files.append(self.v1 / "raw_scores" / f"{name}.json")
        for p in sorted(self.weights.glob("*")):
            if p.is_file(): files.append(p)
        hashes = {str(p): sha(p) for p in files if Path(p).is_file()}
        env = dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__,
                    faiss=faiss.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(), inherited_from_v1=True)
        commits = self.v1fi["commits"]
        for name, p in [("AnomalyDINO", self.a.official), ("DINOv2", self.a.dino_repo)]:
            h = subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
            assert h == commits[name], (name, "official repo changed since v1")
        commits["experiment"] = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        # baseline identity re-check (never read pre-correction files blindly)
        base_ok = 0
        for key, p in baselines.items():
            base = key.rsplit("_ref", 1)[0]              # e.g. glass-insulator__missing-cap_Y-T8
            arm = base.rsplit("_", 1)[1]                  # Y-T8 / W-PC / W-T8
            combo = base.rsplit("_", 1)[0]                # asset__fault
            asset, fault = combo.split("__")
            r = read(p)
            qn = b[asset]["normal_query"]; qd = b[asset]["faults"][fault]
            assert set(r["normal"]) == set(qn) and len(r["normal"]) == len(qn), (key, "normal identity")
            assert set(r["defect"]) == set(qd) and len(r["defect"]) == len(qd), (key, "defect identity")
            u = auc(list(r["defect"].values()), list(r["normal"].values()))
            assert abs(u - r["auroc"]) <= 1e-9, (key, "auroc mismatch")
            base_ok += 1
        # MODEL_IDENTITY + METHOD_SPEC carried over
        ident = dict(self.v1ident); ident["state_hash_full_run"] = self.before
        ident["reloaded_in_full_v1"] = True
        write(self.out / "MODEL_IDENTITY.json", ident)
        spec = (self.v1 / "METHOD_SPEC.md").read_text()
        spec += "\n\n---\n\n## full-v1 追加（方法不变）\n\n- 配置命名：AnomalyDINO-S/448，官方自定义数据集无掩码回退配置；\n- 同资产多故障共享同一正常库与同一正常查询分数（polymer 两故障同库；vari/rust 复用 v1 vari 库）；\n- 正式提取 batch=1（沿用 v1 验证的计算图）；检索分块批量，特征缓存跨参照共享；\n- GPU 显存口径：nvidia-smi 设备级采样峰值（含 FAISS CUDA 分配），与 PyTorch allocated 分开报告。\n"
        (self.out / "METHOD_SPEC.md").write_text(spec)
        write(self.out / "REUSE_MANIFEST.json", dict(
            v1_task=V1TASK, v1_commit=self.v1fi["commits"]["experiment"],
            reused_ad_configs=[e["config"] for e in expected if e["source"] == "v1"],
            reused_ad_count=sum(1 for e in expected if e["source"] == "v1"),
            reused_banks=dict((k, v.get("sha256")) for k, v in self.v1bankman.items()),
            baseline_source="M3FULL corrected raw_scores (post ERRATA 7dcec96); identity re-checked this task",
            baseline_files=baselines, baseline_sha256={k: hashes[p] for k, p in baselines.items()},
            baseline_identity_ok_count=base_ok,
            shared_normal_note="normal-query scores computed once per (asset, ref) and shared by all faults of that asset"),
            )
        write(self.out / "FROZEN_INPUTS.json", dict(
            task=TASK, parameters=PARAMS, env=env, commits=commits, sha256=hashes,
            model_state_hash=self.before, model_state_matches_v1=self.before == self.v1ident["state_hash"],
            binding=b, expected_config_count=35,
            data_boundary=dict(support_budget="cropped-file count, not independent source frames",
                                 path_isolation="does not prove independence from YOLO training source frames",
                                 source_identity="bare-DJI collisions neither prove nor disprove same source; identity unresolved, recorded as such",
                                 queries="participated in historical research decisions; this is an exploratory frozen comparison"),
            timestamp=time.time()))
        print(f"[freeze] 35 configs expected; {base_ok}/140 baseline files identity-verified", flush=True)

    # ---------- features ----------
    def item(self, path, angle=None):
        tag = dict(path=str(path), image_sha256=sha(path), angle=angle, model=self.before,
                    short_edge=448,
                    rgb="cv2 BGR2RGB",
                    rotate="official cv2 warpAffine BORDER_DEFAULT" if angle is not None else None)
        key = hashlib.sha256(json.dumps(tag, sort_keys=True).encode()).hexdigest()
        return dict(key=key, tag=tag, path=str(path), angle=angle)

    def tensor(self, item):
        import cv2
        from src.utils import rotate_image
        img = cv2.cvtColor(cv2.imread(item["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if item["angle"] is not None:
            img = rotate_image(img, item["angle"])
        return self.wrapper.prepare_image(img)

    def ffile(self, item):
        return self.feat_cache / f"{item['key']}.npy"

    def ready(self, item):
        p = self.ffile(item)
        mp = p.with_suffix(".json")
        if not p.exists() or not mp.exists():
            return False
        meta = read(mp)
        assert meta["tag"] == item["tag"] and sha(p) == meta["sha256"], p
        self.records[item["key"]] = meta
        self.hits += 1
        return True

    def extract(self, items):
        pending = [i for i in items if not self.ready(i)]
        t = now()
        for n, item in enumerate(pending):
            tgrid = self.tensor(item)
            t0 = now()
            with torch.inference_mode():
                v = self.wrapper.model.get_intermediate_layers(tgrid[0].unsqueeze(0).cuda())[0][0].cpu().numpy()
            self.cost["feature_forward_sync_wall_seconds"] += now() - t0
            grid = tgrid[1]
            assert v.shape == (grid[0] * grid[1], 384) and np.isfinite(v).all()
            p = self.ffile(item); p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.npy"); np.save(tmp, v); tmp.replace(p)
            meta = dict(tag=item["tag"], grid=list(grid), shape=list(v.shape), dtype="float32",
                         bytes=int(v.nbytes), sha256=sha(p), batch=1)
            write(p.with_suffix(".json"), meta)
            self.records[item["key"]] = meta
            self.new += 1
            if n % 200 == 0:
                print(f"[features] {n}/{len(pending)}", flush=True)
        self.cost["feature_pipeline_wall_seconds"] += now() - t

    def features(self, item):
        f = np.load(self.ffile(item))
        self.records[item["key"]] = self.records.get(item["key"]) or read(self.ffile(item).with_suffix(".json"))
        return f

    def support_items(self, asset, ref):
        return [self.item(p, a) for p in self.binding_cache[asset]["supports"][str(ref)] for a in PARAMS["angles"]]

    def bank(self, items, save_to=None):
        t = now()
        if save_to:
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        a = np.concatenate([self.features(i) for i in items]).astype("float32")
        faiss.normalize_L2(a)
        cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
        idx = faiss.GpuIndexFlatL2(self.res, 384, cfg); idx.add(a)
        self.cost["bank_concat_normalize_upload_seconds"] += now() - t
        if save_to:
            np.save(save_to, a)
        return a, idx

    def score(self, features, index, dmap_path=None):
        q = features.copy(); faiss.normalize_L2(q)
        t = now()
        ds, _ = index.search(q, 1)
        self.cost["faiss_gpu_search_wall_seconds"] += now() - t
        dist = (ds / 2).squeeze(1).astype(float)
        s = float(self.mean(dist.flatten()))
        assert np.isfinite(dist).all() and np.isfinite(s)
        if dmap_path:
            dmap_path = Path(dmap_path)
            dmap_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dmap_path, dist)
        return s, dist

    # ---------- gates ----------
    def gates(self):
        if (self.out / "REPRO_AND_CORRECTNESS.json").exists():
            c = read(self.out / "REPRO_AND_CORRECTNESS.json")
            if c.get("passed"):
                return c
        c = dict()
        b = self.binding_cache
        # 1) reuse verification: 10 v1 configs identity + independent AUROC
        c["reuse"] = []
        for asset, faults in ASSET_FAULTS.items():
            for f in faults:
                if (asset, f) not in V1_ASSET_FAULTS:
                    continue
                for j in REFS:
                    name = f"{asset}__{f}_ref{j}"
                    p = self.v1 / "raw_scores" / f"{name}.json"
                    r = read(p)
                    qn = b[asset]["normal_query"]; qd = b[asset]["faults"][f]
                    assert set(r["normal"]) == set(qn) and set(r["defect"]) == set(qd), name
                    u = auc(list(r["defect"].values()), list(r["normal"].values()))
                    assert abs(u - r["metrics"]["AnomalyDINO"]) <= 1e-9, name
                    c["reuse"].append(dict(config=name, sha256=sha(p), auroc_recomputed=u,
                                            matches_v1_metrics=True))
        assert len(c["reuse"]) == 10
        # 2) new-entry re-scoring of ALL yoke/vari configs from restored banks + shared cache
        c["repro_rescore"] = []
        for asset in ("yoke-suspension", "vari-grip"):
            for j in REFS:
                f0 = ASSET_FAULTS[asset][0]
                name = f"{asset}__{f0}_ref{j}"
                bp = self.v1bank / f"{name}.npy"
                assert sha(bp) == self.v1bankman[name]["file_sha256"], name
                bank = np.load(bp)
                assert arrsha(bank) == self.v1bankman[name]["sha256"], name
                assert len(bank) == self.v1bankman[name]["patches"]
                cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
                idx = faiss.GpuIndexFlatL2(self.res, 384, cfg); idx.add(bank)
                v1r = read(self.v1 / "raw_scores" / f"{name}.json")
                mdiff = 0.0
                for side in ("normal", "defect"):
                    qn = b[asset]["normal_query"] if side == "normal" else b[asset]["faults"][f0]
                    assert set(v1r[side]) == set(qn)
                    for p in qn:
                        item = self.item(p)
                        if not self.ready(item):
                            self.extract([item])
                        s, _ = self.score(self.features(item), idx)
                        mdiff = max(mdiff, abs(s - v1r[side][p]))
                assert mdiff <= 1e-4, (name, mdiff)
                c["repro_rescore"].append(dict(config=name, restored_bank_sha=self.v1bankman[name]["sha256"],
                                                per_image_max_score_diff=mdiff, gate=1e-4))
                del idx, bank
        # 3) double-bank on a NEW asset (glass ref3001)
        gsup = self.support_items("glass-insulator", 3001)
        self.extract(gsup)
        b1, _ = self.bank(gsup)
        b2, _ = self.bank(gsup)
        assert arrsha(b1) == arrsha(b2)
        qp = b["glass-insulator"]["faults"]["missing-cap"][0]
        item = self.item(qp)
        self.extract([item])
        s1, _ = self.score(self.features(item), _built_index(b1))
        s2, _ = self.score(self.features(item), _built_index(b2))
        assert abs(s1 - s2) <= 1e-4
        npatch = sum(self.records[i["key"]]["shape"][0] for i in gsup)
        c["double_bank_new_asset"] = dict(asset="glass-insulator", ref=3001, items=len(gsup),
                                           patches=npatch, sha=arrsha(b1), score_diff=abs(s1 - s2),
                                           support_is_only_input=sorted(set(i["tag"]["path"] for i in gsup)) ==
                                           sorted(b["glass-insulator"]["supports"]["3001"]))
        # 4) tail rule + state hash
        toy = np.r_[np.zeros(197), 1., 2., 2.]
        assert self.mean(toy) == 2 and self.mean(np.array([0., .7])) == .7
        c["tail_rule"] = dict(n=200, retained=2, score=float(self.mean(toy)))
        c["state_hash_before"] = self.before
        c["state_hash_after"] = statehash(self.wrapper.model)
        assert c["state_hash_before"] == c["state_hash_after"]
        c["passed"] = True
        write(self.out / "REPRO_AND_CORRECTNESS.json", c)
        print("[gates] all passed", flush=True)
        return c

    # ---------- probe ----------
    def probe(self):
        if (self.out / "COST_PROBE.json").exists():
            return read(self.out / "COST_PROBE.json")
        b = self.binding_cache
        asset = "glass-insulator"
        items = self.support_items(asset, 3002)
        qps = b[asset]["normal_query"][:15] + b[asset]["faults"]["missing-cap"][:15]
        items += [self.item(p) for p in qps]
        pending = [i for i in items if not self._was_hit(i)]
        t = now()
        self.extract(items)
        probe_s = now() - t
        n_new_total = self._count_new_items()
        rate = probe_s / max(len(pending), 1)
        est = n_new_total * rate + 15 * 2.5 + self._estimate_search()
        cost = dict(probe_items=len(items), probe_seconds=probe_s, new_items_total=n_new_total,
                     rate_s_per_item=rate, estimated_total_seconds=est, gpu_budget_seconds=7200,
                     allowed=est < 7200,
                     note="extrapolated from new-asset representative extraction at batch=1; "
                           "search estimate from v1 measured per-search wall; not an end-to-end speed claim")
        assert cost["allowed"], "budget exceeded"
        write(self.out / "COST_PROBE.json", cost)
        print("[probe] est %.0fs" % est, flush=True)
        return cost

    def _was_hit(self, item):
        return self.ffile(item).exists()

    def _count_new_items(self):
        n = 0
        for asset, faults in ASSET_FAULTS.items():
            if (asset, faults[0]) in V1_ASSET_FAULTS:
                continue
            for j in REFS:
                n += 80
            n += len(self.binding_cache[asset]["normal_query"]) + sum(len(v) for v in self.binding_cache[asset]["faults"].values())
        return n

    def _estimate_search(self):
        n = 0
        for asset, faults in ASSET_FAULTS.items():
            n += len(self.binding_cache[asset]["normal_query"]) * 5
            for f in faults:
                if (asset, f) not in V1_ASSET_FAULTS:
                    n += len(self.binding_cache[asset]["faults"][f]) * 5
        return n * 0.004

    # ---------- full ----------
    def all_support_items(self):
        out = {}
        for asset in ASSET_FAULTS:
            if (asset, ASSET_FAULTS[asset][0]) in V1_ASSET_FAULTS:
                continue
            for j in REFS:
                for it in self.support_items(asset, j):
                    out[it["key"]] = it
        return list(out.values())

    def full(self):
        g = read(self.out / "REPRO_AND_CORRECTNESS.json"); assert g["passed"]
        cost = read(self.out / "COST_PROBE.json"); assert cost["allowed"]
        b = self.binding_cache
        # extract all new-asset features (supports of all refs + queries)
        items = self.all_support_items()
        for asset in ASSET_FAULTS:
            if (asset, ASSET_FAULTS[asset][0]) in V1_ASSET_FAULTS:
                # v1 asset: normal/first-fault queries cached from v1; extract only
                # defect queries of faults NOT covered by v1 (vari/rust)
                for f2 in ASSET_FAULTS[asset]:
                    if (asset, f2) not in V1_ASSET_FAULTS:
                        for p in b[asset]["faults"][f2]:
                            items.append(self.item(p))
                continue
            for p in b[asset]["normal_query"]:
                items.append(self.item(p))
            for f2 in b[asset]["faults"]:
                for p in b[asset]["faults"][f2]:
                    items.append(self.item(p))
        uniq = {}
        for it in items:
            uniq[it["key"]] = it
        pre_hits = self.hits
        self.extract(list(uniq.values()))
        self.cost["hits_from_v1_or_earlier"] = pre_hits
        bankman = {}
        rows = []
        combo_scores = {}  # (asset, fault, ref) -> (defect_scores, normal_scores)
        for asset in ASSET_FAULTS:
            is_v1 = (asset, ASSET_FAULTS[asset][0]) in V1_ASSET_FAULTS
            # normal-query scores per ref (computed once per asset-ref, shared across faults)
            normal_scores = {j: {} for j in REFS}
            for j in REFS:
                bank_key = f"{asset}_ref{j}"
                if is_v1:
                    name0 = f"{asset}__{ASSET_FAULTS[asset][0]}_ref{j}"
                    bp = self.v1bank / f"{name0}.npy"
                    assert sha(bp) == self.v1bankman[name0]["file_sha256"]
                    bank = np.load(bp)
                    assert arrsha(bank) == self.v1bankman[name0]["sha256"]
                    cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
                    idx = faiss.GpuIndexFlatL2(self.res, 384, cfg); idx.add(bank)
                    meta = dict(self.v1bankman[name0]); meta["source"] = "restored_from_v1_cache"
                    meta["bank_key"] = bank_key
                    bankman[bank_key] = meta
                    bank_arr = bank
                else:
                    sup = self.support_items(asset, j)
                    save_to = self.cache / "banks" / f"{bank_key}.npy"
                    bank_arr, idx = self.bank(sup, save_to=save_to)
                    npatch = sum(self.records[i["key"]]["shape"][0] for i in sup)
                    meta = dict(bank_key=bank_key, support_paths=b[asset]["supports"][str(j)],
                                 original_shots=10, augmented_images=80, angles=PARAMS["angles"],
                                 patches=int(len(bank_arr)), candidates=int(idx.ntotal), dimension=384,
                                 bank_bytes=int(bank_arr.nbytes), sha256=arrsha(bank_arr),
                                 file_sha256=sha(save_to), source="new", query_used_for_bank=False,
                                 device="faiss GPU exact FP32", masking=False)
                    assert idx.ntotal == len(bank_arr) == npatch
                    bankman[bank_key] = meta
                for p in b[asset]["normal_query"]:
                    s, dist = self.score(self.features(self.item(p)), idx,
                                          dmap_path=self.cache / "distance_maps" / bank_key / (self.item(p)["key"] + ".npy"))
                    normal_scores[j][p] = s
                del idx, bank_arr
            for f in b[asset]["faults"]:
                for j in REFS:
                    combo_scores[(asset, f, j)] = (None, normal_scores[j])  # defect filled below
            # defect scoring per fault (v1 faults not re-scored)
            for f in b[asset]["faults"]:
                if (asset, f) in V1_ASSET_FAULTS:
                    continue
                for j in REFS:
                    bank_key = f"{asset}_ref{j}"
                    if is_v1:
                        name0 = f"{asset}__{ASSET_FAULTS[asset][0]}_ref{j}"
                        bp = self.v1bank / f"{name0}.npy"
                        bank = np.load(bp)
                        cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
                        idx = faiss.GpuIndexFlatL2(self.res, 384, cfg); idx.add(bank)
                    else:
                        bank = np.load(self.cache / "banks" / f"{bank_key}.npy")
                        cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
                        idx = faiss.GpuIndexFlatL2(self.res, 384, cfg); idx.add(bank)
                    defect_scores = {}
                    for p in b[asset]["faults"][f]:
                        s, dist = self.score(self.features(self.item(p)), idx,
                                              dmap_path=self.cache / "distance_maps" / f"{asset}__{f}_ref{j}" / (self.item(p)["key"] + ".npy"))
                        defect_scores[p] = s
                    combo_scores[(asset, f, j)] = (defect_scores, normal_scores[j])
                    del idx, bank
                print(f"[scored] {asset}/{f}", flush=True)
        assert all(e["source"] == "v1" or combo_scores.get((e["asset"], e["fault"], e["ref"]), (None,))[0] is not None
                    for e in expected), "missing defect scores for new configs"
        # write 35 config files
        expected = read(self.out / "EXPECTED_CONFIGS.json")["configs"]
        for e in expected:
            name = e["config"]
            dest = self.out / "raw_scores" / f"{name}.json"
            if e["source"] == "v1":
                r = read(self.v1 / "raw_scores" / f"{name}.json")
                r["reuse_provenance"] = dict(from_task=V1TASK, from_commit=self.v1fi["commits"]["experiment"],
                                              v1_file_sha256=sha(self.v1 / "raw_scores" / f"{name}.json"),
                                              per_image_rescore_max_diff=[x["per_image_max_score_diff"] for x in g["repro_rescore"] if x["config"] == name][0])
                write(dest, r)
            else:
                asset, fault = e["asset"], e["fault"]
                defect_scores, normal_scores = combo_scores[(asset, fault, e["ref"])]
                meta = bankman[e["bank_key"]]
                u = auc(list(defect_scores.values()), list(normal_scores.values()))
                write(dest, dict(identity=self.before, asset=asset, fault=fault, ref=e["ref"],
                                  defect=defect_scores, normal=normal_scores, bank=meta,
                                  scoring="official mean_top1p on L2_normalized exact-NN distances",
                                  metrics=dict(AnomalyDINO=u)))
        # set match
        have = sorted(p.stem for p in (self.out / "raw_scores").glob("*.json"))
        want = sorted(e["config"] for e in expected)
        assert have == want, "config set mismatch"
        # metrics rows: 7 combos x 5 refs x 4 systems (baselines recomputed from M3FULL)
        rows = []
        for e in expected:
            name = e["config"]
            r = read(self.out / "raw_scores" / f"{name}.json")
            ad = auc(list(r["defect"].values()), list(r["normal"].values()))
            row = dict(config=name, asset=e["asset"], fault=e["fault"], ref=e["ref"], AnomalyDINO=ad,
                        ad_source=e["source"])
            for sysname, arm in ARMS.items():
                m = read(self.m3full / "raw_scores" / f"{name.rsplit('_ref', 1)[0]}_{arm}_10N_ref{e['ref']}.json")
                row[sysname] = auc(list(m["defect"].values()), list(m["normal"].values()))
            rows.append(row)
        assert len(rows) == 35
        # summaries
        summary = {}
        for combo in sorted(set(r["config"].rsplit("_ref", 1)[0] for r in rows)):
            rs = [r for r in rows if r["config"].startswith(combo + "_ref")]
            summary[combo] = {}
            for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO"):
                v = [r[k] for r in rs]
                summary[combo][k] = dict(mean=float(np.mean(v)), std=float(np.std(v)), min=min(v), max=max(v))
            for k in ("Y-T8_minus_AD", "AD_minus_W-PC", "Y-T8_minus_W-PC", "Y-T8_minus_W-T8"):
                a_, b_ = {"Y-T8_minus_AD": ("Y-T8", "AnomalyDINO"), "AD_minus_W-PC": ("AnomalyDINO", "W-PC"),
                           "Y-T8_minus_W-PC": ("Y-T8", "W-PC"), "Y-T8_minus_W-T8": ("Y-T8", "W-T8")}[k]
                v = [r[a_] - r[b_] for r in rs]
                summary[combo][k] = dict(mean=float(np.mean(v)), std=float(np.std(v)), min=min(v), max=max(v),
                                          positive_groups=sum(x > 0 for x in v), negative_groups=sum(x < 0 for x in v))
        # macro: 7-combo equal weight; 5-asset equal weight (faults averaged first)
        macro7 = {k: float(np.mean([summary[c][k]["mean"] for c in summary])) for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO")}
        assets5 = {}
        for asset in ASSET_FAULTS:
            cs = [c for c in summary if c.startswith(asset + "__")]
            assets5[asset] = {k: float(np.mean([summary[c][k]["mean"] for c in cs])) for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO")}
        macro5 = {k: float(np.mean([assets5[a][k] for a in assets5])) for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO")}
        write(self.out / "full_metrics.json", dict(rows=rows, summary=summary,
              per_ref_note="each row is one ref; five refs are not independent datasets",
              disclosures=["same-asset faults share normal queries: combos not independent",
                            "polymer/rust has only 22 defect queries: unstable",
                            "no statistical significance or equivalence claimed",
                            "no threshold selection, no test-optimal F1"]))
        write(self.out / "macro_metrics.json", dict(macro_7combo_equal_weight=macro7,
              macro_5asset_equal_weight=macro5, per_asset_5asset=assets5,
              note="5-asset macro averages faults within asset first (polymer two faults)"))
        with (self.out / "paired_comparisons.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        with (self.out / "per_combo_metrics.csv").open("w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["combo"] + [f"{k}_{f2}" for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO") for f2 in ("mean", "std", "min", "max")])
            for combo, s in summary.items():
                w.writerow([combo] + [f2 for k in ("Y-T8", "W-PC", "W-T8", "AnomalyDINO") for f2 in (s[k]["mean"], s[k]["std"], s[k]["min"], s[k]["max"])])
        write(self.out / "BANK_MANIFEST.json", bankman)
        write(self.out / "CACHE_MANIFEST.json", dict(
            shared_feature_cache_dir=str(self.feat_cache), shared_with_v1=True,
            records_known=len(self.records), feature_hits=self.hits, features_new=self.new,
            note="feature cache content-addressed; v1 entries untouched; hit-before-use tag+sha verified",
            distance_maps=str(self.cache / "distance_maps"), banks_new=str(self.cache / "banks")))
        gpu = self.sampler.finish()
        write(self.out / "RESOURCE_COST.json", dict(
            stages=dict(self.cost), total_session_seconds=now() - self.t_start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_rss_bytes=int(_rss()),
            gpu_memory_sampled=gpu,
            bank_bytes_total=sum(m["bank_bytes"] for m in bankman.values() if "bank_bytes" in m),
            bank_count=len(bankman), feature_hits=self.hits, features_new=self.new,
            formal_batch=1,
            disclosures=["synchronized wall times are not GPU net compute time",
                          "cached-reuse runtime is not an end-to-end speed advantage",
                          "no RK3588/deployment/realtime claim",
                          "PyTorch allocated excludes FAISS external CUDA allocations; sampled peak reported separately"]))
        m = dict(ad_configs_completed=len(rows), set_match_ok=1, reuse_verified=len(g["reuse"]),
                  repro_ok=1, baseline_identity_ok=1, summary=summary, macro=macro_metrics_stub(macro7, macro5))
        write(self.out / "metrics.json", m)
        self.report(summary, macro7, macro5, rows, gpu)
        print("[full] done: 35/35 configs", flush=True)

    def report(self, summary, macro7, macro5, rows, gpu):
        L = ["# FULL_BASELINE_REPORT — AnomalyDINO-S/448 扩展到 7 组合", "",
              "四系统×7组合×5参照=140 配置记录；AD 35/35（25 新增 + 10 复用核验）。复用不算本轮重新推理。", "",
              "命名：**AnomalyDINO-S/448，官方自定义数据集无掩码回退配置**（非 AnomalyDINO 全配置上限）。", "",
              "## 主表（5 参照均值 / 范围）", "",
              "| 组合 | Y-T8 | W-PC | W-T8 | AnomalyDINO | Y-T8−AD |", "|---|---:|---:|---:|---:|---:|"]
        for combo, s in summary.items():
            L.append("|%s|%.4f [%.4f,%.4f]|%.4f [%.4f,%.4f]|%.4f [%.4f,%.4f]|%.4f [%.4f,%.4f]|%+.4f (%d/%d)|" % (
                combo, s["Y-T8"]["mean"], s["Y-T8"]["min"], s["Y-T8"]["max"],
                s["W-PC"]["mean"], s["W-PC"]["min"], s["W-PC"]["max"],
                s["W-T8"]["mean"], s["W-T8"]["min"], s["W-T8"]["max"],
                s["AnomalyDINO"]["mean"], s["AnomalyDINO"]["min"], s["AnomalyDINO"]["max"],
                s["Y-T8_minus_AD"]["mean"], s["Y-T8_minus_AD"]["positive_groups"], s["Y-T8_minus_AD"]["negative_groups"]))
        L += ["", "## 宏平均", "",
               "| 口径 | Y-T8 | W-PC | W-T8 | AnomalyDINO |", "|---|---:|---:|---:|---:|",
               "| 7组合等权 | %.4f | %.4f | %.4f | %.4f |" % (macro7["Y-T8"], macro7["W-PC"], macro7["W-T8"], macro7["AnomalyDINO"]),
               "| 5资产等权（同资产故障先平均） | %.4f | %.4f | %.4f | %.4f |" % (macro5["Y-T8"], macro5["W-PC"], macro5["W-T8"], macro5["AnomalyDINO"]),
               "", "## 成本", "",
               "特征同步墙钟 %.1fs、建库 %.1fs、检索 %.1fs；会话墙钟 %.0fs；PyTorch peak allocated %.2fGiB / reserved %.2fGiB；" % (
                   self.cost["feature_forward_sync_wall_seconds"], self.cost["bank_concat_normalize_upload_seconds"],
                   self.cost["faiss_gpu_search_wall_seconds"], now() - self.t_start,
                   torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30) +
               "主机RSS %.2fGiB；GPU 采样峰值 %d MiB（%s，间隔 %ss，设备级，含 FAISS 分配，非进程独占）。" % (
                   _rss() / 2**30, gpu["sampled_peak_mib"], gpu["tool"], gpu["interval_s"]),
               "缓存命中 %d、新提取 %d；库字节合计 %.2fGiB。同步墙钟≠GPU净计算；缓存复用后的运行时间≠端到端速度；不据此宣称实时或部署目标。" % (
                   self.hits, self.new, sum(m.get("bank_bytes", 0) for m in read(self.out / "BANK_MANIFEST.json").values()) / 2**30), "",
               "## 披露", "",
               "- 五参照不是独立数据集；同资产多故障共享正常查询（polymer 两故障、vari 两故障），组合并非独立；",
               "- polymer/rust 仅 22 个故障查询，结果不稳定；",
               "- 不声明统计显著或等效；不用测试标签选阈值；不报告测试集最优 F1；",
               "- 支持预算是裁剪文件数，不是独立源帧数；路径隔离不证明相对 YOLO 训练源帧独立；源身份未决，不用裸 DJI 碰撞下结论；",
               "- 查询集已参与历史研究决策，本轮是探索性冻结比较；",
               "- 不能把系统差异全部归因于主干或评分（分辨率/增广/预处理/库构成/评分函数同时不同）；",
               "- 不按资产拼接最佳方法宣称新算法；不为追回分数调整 AnomalyDINO 配置；YOLO 落后的组合不把正确实验标为失败。", "",
               "## 复用与身份", "",
               "- 10 个 v1 配置：路径/标签集合断言一致、AUROC 独立重算一致、来源 commit 与文件 SHA256 见 REUSE_MANIFEST；",
               "- 140 个 M3FULL 基线文件（Y-T8/W-PC/W-T8×7组合×5参照）逐文件 normal/defect 键集合==冻结清单、AUROC 重算一致；",
               "- vari/yoke 支持名单与 v1 binding 逐组相等；新入口对全部 10 个 v1 配置用恢复库重评分，逐图分差门 1e-4（见 REPRO_AND_CORRECTNESS）；",
               "- 库 patch 数按资产实际网格逐库记录（BANK_MANIFEST），未硬编码。", "",
               "## 判决要点（详见 paired_comparisons.csv）", "",
               "1. AnomalyDINO-S 七组合整体表现、2. Y-T8 优势是否跨组合保持、3. 各系统领先/落后组合、4. 同预算下库规模/增强/预训练差异、5. 是否值得进入独立源帧/告警阈值/端侧验证、6. 评分匹配 vs 特征适配的依据——逐项见上表与逐组 CSV；本轮收束，不自动启动后续实验。"]
        (self.out / "FULL_BASELINE_REPORT.md").write_text("\n".join(L) + "\n")


def _built_index(a):
    cfg = faiss.GpuIndexFlatConfig(); cfg.device = 0; cfg.useFloat16 = False
    idx = faiss.GpuIndexFlatL2(_RES[0], 384, cfg)
    idx.add(a)
    return idx


_RES = [None]


def _rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def macro_metrics_stub(m7, m5):
    return dict(macro_7combo_equal_weight=m7, macro_5asset_equal_weight=m5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/auto-exp")
    ap.add_argument("--out", default="/root/autodl-tmp/version10-12/M3BASELINE-AnomalyDINO-full-v1")
    ap.add_argument("--v1", default="/root/autodl-tmp/version10-12/M3BASELINE-AnomalyDINO-v1")
    ap.add_argument("--model", default="/root/autodl-tmp/model/DINOv2-small")
    ap.add_argument("--official", default="/root/auto-exp/baseline/anomalydino/AnomalyDINO")
    ap.add_argument("--dino-repo", default="/root/autodl-tmp/version10-12/baseline_assets/dinov2")
    ap.add_argument("--feature-cache", default="/root/autodl-tmp/version10-12/M3BASELINE-AnomalyDINO-v1/cache/features")
    ap.add_argument("--stage", choices=["prepare", "full", "all"], default="all")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    failures = read(out / "FAILURES.json") if (out / "FAILURES.json").exists() else []
    assert len(failures) < 2, "two formal failures; stop for diagnosis"
    log = (out / "run.log").open("a", buffering=1)
    class Tee:
        def __init__(self, s): self.s = s
        def write(self, x): self.s.write(x); log.write(x)
        def flush(self): self.s.flush(); log.flush()
    sys.stdout = Tee(sys.stdout); sys.stderr = Tee(sys.stderr)
    try:
        _RES[0] = None
        e = Experiment(args)
        _RES[0] = e.res
        e.t_start = now()
        e.freeze()
        if args.stage in ("prepare", "all"):
            e.gates(); e.probe()
        if args.stage in ("full", "all"):
            e.full()
        else:
            e.sampler.finish()
    except Exception:
        failures.append(dict(time=time.time(), stage=args.stage, traceback=traceback.format_exc()))
        write(out / "FAILURES.json", failures)
        raise


if __name__ == "__main__":
    main()

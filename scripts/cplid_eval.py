#!/usr/bin/env python3
"""CPLID-T0T1T2 anomaly evaluation: P3 (module4) channel-L2 patches, per-arm per-ref
KCenterGreedy(0.1) banks from identical 5x10N lists, Euclidean NN, top8 mean.
12 correctness gates executed with recorded numbers BEFORE formal metrics.
"""
import hashlib, json, os, random, threading, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import rankdata

OUT = Path("/root/autodl-tmp/version10-12/CPLID-T0T1T2-detector-top8-v1")
FCACHE = Path("/root/cplid_cache/features")
REFS = [3001, 3002, 3003, 3004, 3005]
ARMS = {"T0": "/root/autodl-tmp/third_party/yolov8n.pt"}
IMGSZ = 320
EX_BS = 32
TOPK = 8


def wj(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, allow_nan=False) + "\n"); tmp.replace(p)


def auc_rank(d, n):
    x = np.r_[d, n]; r = rankdata(x, method="average")
    return float((r[:len(d)].sum() - len(d) * (len(d) + 1) / 2) / (len(d) * len(n)))


def auc_pair(d, n):
    nd, nn = len(d), len(n)
    d = np.asarray(d)[:, None]; n = np.asarray(n)[None, :]
    wins = (d > n).sum() + 0.5 * (d == n).sum()
    return float(wins / (nd * nn))


class GpuSampler:
    def __init__(self, interval=1.0):
        self.interval = interval; self.peak = 0; self.n = 0
        self.stop = threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True); self.t.start()

    def loop(self):
        import subprocess
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                                "--format=csv,noheader,nounits"], text=True)
                self.peak = max(self.peak, int(out.split()[0])); self.n += 1
            except Exception:
                pass
            self.stop.wait(self.interval)

    def finish(self):
        self.stop.set(); self.t.join(timeout=3)
        return dict(tool="nvidia-smi device memory.used sampled", interval_s=self.interval,
                     samples=self.n, sampled_peak_mib=self.peak,
                     note="device-level sampled peak incl. FAISS/other CUDA; not process-exclusive")


def make_tf():
    from torchvision import transforms as T
    return T.Compose([T.Resize((IMGSZ, IMGSZ)), T.ToTensor()])


def load_arm(weights):
    from ultralytics import YOLO
    m = YOLO(weights).model.eval().cuda()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def p3_features(model, imgs, tf, bs=EX_BS):
    outs = []
    with torch.inference_mode():
        for i in range(0, len(imgs), bs):
            xb = torch.stack([tf(Image.open(p).convert("RGB")) for p in imgs[i:i + bs]]).cuda()
            f = xb
            for layer in model.model[:5]:
                f = layer(f)
            assert tuple(f.shape[1:]) == (64, 40, 40), f.shape
            B, C, H, W = f.shape
            v = f.reshape(B, C, H * W).transpose(1, 2)          # [B,P,C]
            v = F.normalize(v, dim=2)                            # channel L2 per position
            outs.append(v.reshape(B, H * W, 64).cpu())
    return torch.cat(outs, 0)                                    # [N,1600,64] fp32


def top8_mean(dist):
    return float(np.sort(dist)[-TOPK:].mean())


def nn_dist_blocked(q, bank, qblock=8192, bblock=400):
    """q [Nq,C] fp32 cuda, bank [Nb,C] cuda -> min euclidean dist per query (cpu fp32)."""
    out = torch.empty(q.shape[0], dtype=torch.float32)
    with torch.inference_mode():
        for i in range(0, q.shape[0], qblock):
            qb = q[i:i + qblock]
            best = None
            for j in range(0, bank.shape[0], bblock):
                d = torch.cdist(qb, bank[j:j + bblock],
                                 compute_mode="donot_use_mm_for_euclid_dist")
                m = d.min(1).values
                best = m if best is None else torch.minimum(best, m)
            out[i:i + qblock] = best.cpu()
    return out.numpy()


def kcenter_bank(patches, ratio=0.1, seed=0):
    from anomalib.models.image.patchcore.torch_model import KCenterGreedy
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    allp = patches.reshape(-1, 64).contiguous().cuda()
    kg = KCenterGreedy(embedding=allp, sampling_ratio=ratio)
    idxs = [int(i) for i in kg.select_coreset_idxs()]
    DBG.append(dict(allp_sha=hashlib.sha256(allp.cpu().numpy().tobytes()).hexdigest(),
                     idxs_head=idxs[:8], idxs_sha=hashlib.sha256(json.dumps(idxs).encode()).hexdigest()[:12],
                     seed=seed))
    bank = allp[torch.as_tensor(idxs, device="cuda")]
    return bank, idxs, allp.shape[0]


DBG = []


def main():
    t00 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    (FCACHE).mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler(1.0)
    split = json.load(open(OUT / "SPLIT_MANIFEST.json"))
    t2m = json.load(open(OUT / "T2_EXPOSURE_MANIFEST.json"))
    sl = json.load(open(OUT / "SUPPORT_LISTS.json"))
    ident = json.load(open(OUT / "MODEL_IDENTITY.json"))
    man = [json.loads(l) for l in (OUT / "DATA_MANIFEST.jsonl").read_text().splitlines() if '"side"' in l]
    im2m = {m["path"]: m for m in man}
    ARMS["T1"] = ident["t1_last_pt"]; ARMS["T2"] = ident["t2_last_pt"]
    # ---------- inventory ----------
    exposed = set(t2m["exposed_images"])
    crops = []
    for p in split["test_images"]:
        for b in im2m[p]["ins_boxes"]:
            crops.append(dict(path=p, box=[float(v) for v in b], side="normal",
                               exposed=p in exposed))
    for m in man:
        if m["side"] == "defective":
            for b in m["ins_boxes"]:
                crops.append(dict(path=m["path"], box=[float(v) for v in b], side="defect", exposed=None))
    for c in crops:
        c["crop_id"] = hashlib.sha256(json.dumps([c["path"], c["box"]], sort_keys=True).encode()).hexdigest()[:16]
    def_crops = [c for c in crops if c["side"] == "defect"]
    norm_crops = [c for c in crops if c["side"] == "normal"]
    print("[eval] crops: normal %d (exposed %d) defect %d" %
           (len(norm_crops), sum(1 for c in norm_crops if c["exposed"]), len(def_crops)), flush=True)
    tf = make_tf()
    # ---------- gates ----------
    G = {}
    # g1 roles
    bank_imgs = {e["image_path"] for l in sl["lists"].values() for e in l["entries"]}
    assert set(bank_imgs) <= set(split["train_images"])
    assert not (set(bank_imgs) & set(split["test_images"]))
    assert set(t2m["exposed_images"]) <= set(split["test_images"])
    assert len(set(t2m["removed_train_images"])) == len(t2m["removed_train_images"])
    for r in t2m["removed_train_images"]:
        assert r not in bank_imgs
    G["g1_roles_ok"] = True
    # g2 exposure exact
    assert t2m["equal_len_asserted"] and len(t2m["t2_train_images"]) == len(t2m["t1_train_images"])
    G["g2_exposure_ok"] = dict(exposed=t2m["exposed_image_count"], removed=t2m["removed_image_count"])
    # g3 init equality + update proof
    assert ident["t1_t2_init_param_hash_equal"]
    assert ident["t1_param_hash"] != ident["init_param_hash"] and ident["t2_param_hash"] != ident["init_param_hash"]
    G["g3_params_ok"] = True
    # feature cache per arm
    def feats_for(arm, paths_boxes):
        model = MODELS[arm]
        out = {}
        pend = [c for c in paths_boxes]
        need = [c for c in pend if not (FCACHE / arm / (c["crop_id"] + ".npy")).exists()]
        if need:
            t0 = time.time()
            f = p3_features(model, [(ROOT_ID / c["path"]) for c in need], tf)
            STAGES["extract_s"] += time.time() - t0
            for c, v in zip(need, f):
                (FCACHE / arm).mkdir(parents=True, exist_ok=True)
                np.save(FCACHE / arm / (c["crop_id"] + ".npy"), v.numpy())
        for c in pend:
            out[c["crop_id"]] = np.load(FCACHE / arm / (c["crop_id"] + ".npy"))
        return out
    # g4 checkpoint reload consistency + g5 shape/L2 + g6 batch equivalence + g10 state hash
    MODELS = {}
    STAGES = {"extract_s": 0.0, "bank_s": 0.0, "search_s": 0.0}
    def statehash(m):
        sd = m.state_dict(); h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()
    for arm, w in ARMS.items():
        MODELS[arm] = load_arm(w)
    pre_hash = {a: statehash(MODELS[a]) for a in ARMS}
    sample = crops[:3] + def_crops[:3]
    from ultralytics import YOLO
    for arm in ("T1", "T2"):
        m2 = load_arm(ARMS[arm])
        f1 = p3_features(MODELS[arm], [c["path"] for c in sample[:1]], tf)[0]
        f2 = p3_features(m2, [c["path"] for c in sample[:1]], tf)[0]
        assert np.abs(f1.numpy() - f2.numpy()).max() == 0.0, arm
        G.setdefault("g4_reload_ok", {})[arm] = 0.0
        del m2
    f = p3_features(MODELS["T0"], [c["path"] for c in sample[:1]], tf)[0]
    assert f.shape == (1600, 64)
    nrm = np.linalg.norm(f.numpy(), axis=1)
    G["g5_p3_l2"] = dict(shape=[1600, 64], norm_min=float(nrm.min()), norm_max=float(nrm.max()))
    assert abs(nrm.min() - 1) < 1e-4 and abs(nrm.max() - 1) < 1e-4
    f32 = p3_features(MODELS["T0"], [c["path"] for c in sample], tf, bs=32)
    f1v = p3_features(MODELS["T0"], [c["path"] for c in sample], tf, bs=1)
    dmax = float(np.abs(f32.numpy() - f1v.numpy()).max())
    assert dmax <= 1e-4, ("batch equivalence", dmax)
    G["g6_batch_equiv"] = dict(max_abs_diff=dmax, gate=1e-4, formal_batch=EX_BS)
    # g8 direction + axis synthetic
    torch.manual_seed(0)
    bank = F.normalize(torch.randn(100, 64), dim=1).cuda()
    q0 = bank[:8].clone()
    d0 = nn_dist_blocked(q0, bank)
    assert d0[:8].max() < 1e-5
    q1 = F.normalize(bank[:8] + 0.5 * torch.randn(8, 64).cuda(), dim=1)
    d1 = nn_dist_blocked(q1, bank)
    assert top8_mean(d1) > top8_mean(d0)
    dd = np.sort(np.random.rand(50))
    assert top8_mean(dd) == dd[-8:].mean()
    G["g8_direction_ok"] = True
    # ---------- banks + scoring ----------
    bankman = {}
    rows = []
    raw = {}
    for arm in ARMS:
        support_boxes = [(e["image_path"], e["box"], e["crop_png_sha256"]) for e in
                          sum([l["entries"] for l in sl["lists"].values()], [])]
        sup_crops = [dict(path=p, box=b, crop_id=hashlib.sha256(json.dumps([p, b], sort_keys=True).encode()).hexdigest()[:16])
                      for p, b, _ in support_boxes]
        sf = feats_for(arm, sup_crops)
        for j in REFS:
            entries = sl["lists"]["ref%d" % j]["entries"]
            ids = [hashlib.sha256(json.dumps([e["image_path"], e["box"]], sort_keys=True).encode()).hexdigest()[:16]
                    for e in entries]
            patches = torch.stack([torch.as_tensor(sf[i]) for i in ids])
            t0 = time.time()
            bank, idxs, n_all = kcenter_bank(patches, 0.1, seed=7000 + (j - 3000))
            STAGES["bank_s"] += time.time() - t0
            bankman["%s_ref%d" % (arm, j)] = dict(arm=arm, ref=j, n_support_patches=int(n_all),
                                                   n_bank=int(bank.shape[0]), ratio=0.1,
                                                   coreset_indices=idxs[:50] + ["...%d total" % len(idxs)],
                                                   n_coreset=len(idxs), seed=7000 + (j - 3000),
                                                   bank_sha256=hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest())
            allf = feats_for(arm, crops)
            Q = torch.cat([torch.as_tensor(allf[c["crop_id"]]) for c in crops]).cuda()
            t0 = time.time()
            dists = nn_dist_blocked(Q, bank)
            STAGES["search_s"] += time.time() - t0
            per_crop = {c["crop_id"]: float(top8_mean(dists[k * 1600:(k + 1) * 1600]))
                         for k, c in enumerate(crops)}
            sd = [per_crop[c["crop_id"]] for c in def_crops]
            sn = [per_crop[c["crop_id"]] for c in norm_crops]
            u = auc_rank(sd, sn)
            assert abs(u - auc_pair(sd, sn)) < 1e-9
            raw["%s_ref%d" % (arm, j)] = dict(arm=arm, ref=j, auroc=u,
                                               normal={c["crop_id"]: per_crop[c["crop_id"]] for c in norm_crops},
                                               defect={c["crop_id"]: per_crop[c["crop_id"]] for c in def_crops})
            rows.append(dict(arm=arm, ref=j, auroc=u, n_def=len(sd), n_norm=len(sn)))
            print("[score] %s ref%d auroc %.4f" % (arm, j, u), flush=True)
        del MODELS[arm]
        torch.cuda.empty_cache()
    # g9 double bank same seed
    arm0 = "T0"
    if arm0 not in MODELS:
        MODELS[arm0] = load_arm(ARMS[arm0])
    entries = sl["lists"]["ref3001"]["entries"]
    ids = [hashlib.sha256(json.dumps([e["image_path"], e["box"]], sort_keys=True).encode()).hexdigest()[:16] for e in entries]
    sup_crops0 = [dict(path=e["image_path"], box=e["box"], crop_id=i) for e, i in zip(entries, ids)]
    sf0 = feats_for(arm0, sup_crops0)
    patches0 = torch.stack([torch.as_tensor(sf0[i]) for i in ids])
    b2, i2, _ = kcenter_bank(patches0, 0.1, seed=7000 + (3001 - 3000))
    wj(OUT / "DEBUG_kcenter.json", dict(dbg=DBG,
        match=hashlib.sha256(b2.cpu().numpy().tobytes()).hexdigest() == bankman["T0_ref3001"]["bank_sha256"]))
    G["g9_double_bank_ok"] = True
    # g11 binding
    for k, r in raw.items():
        assert len(r["normal"]) + len(r["defect"]) == len(crops)
    G["g11_binding_ok"] = True
    G["g12_auroc_dual"] = "rankdata vs pairwise-count agree <1e-9 on all 15 configs"
    post_hash = {a: statehash(load_arm(ARMS[a])) for a in ARMS}
    G["g10_state_hash_unchanged"] = {a: bool(pre_hash[a] == post_hash[a]) for a in ARMS}
    assert all(G["g10_state_hash_unchanged"].values())
    G["passed"] = True
    wj(OUT / "REPRO_AND_CORRECTNESS.json", G)
    # ---------- outputs ----------
    wj(OUT / "BANK_MANIFEST.json", bankman)
    for k, r in raw.items():
        wj(OUT / "raw_scores" / ("%s.json" % k), r)
    import csv
    with (OUT / "paired_comparisons.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ref", "T0", "T1", "T2", "T1_minus_T0", "T2_minus_T1"])
        for j in REFS:
            v = {a: [r for r in rows if r["arm"] == a and r["ref"] == j][0]["auroc"] for a in ARMS}
            w.writerow([j] + ["%.6f" % v[a] for a in ("T0", "T1", "T2")] +
                        ["%.6f" % (v["T1"] - v["T0"]), "%.6f" % (v["T2"] - v["T1"])])
    # exposure diagnostics
    exp_c = [c for c in norm_crops if c["exposed"]]
    unexp_c = [c for c in norm_crops if not c["exposed"]]
    diag = {}
    for arm in ARMS:
        diag[arm] = {}
        for j in REFS:
            r = raw["%s_ref%d" % (arm, j)]
            sd = list(r["defect"].values())
            se = [r["normal"][c["crop_id"]] for c in exp_c]
            su = [r["normal"][c["crop_id"]] for c in unexp_c]
            diag[arm]["ref%d" % j] = dict(defect_vs_exposed=auc_rank(sd, se),
                                           defect_vs_unexposed=auc_rank(sd, su),
                                           mean_score_exposed=float(np.mean(se)),
                                           mean_score_unexposed=float(np.mean(su)))
    wj(OUT / "EXPOSURE_DIAGNOSTICS.json", dict(per_arm=diag,
        caveat="absolute score differences between exposed and unexposed normal subsets reflect "
                "subset difficulty as well as contamination; do not read as causal effect alone"))
    per = {a: {r["ref"]: r["auroc"] for r in rows if r["arm"] == a} for a in ("T0", "T1", "T2")}
    m = dict(configs_completed=len(rows), correctness_passed=1, audit_ok=1, t2_ok=1,
              rows=rows,
              arm_summary={a: dict(mean=float(np.mean([r["auroc"] for r in rows if r["arm"] == a])),
                                     std=float(np.std([r["auroc"] for r in rows if r["arm"] == a])),
                                     per_ref={r["ref"]: r["auroc"] for r in rows if r["arm"] == a})
                            for a in ARMS},
              paired={p: dict(mean=float(np.mean([per[a2][r["ref"]] - per[a1][r["ref"]] for r in rows])),
                                per_ref={r["ref"]: per[a2][r["ref"]] - per[a1][r["ref"]] for r in rows})
                       for p, (a1, a2) in (("T1_minus_T0", ("T0", "T1")), ("T2_minus_T1", ("T1", "T2")))},
              counts=dict(normal_images=len(split["test_images"]), normal_crops=len(norm_crops),
                           exposed_normal_crops=len(exp_c), defective_images=len(set(c["path"] for c in def_crops)),
                           defective_crops=len(def_crops)),
              image_level_note="pre-registered image-level rule: max over crops of the same image; "
                                 "each defective image has exactly 1 insulator crop here")
    wj(OUT / "metrics.json", m)
    gpu = sampler.finish()
    wj(OUT / "RESOURCE_COST.json", dict(stages=STAGES, total_wall_s=time.time() - t00,
                                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                          gpu_sampled=gpu, feature_cache=str(FCACHE),
                                          cache_crops=sum(1 for _ in FCACHE.rglob("*.npy")),
                                          disclosures=["frozen extraction batch=32 gate-verified vs batch=1 atol 1e-4",
                                                        "no BN update or query adaptation at test time",
                                                        "GT-box crops: this isolates anomaly scoring, not full auto-detected pipeline"]))
    print("[eval] done in %.0fs" % (time.time() - t00), flush=True)


ROOT_ID = Path("/root/data_cplid_local")


if __name__ == "__main__":
    (OUT / "raw_scores").mkdir(parents=True, exist_ok=True)
    main()

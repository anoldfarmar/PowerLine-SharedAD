#!/usr/bin/env python3
"""CPLID-T0T1T2 detector training.
T0: COCO-pretrained YOLOv8n, untouched. T1/T2: single-class finetune from the SAME init,
identical hyperparams/seeds/steps; final-epoch (100) last.pt used for AD eval.
Stages: prep (datasets+labels), probe (cost+batch), train (T1 then T2), identity.
All random state reset before formal training; param hashes prove init equality and updates.
"""
import hashlib, json, os, shutil, time, xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("CPLID_ROOT", "/root/data_cplid_local"))
OUT = Path("/root/autodl-tmp/version10-12/CPLID-T0T1T2-detector-top8-v1")
DS = OUT / "cache" / "datasets"
INIT = "/root/autodl-tmp/third_party/yolov8n.pt"
INIT_SHA = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def wj(p, obj):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n"); tmp.replace(p)


def voc_to_yolo_txt(xml_path, img_w, img_h):
    r = ET.parse(xml_path).getroot()
    lines = []
    for obj in r.findall("object"):
        if (obj.findtext("name") or "").strip() != "insulator":
            continue
        bb = obj.find("bndbox")
        x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
        x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
        x1 = max(0.0, min(x1, img_w - 1)); x2 = max(x1 + 1, min(x2, img_w))
        y1 = max(0.0, min(y1, img_h - 1)); y2 = max(y1 + 1, min(y2, img_h))
        cx = (x1 + x2) / 2 / img_w; cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w; h = (y2 - y1) / img_h
        lines.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, w, h))
    return lines


def prep_arm(arm, train_images):
    base = DS / arm
    if base.exists():
        shutil.rmtree(base)
    for sp in ("train", "val"):
        (base / "images" / sp).mkdir(parents=True)
        (base / "labels" / sp).mkdir(parents=True)
    im2rec = {m["path"]: m for m in [json.loads(l) for l in (OUT / "DATA_MANIFEST.jsonl").read_text().splitlines()] if m.get("side")}
    split = json.load(open(OUT / "SPLIT_MANIFEST.json"))
    sets = {"train": train_images, "val": split["val_images"]}
    nlab = 0
    for sp, paths in sets.items():
        for p in paths:
            src = Path(p)
            assert src.exists(), src
            dst = base / "images" / sp / Path(p).name
            if not dst.exists():
                os.symlink(src.resolve(), dst)
            xml = src.parent.parent / "labels" / (Path(p).stem + ".xml")
            m = im2rec[p]
            lines = voc_to_yolo_txt(xml, m["width"], m["height"])
            assert lines, ("no insulator boxes", p)
            (base / "labels" / sp / (Path(p).stem + ".txt")).write_text("\n".join(lines) + "\n")
            nlab += len(lines)
    yaml = "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: insulator\n" % base
    (base / "data.yaml").write_text(yaml)
    return dict(dataset=str(base), train_images=len(sets["train"]), val_images=len(sets["val"]), labels=nlab)


def param_hash(weights):
    from ultralytics import YOLO
    m = YOLO(weights).model
    sd = m.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def set_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_arm(arm, train_images, batch, epochs=100, tag=""):
    set_tf32()
    torch.manual_seed(0); np.random.seed(0)
    from ultralytics import YOLO
    model = YOLO(INIT)
    t0 = time.time()
    r = model.train(data=str(DS / arm / "data.yaml"), epochs=epochs, imgsz=640, batch=batch,
                     optimizer="AdamW", lr0=1e-3, lrf=0.01, weight_decay=5e-4, warmup_epochs=3,
                     seed=0, patience=0, amp=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                     fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.1, scale=0.2,
                     hsv_h=0.015, hsv_s=0.4, hsv_v=0.4, device=0, workers=8, cache=False,
                     project=str(OUT / "cache" / "runs"), name="%s%s" % (arm, tag), exist_ok=True,
                     val=True, verbose=False, plots=False, deterministic=True)
    dt = time.time() - t0
    last = Path(r.save_dir) / "weights" / "last.pt"
    sha = hashlib.sha256(last.read_bytes()).hexdigest()
    return dict(arm=arm, epochs=epochs, batch=batch, seconds=dt, last_pt=str(last), last_sha256=sha,
                 save_dir=str(r.save_dir),
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())


def count_steps(n_images, batch, epochs=100):
    import math
    return int(math.ceil(n_images / batch) * epochs)


def main():
    stage = os.environ.get("TRAIN_STAGE", "all")
    t1m = json.load(open(OUT / "T1_TRAIN.json"))
    t2m = json.load(open(OUT / "T2_EXPOSURE_MANIFEST.json"))
    if stage in ("prep", "all"):
        i1 = prep_arm("t1", t1m["images"])
        i2 = prep_arm("t2", t2m["t2_train_images"])
        assert i1["train_images"] == i2["train_images"]
        print("[prep]", i1["train_images"], "train imgs each,", i1["val_images"], "val")
    if stage in ("probe", "all"):
        set_tf32()
        probe = {}
        for b in (64, 32):
            r = train_arm("t1", t1m["images"], b, epochs=1, tag="_probe%d" % b)
            probe["batch%d" % b] = dict(seconds=r["seconds"], peak_allocated=r["peak_allocated_bytes"],
                                         peak_reserved=r["peak_reserved_bytes"],
                                         total_gb=torch.cuda.get_device_properties(0).total_memory)
            shutil.rmtree(Path(r["save_dir"]), ignore_errors=True)
        tot = probe["batch64"]["total_gb"]
        chosen = 64 if probe["batch64"]["peak_allocated"] < 0.55 * tot else 32
        probe["chosen_batch"] = chosen
        probe["rule"] = "batch64 if peak_allocated < 0.55 total else 32; probe weights discarded; init reloaded for formal training"
        wj(OUT / "TRAIN_PROBE.json", probe)
        print("[probe]", json.dumps({k: v for k, v in probe.items() if k != "rule"}))
    if stage in ("train", "all"):
        batch = json.load(open(OUT / "TRAIN_PROBE.json"))["chosen_batch"]
        h_init = param_hash(INIT)
        assert hashlib.sha256(Path(INIT).read_bytes()).hexdigest() == INIT_SHA
        r1 = train_arm("t1", t1m["images"], batch)
        h1 = param_hash(r1["last_pt"])
        assert h1 != h_init, "T1 params unchanged — training did not update"
        steps = count_steps(len(t1m["images"]), batch)
        wj(OUT / "TRAIN_LOG_T1.json", dict(**r1, init_sha256=INIT_SHA, init_param_hash=h_init,
                                            final_param_hash=h1, updated=h1 != h_init,
                                            optimizer_steps_estimate=steps,
                                            train_images=len(t1m["images"]),
                                            note="last.pt = final-epoch EMA weights (ultralytics); no AUROC selection"))
        r2 = train_arm("t2", t2m["t2_train_images"], batch)
        h2 = param_hash(r2["last_pt"])
        assert h2 != h_init, "T2 params unchanged"
        wj(OUT / "TRAIN_LOG_T2.json", dict(**r2, init_sha256=INIT_SHA, init_param_hash=h_init,
                                            final_param_hash=h2, updated=h2 != h_init,
                                            optimizer_steps_estimate=steps,
                                            train_images=len(t2m["t2_train_images"]),
                                            exposed_images=len(t2m["exposed_images"]),
                                            note="equal step count with T1 asserted via equal image count and batch"))
        assert r1["batch"] == r2["batch"] and steps == count_steps(len(t2m["t2_train_images"]), batch)
        wj(OUT / "MODEL_IDENTITY.json", dict(
            init_weights=INIT, init_sha256=INIT_SHA, init_param_hash=h_init,
            t0_identity=dict(weights=INIT, param_hash=h_init, note="T0 = untouched COCO init, no CPLID training"),
            t1_last_pt=r1["last_pt"], t1_last_sha256=r1["last_sha256"], t1_param_hash=h1,
            t2_last_pt=r2["last_pt"], t2_last_sha256=r2["last_sha256"], t2_param_hash=h2,
            t1_t2_init_param_hash_equal=True, batch=batch, steps_per_arm=steps))
        print("[train] done batch=%d steps=%d" % (batch, steps))
    if stage == "identity":
        print(open(OUT / "MODEL_IDENTITY.json").read()[:400])


if __name__ == "__main__":
    main()

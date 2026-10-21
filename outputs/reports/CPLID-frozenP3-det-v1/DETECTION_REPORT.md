# CPLID strict frozen-P3 detector

One class: insulator. Normal-only detection training; no defective image was used in training or checkpoint selection.
Original group split reused with SHA256-verified path relocation. Image counts differ from source-group counts.

| Split | Images | GT insulator boxes | best mAP50 | best mAP50:95 |
|---|---:|---:|---:|---:|
| val | 61 | 95 | 0.9693 | 0.8305 |
| test | 120 | 220 | 0.9753 | 0.8091 |

Training recipe: {"epochs": 60, "imgsz": 1280, "batch": 8, "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005, "warmup_epochs": 3, "seed": 0, "patience": 0, "amp": false, "freeze": 5, "channels_last": false, "compile": false, "workers": 8, "mosaic": 1.0, "close_mosaic": 10, "mixup": 0.0, "copy_paste": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5, "fliplr": 0.5, "flipud": 0.0, "device": 0, "cache": false, "deterministic": true, "val": true, "plots": true, "save": true, "save_period": 10, "verbose": true}
Both serialized best.pt and last.pt preserve original COCO modules0–4 including BN exactly; fixed-input P3 checks passed.
P4 entry updates were verified, so only preserving the prefix is not confused with no training.
Validation selects best checkpoint; normal test is evaluated only afterwards, without selecting thresholds or checkpoint.
This is localization training, not a defect-class detector, joint anomaly training, shared-ROI evaluation or AUROC experiment.
Old CPLID T1/T2 used different recipe and full fine-tuning; not a controlled matched comparison
CPLID faulty images are synthetic and source identity may be unknown; no unverified source-independence claim.

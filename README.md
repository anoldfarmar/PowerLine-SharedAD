# PowerLine-SharedAD

**Shared-backbone detection and few-shot anomaly scoring for power-line assets.**

PowerLine-SharedAD studies whether one YOLOv8n forward pass can both locate an
asset and supply local features for normal-reference anomaly detection. The
detector starts from COCO weights. Modules 0–4, including the 64-channel,
stride-8 backbone P3 output and their batch-normalization statistics, remain
frozen while the later detection layers adapt to the target assets. At inference,
predicted boxes index features from that same forward pass. Each ROI is matched
to a memory bank built from normal support objects; the image/object score is
the mean of the **eight largest** patch-to-bank nearest-neighbor distances.

The anomaly branch is a frozen-feature memory method. “End-to-end” here means
the complete inference pipeline, not joint training of detection and anomaly
losses.

## Current evidence

| Dataset and setting | Result | Boundary |
|---|---|---|
| InsPLAD, 10 normal supports, 7 asset–fault pairs | COCO YOLO P3 + top8 macro image AUROC **0.9159**, versus 0.8306 for the recorded WRN PatchCore baseline | Pre-cropped images; support frames are not proven independent of detector training frames |
| InsPLAD frozen-prefix detector | AP50 **0.9594**, AP50:95 **0.8215** | Historical full-tuning reference used a different training recipe |
| InsPLAD small-crop shared pipeline | 13.665 → **9.531 ms** mean per input; batch-16 throughput 122.86 → **203.83 images/s** | Existing crops, not original full UAV frames |
| CPLID normal-image detector | Validation AP50 **0.9693**, AP50:95 **0.8305** | One insulator class; defect images were excluded from training and model selection |
| CPLID full-frame two-stage R1 vs shared R2 | Mean 18.554 → **15.177 ms** (1.222× speedup); conditional object AUROC **0.7056 → 0.6800** | Sharing saves compute but lowers mean AUROC by 0.0256 across five reference selections |

CPLID R0, which uses ground-truth boxes and a fresh crop feature pass, has
mean AUROC 0.7287. Its faulty images are synthetic; common foreground or
background ancestry has not been fully resolved. The five references are
different bank samples on one query set, not five independent test sets.
Localization coverage and thresholded fault recall are distinct: the latter
has not been calibrated. These findings do not establish universal accuracy
preservation or RK3588 performance.

## Method and experiments

```text
Full UAV image -> frozen-P3 YOLO detector -> predicted asset boxes
                  |                           |
                  |                           +-> R1: native crop -> resize 320
                  |                                  -> extra COCO prefix -> crop bank
                  +-> P3 from this forward -> R2: ROIAlign 40x40 -> ROI bank

Each query patch -> exact nearest normal prototype -> mean largest 8 distances
```

R0 uses ground-truth boxes and crop features as a diagnostic reference. R1 and
R2 use the same detector and predicted boxes. Their support objects are
ground-truth anchored, and each feature path has its own matching bank.

## Repository layout

| Path | Purpose |
|---|---|
| `scripts/cplid_*.py`, `scripts/insplad_*.py` | Reproducible experiment entry points, feature extraction, detector audits |
| `scripts/m3_common.py`, `scripts/m3score_pcverify.py` | Historical utility modules still imported by the current pipelines |
| `scripts/m3share_speed*.py` | InsPLAD small-crop pipeline cost comparison |
| `scripts/start_task.py`, `accept_check.py`, `log_run.py` | Frozen experiment activation, acceptance, registration |
| `tests/` | CPU and numerical regression tests |
| `docs/cplid/`, `docs/insplad/` | Dataset-specific runbooks and scope notes |
| `outputs/reports/` | Versioned reports, metrics and lightweight evidence |
| `plan/`, `registry/` | Historical task contracts and experiment provenance |

Only obsolete M3 investigation entry points were removed from the current
working tree. They remain recoverable from Git history. Historical contracts
and reports are retained as provenance; some older contracts require a
historical checkout to rerun.

## Reproduce the recorded pipeline

The scripts were run on a Linux CUDA server with Python, PyTorch, Ultralytics,
torchvision, anomalib, NumPy and the original datasets and weights at paths
documented in each frozen run. Large datasets, model weights and caches are
not part of this checkout. Verify their identities before running.

```bash
cd /root/auto-exp
bash scripts/cplid_frozen_p3_launch.sh
bash scripts/cplid_shared_p3_launch.sh
```

Each launcher uses the repository's task and gate workflow. Read the runbooks
and verify the required prior outputs before starting a run:

- [CPLID frozen-P3 detector](docs/cplid/CPLID_FROZEN_P3.md)
- [CPLID full-frame R0/R1/R2](docs/cplid/CPLID_SHARED_P3.md)
- [InsPLAD frozen detector](docs/insplad/INSPLAD_FROZEN_P3.md)
- [InsPLAD shared ROI](docs/insplad/INSPLAD_SHARED_ROI.md)
- [InsPLAD small-crop speed](docs/insplad/M3SHARE_SPEED.md)

Run the lightweight tests with `python -m unittest discover -s tests -p 'test_*.py'`.
GPU-backed experiments also run their own frozen correctness gates before
scoring. Detailed metrics and limitations are in
[CPLID R0/R1/R2](outputs/reports/CPLID-sharedP3-R0R1R2-v1/REPORT.md),
[CPLID detection](outputs/reports/CPLID-frozenP3-det-v1/DETECTION_REPORT.md),
and [InsPLAD small-crop speed](outputs/reports/M3SHARE-speed-v1/SPEED_REPORT.md).

The [Chinese research summary](README.zh-CN.md) contains the full chronology.

## Retained experiment outputs

Obsolete intermediate results were removed on 2026-10-27. COCO + top8,
YOLO + top8, PatchCore, shared-backbone results, and required frozen/audit
evidence are retained. See the [output index](outputs/README.md) for the
keep/delete policy and Git recovery instructions. Historical scripts may
require restoring their old outputs. No algorithm or evaluation scores changed.

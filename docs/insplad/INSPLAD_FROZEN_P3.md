# InsPLAD strict frozen-P3 detection experiment

User-approved target classes: C01-C12, C17, C18 (14 classes). C13-C16 excluded.
Only InsPLAD detection images and original COCO-format train/val annotations
are used. Unambiguous train/val image files remain in their original split;
non-target annotations are omitted and background-only images are retained.
Original labels and images are never edited.
Each retained image is a symlink. The train JSON contains 46 filename groups
with multiple image IDs and incompatible annotations. These ambiguous files
are excluded from the derived training set rather than overwriting/merging
labels. Details and affected annotation counts are saved in DATA_AUDIT.json.
New labels map the selected original IDs to contiguous IDs 0-13.
Sphere (C18) has 26 training boxes and zero validation boxes in the original
split; 15 training boxes remain after ambiguous-file exclusion. It remains a
training target; validation AP/recall is N/A. Overall mAP
covers the 13 selected classes with validation GT, explicitly recorded in results.

Train from the local original COCO YOLOv8n checkpoint. Freeze modules 0-4,
including P3 output module4, all parameters and all BN buffers. Train subsequent
P4/P5/neck/head modules for 60 epochs, imgsz1280, batch8, AdamW lr.001, seed0,
FP32 / non-TF32. Mosaic/augmentation details are frozen in FROZEN_INPUTS.json.
No Adapter is included in this first experiment.

The installed Ultralytics already freezes BN of named frozen modules in
_model_train(). Our subclass explicitly reasserts the prefix's eval/gradient
state before batches and audits live+EMA prefix identity every epoch. It
also verifies downstream parameter updates and the actual serialized best.pt
and last.pt P3 outputs against original COCO weights on fixed real GT crops.

CPU-only preparation (assistant may execute):

```bash
cd /root/auto-exp
OMP_NUM_THREADS=4 /root/miniconda3/bin/python scripts/insplad_frozen_p3.py --mode prepare
```

User starts training:

```bash
cd /root/auto-exp
nohup bash scripts/insplad_frozen_p3_launch.sh >> /root/autodl-tmp/version10-12/InsPLAD-frozenP3-14class-v1.log 2>&1 &
tail -f /root/autodl-tmp/version10-12/InsPLAD-frozenP3-14class-v1.log
```

If interrupted during training, resume with the same command plus --resume.
If only final evaluation was interrupted, run the Python entry with --mode
evaluate (same environment and frozen defaults). After completed training,
Ultralytics strips optimizer state; do not attempt to resume a completed run.
No code/recipe/data changes are allowed within an existing frozen directory.

Result directory:
`/root/autodl-tmp/version10-12/InsPLAD-frozenP3-14class-v1`

- runs/frozen_p3/weights/best.pt and last.pt: detector weights.
- runs/frozen_p3/results.csv and plots: training/validation history.
- runs/frozen_p3/freeze_audit/: per-epoch hashes, P3 differences, update checks.
- FREEZE_CORRECTNESS.json: actual serialized checkpoint verification.
- DETECTION_RESULTS.json and DETECTION_REPORT.md: localization comparison.
- dataset/: remapped 14-class labels; historical_dataset/: original 18-class IDs
  for the same 14 targets, enabling correct evaluation of the historical model.

After training, both the frozen detector and the existing 18-class full-tuned
detector are evaluated on the same val images and 14 target categories. The
historical predictions are explicitly filtered and mapped by original class ID.
Report mAP50, mAP50:95, per-class AP/recall and fixed-confidence (.25), IoU.5
recall in input-size bins. Small means resized GT area <32² at the declared
input size; this is not official COCO AP-small. The historical detector used a
different training recipe/head class count, so this is a descriptive reference,
not a controlled same-recipe freeze-vs-full ablation. Only one new detector is trained.

This stage verifies localization and preservation for identical inputs. Shared
full-image P2/P3 ROI anomaly inference and its AUROC require a separate experiment;
the existing crop memory banks cannot be reused directly for shared ROI features.
No anomaly supports/queries are changed by these scripts. The original detection
train/val split is retained, and source-frame isolation from anomaly test sets
has not been established. Exact train/val duplicate hashes are audited separately.

C01/C03/C17 have no positive anomaly test examples in current root anomaly
folders; C18 has no anomaly folder. They can be detector targets, but no anomaly
AUROC can be claimed for them without an appropriate frozen positive test set.

Runtime activation, acceptance and PASS registration are performed by the driver
only when the user launches training/evaluation. Preparation does not start a GPU run.

## Completed-run reporting recovery

The original exporter did not serialize NumPy integer summaries. The first
recovery also rejected valid non-target C13-C16 outputs from the old 18-class
head. Both reporting issues are handled by the recovery entry without changing
the original three frozen training files. With both prediction JSON files saved,
finish on CPU using:

```bash
cd /root/auto-exp
OMP_NUM_THREADS=4 /root/miniconda3/bin/python scripts/insplad_frozen_p3_recover.py --from-predictions
```

This explicitly filters non-target classes only for the old head. Exported
IDs 13/14 remain C17/C18 in the new 14-class head. It verifies stored checkpoint
identities, image bindings and derived labels, then evaluates both cached
predictions under common COCO AP (maxDets100). Native YOLO AP is kept separately
where available; the two metric definitions must not be mixed in one comparison.
The result includes all validation GTs, including objects missed by either model.

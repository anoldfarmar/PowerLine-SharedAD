# InsPLAD strict frozen-P3 detector — completed from cached predictions

Both methods use the same COCO AP protocol (maxDets=100), recomputed on CPU from saved prediction JSON. No new training or model inference.
Prediction JSON has rounded boxes/scores; these COCO metrics are distinct from native YOLO AP. Available native metrics are preserved separately in JSON.
Training targets: 14; validation AP covers 13 classes with GT. C18 sphere has no validation GT.

| Method | COCO AP50 | COCO AP50:95 |
|---|---:|---:|
| frozen_p3 | 0.9594 | 0.8215 |
| historical_full | 0.9557 | 0.8177 |

Frozen−historical AP50:95 = +0.38 percentage points.

| Class | Frozen AP50:95 | Historical AP50:95 | Frozen recall@.25 | Historical recall@.25 | Small GT | Frozen small recall | Historical small recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| yoke | 0.8253 | 0.8379 | 0.9403 | 0.9686 | 0 | N/A | N/A |
| yoke suspension | 0.8650 | 0.8719 | 0.9904 | 0.9912 | 0 | N/A | N/A |
| spacer | 0.4317 | 0.4019 | 0.5909 | 0.7727 | 0 | N/A | N/A |
| stockbridge damper | 0.8570 | 0.8662 | 0.9785 | 0.9817 | 0 | N/A | N/A |
| lightning rod shackle | 0.6460 | 0.6245 | 0.9200 | 0.9600 | 0 | N/A | N/A |
| lightning rod suspension | 0.9304 | 0.9339 | 1.0000 | 1.0000 | 0 | N/A | N/A |
| polymer insulator | 0.9337 | 0.9367 | 0.9895 | 0.9965 | 0 | N/A | N/A |
| glass insulator | 0.8522 | 0.8497 | 0.9637 | 0.9605 | 0 | N/A | N/A |
| tower id plate | 0.9947 | 0.9863 | 1.0000 | 1.0000 | 0 | N/A | N/A |
| vari-grip | 0.9723 | 0.9673 | 1.0000 | 1.0000 | 0 | N/A | N/A |
| polymer insulator lower shackle | 0.6617 | 0.6668 | 0.8560 | 0.8665 | 7 | 0.4286 | 0.7143 |
| polymer insulator upper shackle | 0.8747 | 0.8649 | 0.9788 | 0.9788 | 0 | N/A | N/A |
| spiral damper | 0.8347 | 0.8227 | 0.9947 | 1.0000 | 0 | N/A | N/A |
| sphere | N/A | N/A | N/A | N/A | 0 | N/A | N/A |

Recall uses fixed confidence .25 and IoU .5. Small bins use area <32² after uniform resize to the declared detector input, not COCO AP-small.
Historical model outputs for excluded C13-C16 are ignored in this selected-target evaluation. Unknown/out-of-range class IDs remain errors.
Historical 18-class fine-tune evaluated on same 14 target categories; not a matched same-recipe 14-class training ablation
Both best and last checkpoints passed prefix parameter/BN identity and same-input P3 tests. This does not establish shared-ROI anomaly AUROC.

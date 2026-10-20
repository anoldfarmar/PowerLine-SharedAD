# InsPLAD small-image shared P3 A/B/C

10N / ref3001 only. AUROC; one support bank per asset and arm, no training.

| Asset / fault | A: original COCO top8 | B: full-image ROI | C: predicted ROI | C−A (pp) |
|---|---:|---:|---:|---:|
| glass-insulator/missing-cap | 0.9026 | 0.9026 | 0.9129 | +1.03 |
| lightning-rod-suspension/rust | 0.9755 | 0.9755 | 0.9717 | -0.37 |
| polymer-insulator-upper-shackle/corrosion | 0.8923 | 0.8923 | 0.8973 | +0.50 |
| polymer-insulator-upper-shackle/rust | 0.8663 | 0.8663 | 0.8635 | -0.27 |
| vari-grip/bird-nest | 0.9256 | 0.9256 | 0.9139 | -1.17 |
| vari-grip/rust | 0.8738 | 0.8738 | 0.8911 | +1.73 |
| yoke-suspension/rust | 0.9541 | 0.9541 | 0.9388 | -1.52 |
| MACRO (7 pairs) | 0.9129 | 0.9129 | 0.9128 | -0.01 |

A is re-scored from the identical frozen COCO prefix and checked against all historical per-image scores.
B is an identity/control under aligned=True, sampling_ratio=1, full 320 ROI and output40.
C uses the highest-confidence prediction of the known asset class; class comes from the benchmark asset, not the fault label.
This is class-conditional component diagnosis, not autonomous unknown-class end-to-end recognition.
Missing/invalid expected-class detections fall back to the full image for BOTH supports and queries; none are dropped.
Inspect DETECTION_SUMMARY.json and detections/ to distinguish real ROI use from fallback.
No bbox GT: no detection mAP, localization accuracy or detection recall claim on these crops.
320-square crop input differs from detector training1280/LetterBox; detection domain/scale shift is a limitation.
Small pre-crops do not establish performance on original1920x1080 scenes. One ref is not a robustness/significance study.
All arms have channel-L2 P3, 1600-patch output, normal-only KCenter .1 seed7001 and exact Euclidean top8.

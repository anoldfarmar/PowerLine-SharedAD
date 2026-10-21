# InsPLAD COCO + top8 full comparison

Status: COMPLETE; COCO 70/70, total 210/210 configurations.

AUROC mean ± population std across ref3001–3005. Same crop files, query labels and normal support lists at each budget.

## 5N

| Combination | COCO + top8 | Trained YOLO + top8 | WRN PatchCore | COCO−YOLO (pp) | COCO−WRN (pp) |
|---|---:|---:|---:|---:|---:|
| glass-insulator__missing-cap | 0.8632 ± 0.0142 | 0.8063 ± 0.0278 | 0.8204 ± 0.0123 | +5.69 | +4.28 |
| lightning-rod-suspension__rust | 0.9603 ± 0.0222 | 0.8778 ± 0.0418 | 0.7413 ± 0.0884 | +8.26 | +21.90 |
| polymer-insulator-upper-shackle__corrosion | 0.8741 ± 0.0107 | 0.8350 ± 0.0281 | 0.7969 ± 0.0456 | +3.91 | +7.72 |
| polymer-insulator-upper-shackle__rust | 0.8519 ± 0.0127 | 0.7982 ± 0.0521 | 0.7222 ± 0.0504 | +5.37 | +12.97 |
| vari-grip__bird-nest | 0.9022 ± 0.0200 | 0.8673 ± 0.0277 | 0.9087 ± 0.0247 | +3.48 | -0.66 |
| vari-grip__rust | 0.8541 ± 0.0342 | 0.8104 ± 0.0457 | 0.7635 ± 0.0449 | +4.37 | +9.06 |
| yoke-suspension__rust | 0.9303 ± 0.0105 | 0.8734 ± 0.0511 | 0.7922 ± 0.0563 | +5.69 | +13.81 |
| MACRO | 0.8909 ± 0.0115 | 0.8383 ± 0.0113 | 0.7922 ± 0.0192 | +5.25 | +9.87 |

## 10N

| Combination | COCO + top8 | Trained YOLO + top8 | WRN PatchCore | COCO−YOLO (pp) | COCO−WRN (pp) |
|---|---:|---:|---:|---:|---:|
| glass-insulator__missing-cap | 0.8897 ± 0.0124 | 0.8311 ± 0.0121 | 0.8258 ± 0.0134 | +5.86 | +6.39 |
| lightning-rod-suspension__rust | 0.9774 ± 0.0031 | 0.9206 ± 0.0150 | 0.7995 ± 0.0957 | +5.68 | +17.79 |
| polymer-insulator-upper-shackle__corrosion | 0.8937 ± 0.0040 | 0.8598 ± 0.0118 | 0.8342 ± 0.0274 | +3.39 | +5.96 |
| polymer-insulator-upper-shackle__rust | 0.8858 ± 0.0133 | 0.8227 ± 0.0180 | 0.7654 ± 0.0419 | +6.30 | +12.03 |
| vari-grip__bird-nest | 0.9219 ± 0.0114 | 0.9018 ± 0.0182 | 0.9361 ± 0.0163 | +2.01 | -1.42 |
| vari-grip__rust | 0.8886 ± 0.0163 | 0.8545 ± 0.0207 | 0.8186 ± 0.0364 | +3.41 | +7.00 |
| yoke-suspension__rust | 0.9541 ± 0.0073 | 0.9172 ± 0.0118 | 0.8345 ± 0.0338 | +3.69 | +11.96 |
| MACRO | 0.9159 ± 0.0045 | 0.8725 ± 0.0099 | 0.8306 ± 0.0215 | +4.33 | +8.53 |

Five references are different normal memory banks on the same test set, not five independent datasets or detector trainings.
5N/10N count crop files, not independent source frames. No test-set threshold selection, significance or pixel-localization claim.
COCO is a generic pretrained baseline; trained-YOLO training exposure remains a separate audit question.
WRN uses its frozen official PatchCore image score; YOLO arms use raw P3 normalized features and top8.

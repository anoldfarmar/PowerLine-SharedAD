# CPLID detector supplemental acceptance

PASS: actual best/last SHA, exact prefix including BN, 60 epoch live/EMA evidence, CPU whole-image/crop P3 probes.

| Split | mAP50 | mAP50:95 | fixed P@.25 | fixed R@.25 | TP / GT |
|---|---:|---:|---:|---:|---:|
| normal val | 0.969346 | 0.830455 | 0.883495 | 0.957895 | 91 / 95 |
| normal test | 0.975266 | 0.809145 | 0.907173 | 0.977273 | 215 / 220 |

Native Ultralytics AP; fixed P/R use cached predictions and confidence-descending one-to-one IoU>=.5 matching.
Native summary P/R use its own operating point and must not be called P/R@.25.
All missed GT remain in the recall denominator. No anomaly performance claim follows from detector acceptance.
GT green / prediction red: /root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1/acceptance/vis

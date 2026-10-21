# Frozen CPLID R0/R1/R2 protocol

| Arm | Query | Normal support bank |
|---|---|---|
| R0 | GT native crop, Resize320, COCO P3 | same10 GT crops |
| R1 | predicted native crop, Resize320, extra COCO prefix | same10 GT crops |
| R2 | same prediction, same-forward full-frame P3, ROI40 | same10 GT ROIs from full-frame P3 |

GT-anchored support construction is deliberate and frozen: no supports dropped/reselected because detector misses.
R1 support geometry is GT, query geometry is predicted; the comparison isolates query localization and full-frame representation, not a fully automatic support selection system.
Five original ref3001–3005 lists; 15 arm/reference evaluations, NOT15 detector trainings or independent datasets.
Full images use1280 square LetterBox, native crops320; layer4 C64 stride8 only. Exact settings: PROTOCOL.json.
No-box: unlocalized, score null, no fallback. Missed faulty GT remain in localization coverage denominator.
Unmatched predictions count as FP and have unknown anomaly label. Do not label every box on a faulty frame as faulty.
AUROC is per GT object, conditional on matching for R1/R2; R0 all-GT and same-matched-subset both reported.
No test-derived alarm threshold or whole-image AUROC. Fault localization coverage is NOT fault diagnosis recall.
Timing: ref3001 only, all queries, three alternating rounds plus separate profiling; batch1 sequential throughput.
R0 is not a complete-system speed baseline. R1/R2 run actual detector and retrieval, no feature cache during timing.
GT crops use boxes (floor/ceil clipping); regression gate checks different boxes on same image give different features.
Source groups only establish exact-duplicate isolation, not proven capture independence; faulty images are synthetic.
No training, adapters, P2, score changes or tuning. GPU execution only on explicit user launch.

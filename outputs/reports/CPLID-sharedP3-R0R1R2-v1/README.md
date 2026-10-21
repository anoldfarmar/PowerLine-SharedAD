# CPLID-sharedP3-R0R1R2-v1

Completed detector: `/root/autodl-tmp/version10-12/CPLID-frozenP3-det-v1`.
Next-stage output: `/root/autodl-tmp/version10-12/CPLID-sharedP3-R0R1R2-v1`.

## Detector acceptance

`cplid_detector_accept.py` rechecks actual best/last SHA, exact prefix parameters and
BN, 60-epoch audit evidence, whole-image LetterBox and GT-crop P3 outputs on normal
training probes. It computes confidence .25 / IoU .5 recall from existing normal
validation/test prediction JSON, retaining missed GT in the denominator. It never
runs fault-image inference. See `DETECTOR_ACCEPTANCE.json/.md` in detector output.

## Frozen comparison

| Arm | Query path | Normal support path |
|---|---|---|
| R0 | GT native crop → Resize320 → original COCO P3 | preselected GT crops |
| R1 | new detector → predicted native crop → Resize320 → extra COCO prefix | same GT crops |
| R2 | same detector/predictions → same-forward P3 → RoIAlign40 | GT ROIs from full-frame P3 |

Support construction is deliberately GT-anchored in all arms. This is not a claim
that normal-bank construction is annotation-free. R0/R1 have identical support
features and share the newly verified crop bank; R2 builds a distinct ROI bank.
No historical CPLID feature cache, bank or anomaly score is used.

Five verified original lists, 10 files/groups/preselected normal targets each;
15 arm/reference evaluations, one already-trained detector. Not15 trainings.
All368 query frames (120 held-out normal,248 synthetic faulty),468 GT objects.
Detector1280 square LetterBox, FP32/noTF32, layer4 only; crop320/P3 40×40×64;
ROI40×40, aligned=True, sampling_ratio=1. Exact1NN Euclidean/top8-largest mean,
official KCenterGreedy10%, seed7001. Settings and input/code/weight hashes are in
`FROZEN_INPUTS.json` and `PROTOCOL.json`.

No-box returns `unlocalized`, no fallback, no anomaly score. Unmatched predictions
are scored and counted as FP but have no assigned anomaly label. R1/R2 AUROC is
conditional on the identical matched GT subset; R0 also reports this subset.
All GT, including missed faults, remain in localization coverage denominators.
No test-derived alarm threshold, whole-image AUROC or calibrated fault recall.
Synthetic source ancestry remains unknown beyond the exact-duplicate audit.

## Launch (user runs CUDA)

```bash
cd /root/auto-exp
nohup bash scripts/cplid_shared_p3_launch.sh > /root/autodl-tmp/version10-12/CPLID-sharedP3-R0R1R2-v1/run.log 2>&1 < /dev/null &
```

```bash
tail -f /root/autodl-tmp/version10-12/CPLID-sharedP3-R0R1R2-v1/run.log
```

Uses configured `/root/miniconda3/bin/python`; does not install packages, download
weights, retrain, add layers or search parameters. GPU must be otherwise idle.
Default wall-time safety cap2h; verified feature/bank caches can resume.
Formal GPU gates execute before any fault scoring. They test geometry, actual
crop bbox use, batch1/crop32, batched ROI, chunked/direct float64 distances,
repeated coreset/build/scores, no-box/multi-box/boundary/portrait paths, state
hashes, and actual forward counts. Failure stops rather than loosening tolerance.

Speed uses ref3001 matching banks, all frames, five warmup frames, three alternating
rounds, frame batch1, multi-crop batch32. Disk/decode separate. A separate stage
profile pass is explanatory, not summed into independent end-to-end measurements.
Reported throughput is sequential batch1, NOT larger-frame-batch throughput.
Partial timing is not silently appended/reused as a complete uninterrupted run:
inspect and preserve it before explicit retiming. Verified feature caches remain.

After success: `REPORT.md`, `per_reference_metrics.csv`, `raw_scores/`,
`COVERAGE.json`, `SPEED_METRICS.json`, `TIMING_RAW.jsonl`, `STAGE_PROFILE.json`,
`BANK_MANIFEST.json`, `CACHE_MANIFEST.json`, `REPRO_AND_CORRECTNESS.json`,
`RESOURCE_COST.json`. Launch activates the task, mechanically accepts and registers
PASS only after successful completion. Send completion log for report review and
scoped Git synchronization; raw images, weights and caches never enter Git.

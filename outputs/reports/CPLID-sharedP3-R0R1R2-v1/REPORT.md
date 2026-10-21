# CPLID R0/R1/R2 results

| Arm | mean AUROC | std(ddof=0) | min–max | paired mean |
|---|---:|---:|---:|---:|
| R0 | 0.728735 | 0.021225 | 0.698479–0.752933 | 0.725900 |
| R1 | 0.705568 | 0.020261 | 0.673893–0.725863 | 0.705568 |
| R2 | 0.679961 | 0.035901 | 0.626979–0.729164 | 0.679961 |

R2−R1 paired reference differences: [0.05527006751687935, -0.0560765191297824, -0.01369092273068273, -0.08790322580645171, -0.02563765941485363]; mean -0.025608; positive/negative/zero 1/4/0.
Localization coverage counts (not alarm recall): {'GT': 468, 'fault_GT': 248, 'predictions': 497, 'matched': 463, 'matched_fault': 248, 'FP': 34}; no-box frames 0.

| Timing scope | R1 mean ms | R2 mean ms | full speedup |
|---|---:|---:|---:|
| all | 18.554 | 15.177 | 1.222x |
| no_box | empty | empty | — |
| 1 | 17.959 | 14.943 | 1.202x |
| 2-4 | 20.485 | 15.965 | 1.283x |
| 5+ | 29.551 | 17.384 | 1.700x |
| located | 18.554 | 15.177 | 1.222x |

Diagnostic-only speedup 3.222x; this is NOT full-system speedup.
P95, medians and sequential batch1 throughput: SPEED_METRICS.json. No larger-frame-batch throughput measured.
R1/R2 actual forward counters in TIMING_RAW.jsonl; R2 has no extra crop prefix. R0 is not system timing baseline.
Normal support objects GT-anchored for all arms. Conditional AUROC excludes unmatched GT but coverage retains them.
No-box is unlocalized, never normal; no alarm threshold selected; no whole-image AUROC or pixel localization claim.
Synthetic defects and unresolved source ancestry limit generalization/causal claims; five references are not independent datasets.
Interpret next steps from accuracy loss and object P3 sizes in COVERAGE.json; do not automatically add P2/adapters/training.

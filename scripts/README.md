# Script entry points

The root `scripts/` path remains stable because frozen experiment contracts,
source hashes, launchers and result manifests refer to it. Current experiment
files are grouped by filename prefix:

- `cplid_*`: full-frame insulator detection and R0/R1/R2 anomaly comparison.
- `insplad_*`: frozen-prefix detector and crop-based shared-ROI studies.
- `m3share_speed*`: retained small-crop cost study used by the current README.
- `m3_common.py`, `m3score_pcverify.py`: retained compatibility dependencies.
- `start_task.py`, `accept_check.py`, `log_run.py`: workflow controls.

Tests live in `../tests/`; protocol notes live in `../docs/`. Historical M3
investigation scripts were removed from this thesis checkout. Their published
results remain in `outputs/reports/` and their source remains in Git history.

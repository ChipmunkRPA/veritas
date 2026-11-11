# Matched JPEG localization on the unopened BG-REAL pairs

Use the 12 independently qualified v262 pairs with publisher-bound masks. The comparison reuses the exact v229 JPEG recipes, current and sensitive readouts, shared mask, scoring weights, source-pixel truth, and paired group statistics. All previously scored v229 inputs stay outside this run. The new panel includes ten publisher training and two validation photos, so it measures transfer to unused local source groups within the same LaMa editing family. [Qualified intake](evaluation-panels/image-bg-real-backup-intake-v262-complete.json) · [Inherited comparison](image-bg-real-jpeg-v229-method.md)

Use original, AI edit, conventional tone, conventional blur and ordinary copy-move views at qualities 75 and 90. Each arm has 120 unique inputs across both qualities and repeats its first input once per quality. The 240 planned unique model calls and four repeats run sequentially. For each arm and quality the fixed three batches have 25, 25 and 10 views. The inherited monitor enforces the same RAM, disk and 900-second limits. Keep ordinary false highlights, missed localization and all unavailable inputs visible. The model-free final seal requires every producer and readout receipt. No policy is fitted or applied to the serving path.

Run from the repository root with the existing environment. Create a new empty private map directory on the same Seagate archive as the v262 cache, named `image-bg-real-backup-jpeg-v264-maps`. Preparation binds the exact 120 encoded JPEG inputs and truth before either arm runs. Review and record the public freeze before scoring. Keep raw maps, private source identities and native truth masks local.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v264() { backend/.venv/bin/python -B -m scripts.image_bg_real_backup_jpeg_v264 "$@"; }
run_v264 preflight
run_v264 prepare
run_v264 freeze
for quality in 75 90; do
  for phase in 0 1 2; do
    run_v264 score baseline "$quality" "$phase"
    run_v264 score candidate "$quality" "$phase"
    run_v264 readout "$quality" "$phase"
  done
done
run_v264 seal
```

The exact aggregate is the private trial's `summary.json`. Use `run_v264 coverage` to inspect completed and unverified producer phases after a failure. Record any incomplete run under its original resource receipt.

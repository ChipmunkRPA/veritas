# Paired image localization on the two licensed reference edits

Use the two completed v276 composites and their paired original, tone, blur,
and mirrored copy views. Each 256×256 crop is evaluated as lossless PNG and
JPEG75, with ten views per format. The exact pre-encoding RGB difference mask
is the same truth for both formats. Additional changes observed in the JPEG
decode are reported separately and never counted as AI truth. Conventional
edits retain zero AI truth with their own observed retouch masks.

Both formats run the v264 current fused check and the frozen sensitive
checkpoint with their unchanged image gates, pixel thresholds and support
rules. This isolates transport on identical source groups. The standard PNG
product route can use a different baseline. The shared result is the overlap of
the two selected masks. Report useful localization at IoU ≥ 0.30 and recall
≥ 0.50 and report false highlights separately for each ordinary control. Two
source groups support descriptive results only.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v280() { backend/.venv/bin/python -B -m scripts.image_flux_detector_comparison_v280 "$@"; }
run_v280 preflight
run_v280 prepare
run_v280 freeze
for format in 0 75; do
  run_v280 score baseline "$format" 0
  run_v280 score candidate "$format" 0
  run_v280 readout "$format" 0
done
run_v280 seal
```

The inherited bounded worker uses `0` for PNG and `75` for JPEG75. It runs
40 planned unique model calls and four exact first-input repeats. Use
`run_v280 coverage` after a stop to report verified and unverified views. The
aggregate is written only when all 20 paired views and all required producer
and readout receipts verify. Prepared photos, native truth, and raw maps stay
in the private evaluation cache.

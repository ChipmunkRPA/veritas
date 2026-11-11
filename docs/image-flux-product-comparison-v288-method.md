# Product image localization on six additional licensed photo edits

Use all six v285 composites with their matched original, tone, blur, and
mirrored-copy views. Each native 256×256 crop is tested as PNG and JPEG75.
The pre-encoding, exact native RGB change mask is shared across both formats.
The report counts JPEG changes outside that mask separately. Ordinary edits
retain zero AI truth and have their own observed retouch masks.

The current local PNG route uses the pinned TIDAL CPU float32 checkpoint.
Its 384×384 raw map must have at least 7,993 cells strictly above 0.70 before
the fixed component readout can display a region. The current JPEG75 route
uses the frozen FUSED MPS float32 baseline and its 0.50 image and component
cutoffs. Both routes use the same production reliability, quantization,
component selection, and display cap. The product switches baselines between
formats, so the two current-route results do not isolate compression effects.

The frozen v264 sensitive checkpoint runs on both formats with no fit. Its
area rule requires at least 1% of source cells strictly above 0.90. Its
JPEG75 result is an offline transfer test. The current product keeps
this sensitive check disabled in legacy mode and its adapter accepts PNG only.
The shared mask is the intersection of the two selected source masks in each
format. Report useful localization at IoU ≥ 0.30 and recall ≥ 0.50, and false
highlights separately for each ordinary control. The six groups support
descriptive paired results rather than population estimates.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run_v288() { backend/.venv/bin/python -B -m scripts.image_flux_product_comparison_v288 "$@"; }
run_v288 preflight
run_v288 prepare
run_v288 freeze
for format in 0 75; do
  for phase in 0 1 2; do
    run_v288 score baseline "$format" "$phase"
    run_v288 score candidate "$format" "$phase"
    run_v288 readout "$format" "$phase"
  done
done
run_v288 seal
```

The inherited worker uses `0` for PNG and `75` for JPEG75. It runs three
10-view batches per format and arm, with 120 unique forwards and four exact
first-input repeats. Limits are 900 seconds, 6 GiB initial RAM and sampled
worker-tree RSS, 3 GiB initial disk, and 1.5 GiB running reserves. Use
`run_v288 coverage` after a stop. The aggregate requires all 60 paired views,
including every control and all resource and readout receipts. Source photos,
raw response maps, and masks stay in the private evaluation cache.

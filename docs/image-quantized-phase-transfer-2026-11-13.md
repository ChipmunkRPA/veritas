# Lower-depth resampling evidence in native PNGs

An optional check now identifies a region with larger stored-value steps
and repeating camera-style curvature in a supported 16-bit PNG. It shows
the original crop, source coordinates and measured value step in the
browser, PDF and JSON. This is evidence about local processing. It adds
zero AI confidence. [Prior Bayer prediction experiment](image-bayer-prediction-transfer-2026-11-13.md) ·
[Evidence selection research](https://arxiv.org/html/2607.29156v1) ·
[Quantization estimation research](https://www.sciencedirect.com/science/article/pii/S0923596525001481)

## Frozen test

Keep existing shift and rotation observations first. For a candidate
rectangle supported by the four-grid camera-pattern-loss check, inspect
its native pixels inset by ten pixels on each side. A value step of 256
or 257 must explain at least 95% of the measured values in every color
channel. The normalized horizontal and vertical curvature must retain
opposite red and blue phase in all four parts of the crop. The maximum
eligible generated score in the eight-camera development set was zero.
The held measurement requires strictly positive joint contrast.
No new weight or external code was used. [Four-grid study](image-native-generated-grid-transfer-2026-11-13.md) ·
[Rotation controls](image-native-rotated-stencil-transfer-2026-11-13.md)

On eight different held CC0 camera models, the check explained two of
three ordinary quantized-translation flags left by the earlier research
arm. One of eight authored translations still produced an unexplained
flag. Five received no candidate from the earlier arm. All 16 useful
generated locations from the rotation-explained research comparison
remained, including promoted and feathered outputs. A third set of 13
camera models tested the actual serving code and exact source geometry.
It found nine of 13 quantized translations at the exact changed-pixel
locations, with no quantized explanation on 26 generated controls or 78
other ordinary deliveries. The earlier rotation path separately flagged
11 true rotations and one diagonal translation. The camera models are
disjoint across all three sets. All sources had earlier project exposure
and the generated variants use one editor. [Study measurements](evaluation-panels/image-quantized-phase-transfer-2026-11-13.json) ·
[Native serving transfer](evaluation-panels/image-quantized-phase-serving-2026-11-13.json)

The option applies only in development mode and requires the existing
native interpolation and rotation options. Set
`VERITAS_IMAGE_NATIVE_QUANTIZED_REVIEW=true` for bare, upright 2048 × 1536
16-bit RGB PNGs with a repeatable camera-style pattern. The signed replay
recomputes the result and both image assets from the submitted original.
In an isolated simulated paid flow, one held quantized delivery reported
source columns **[352, 480)** and rows **[320, 448)** with a stored-value
step of **257**. The original, generated control and unsupported 8-bit
file produced no lower-depth finding. [Local app, PDF and export receipt](evaluation-panels/image-quantized-phase-app-2026-11-13.json)

## Reproduce

The standard-library reader checks the published source commitments,
overlap, decisions and counts. The native runner uses the pinned RAW
decoder, CC0 camera files and retained generated images from the source
studies. Two separate runs reproduced each development, held and
serving row. The original method freeze commits the detector source
as it was before the optional serving hook was added. A new execution
binds the current detector source and reproduces the same row data.

```sh
python3 -B -IS backend/scripts/review_image_quantized_phase_transfer.py \
  docs/evaluation-panels/image-quantized-phase-transfer-2026-11-13.json \
  docs/evaluation-panels/image-quantized-phase-serving-2026-11-13.json
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze development held replay-development replay-held; do
  backend/.venv/bin/python -B -m scripts.image_quantized_phase_transfer \
    "$run" "$rawnind" "$pixls" "$shifted_grid_run" "$rotation_run" || exit 1
done
backend/.venv/bin/python -B -m scripts.image_quantized_serving_transfer \
  "$rawnind" "$pixls" "$earlier_stencil_run" \
  docs/evaluation-panels/image-quantized-phase-transfer-2026-11-13.json \
  "$run/serving.json" --expect docs/evaluation-panels/image-quantized-phase-serving-2026-11-13.json
```

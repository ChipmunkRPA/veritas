# Full Bayer reconstruction on small local image edits

This single-upload test asks whether a fuller camera reconstruction model
can localize small generated replacements. It estimates the dominant 2 × 2
sensor phase from the submitted image. At each predicted red or blue pixel,
it compares the value to the average of the adjacent directly sampled sites,
using horizontal, vertical or diagonal neighbors as appropriate. The error
tolerance scales with the neighbors' spread. The check compares 16-pixel
cells with their local median, requires support in both color channels and
reports at most one native rectangle. It uses no learned weights or earlier
copy. This is a test of local processing consistency with zero AI-confidence
weight. [Sensor residual research](https://arxiv.org/html/2608.20558v1) ·
[Localized edit evaluation](https://arxiv.org/html/2602.00192v2)

The 13-camera development panel includes ordinary rotations, blur, tone
and quantized interpolation. The frozen point is the second-highest
ordinary maximum across camera groups. Eight different camera models supply
104 confirmation deliveries. Each includes the untouched photo, conventional
processing, a new prompt relative to development, and sharp, feathered and
linearly promoted generated replacements. Both panels had previous project
exposure and use the same commercially permitted generator. This separates
cameras and prompts for the new cue. The files retain unchanged pixels
outside the known composited patch. [Development source roster](evaluation-panels/image-native-stencil-interpolation-cc0-roster-2026-11-13.json) ·
[Confirmation sources and generator](image-native-generated-grid-transfer-2026-11-13.md)

| Confirmation outcome | Earlier four-grid candidate | Full Bayer prediction at frozen point | Full Bayer top proposal before cutoff |
| --- | ---: | ---: | ---: |
| Generated sharp, useful | 6/8 | 0/8 | 7/8 |
| Generated feathered, useful | 5/8 | 0/8 | 7/8 |
| Generated linear promotion, useful | 7/8 | 0/8 | 8/8 |
| Matched ordinary highlights | 13/64 | 0/64 | — |
| 8-bit and whole-image resize, assessed | 0/16 | 0/16 | — |

A useful location covers at least half the known changed pixels and reaches
0.30 intersection over union. Rotated and quantized ordinary changes scored
above every held generated replacement. The top-box result suggests the
measurement can help propose a location, but lowering the cutoff would
also highlight ordinary edits. It missed its frozen advancement gate and
remains offline. A separate test would need to verify whether an independent
cue can distinguish those ordinary changes. [Per-file measurements](evaluation-panels/image-bayer-prediction-transfer-2026-11-13.json)

## Reproduce

The standard-library reader recounts the operating point, localization
transitions, ordinary flags, coverage and gate. The native runner reproduced
all 234 input and mask commitments and the numerical results in separate
processes. Supply the pinned RAW decoder cache, the CC0 capture directory,
and the retained generated artifacts from the earlier stencil-interpolation
and shifted-grid studies. The source panels contain instructions for
regenerating the latter artifacts on the pinned local runtime.

```sh
python3 -B -IS backend/scripts/review_image_bayer_prediction_transfer.py \
  docs/evaluation-panels/image-bayer-prediction-transfer-2026-11-13.json
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze calibrate confirm replay-development replay; do
  backend/.venv/bin/python -B -m scripts.image_bayer_prediction_transfer \
    "$run" "$rawnind" "$pixls" "$earlier_stencil_run" "$shifted_grid_run" || exit 1
done
```

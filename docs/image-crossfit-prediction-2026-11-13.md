# Blind pixel prediction for small local image edits

Recent inpainting studies document two shortcuts that complicate small-edit
localization. Whole-image generator traces can disappear when the original
background is restored, while segmentation can snap to object contours. This
test uses exact restored-background masks and matches conventional edits at
the same locations. It tests a local texture measurement with no external
model weights or research code. [FUSED](https://arxiv.org/html/2608.28302v1)
· [GAP-SAM](https://arxiv.org/html/2608.20929v1)

At every eight-pixel cell, predict the luma and two chroma values from adjacent
horizontal and vertical pixels. The first arm measures errors for equal weights.
The second fits a regularized horizontal/vertical weighting on one checkerboard
phase and measures errors on the opposite phase, then reverses the roles. It
retains the held error change and orientation agreement. The third arm combines
both. Matched changed-pixel locations, blurred patches and locally recolored
patches enter only the fitting comparison. Each scoring pass receives one image.
The fitted field uses the same selection geometry as the earlier study, comparing
a native-grid rectangle with its immediate surrounding ring.

The 24 fit, 24 operating-point and 60 held sources are disjoint in this study.
All were previously used in other experiments. TGIF/TGIF2 supplies three
generator families and 48 held sources. BG-REAL supplies a second source set of
12. Each photo yields untouched, blurred, recolored, sharp-edited and feathered
deliveries in three encodings. The comparison reuses the earlier study's
format-specific physical check on the exact same input hashes and masks.
[Earlier study and data preparation](image-residual-third-order-transfer-2026-11-13.md)

## Held results

Each row covers 96 edited TGIF/TGIF2 deliveries plus 144 genuine, blur or
recolor controls. An added location must cover at least half the exact changed
pixels with at least 0.30 intersection over union. Separate points for each arm
and delivery came from the second-highest calibration image-group maximum.

| Encoding | Existing physical check, useful | Plus fixed prediction, useful | Plus cross-fitted prediction, useful | Plus joint prediction, useful |
| --- | ---: | ---: | ---: | ---: |
| JPEG75 | 0 | 2 | 1 | 2 |
| JPEG90 | 4 | 5 | 4 | 5 |
| PNG | 0 | 1 | 0 | 1 |

Cross-fitting added 6–8 ordinary flagged deliveries per TGIF/TGIF2 encoding.
The fixed arm added eight per encoding and the joint arm added eight or nine.
Each arm added 2–4 misplaced boxes per encoding. On BG-REAL, no candidate
flagged any of the 108 controls or found any of the 72 edited deliveries.
The fixed arm only ranked a useful BG-REAL rectangle before the decision
point on 3–4 of 24 edited files per encoding. That ranking failure limits
what changing the point alone can recover. No arm advanced. The comparison
measures two physical checks and this proposed signal separately from the
full app. No media or external weights are distributed.

## Replay

Reproduce the public results with the standard library. The reviewer checks
the frozen inputs, notes which exact boxes are usable, and recomputes the
operating points and held outcomes.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_crossfit_prediction \
  docs/evaluation-panels/image-crossfit-prediction-2026-11-13.json.gz \
  docs/evaluation-panels/image-residual-third-order-transfer-2026-11-13.json.gz
```

To regenerate the predictions, use the archived verified TGIF/TGIF2 and
BG-REAL inputs with a new output directory. Source role and input
commitments are checked again before calibration and scoring.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze calibrate score; do
  backend/.venv/bin/python -B -m scripts.image_crossfit_prediction "$phase" \
    "$run" --bg "$bg_plan_root" || exit 1
done
```

# Native color-channel displacement as an image editing cue

The camera-lens check found a measurable field in two of four native photos
and one of four ordinary re-encodes at each JPEG quality. The coverage fell
short of the frozen requirement for scoring the 768-pixel edited panel.
[Aggregate](evaluation-panels/image-chromatic-field-pilot-2026-11-11.json) ·
[Frozen method](evaluation-panels/image-chromatic-field-pilot-freeze-2026-11-11.json)

## Measurement

Lateral chromatic aberration can shift red and blue edges relative to green
by a distance that varies across a camera's field of view. The original
study compares local and global channel registrations and tests JPEG quality
75 on full-resolution camera photographs. This pilot uses local enhanced
correlation alignment and held-out residuals in a smaller CPU procedure.
It uses no learned weights. [Johnson and Farid's primary study](https://farid.berkeley.edu/downloads/publications/acm06c.pdf)

Each native RGB image supplies a 6 × 6 grid of nonoverlapping tiles. A tile
covers one-eighth of the image's shorter side. Color and texture screening
selects near-neutral edges, and bandpass registration estimates the red and
blue displacement relative to green in native pixels. Two checkerboard
folds fit a robust radial slope and a two-dimensional intercept. A measurable
field must predict held tiles better than a translation-only model, with
both fold slopes in the same direction. The exact measurement and coverage
requirements were frozen before decoding the camera captures.

A generated 768 × 576 control with known radial displacements supplied all
36 tiles in both channels. Adding a feathered local displacement to the red
channel raised its held-out target-tile residual from **0.0089** to **1.3480
pixels**. The target ranked first. The authored image verifies the geometry
of an imposed channel shift.

## Camera coverage

The test used the two Sony and two Canon RAW captures qualified in v239.
Their exact publisher lengths and checksums were verified again. The
pinned rawpy decoder rendered the same native RGB profile, with chromatic
correction disabled. Each source contributed the native render and separate
JPEG 90 and JPEG 75 re-encodes with 4:2:0 chroma.
[Camera source and decoder receipts](image-rawnind-phase-v239-method.md) ·
[Institutional RawNIND release](https://doi.org/10.14428/DVN/DEQCIM)

| View | Measurable captures | Planned captures |
| --- | ---: | ---: |
| Native RGB | 2 | 4 |
| JPEG 90 | 1 | 4 |
| JPEG 75 | 1 | 4 |

One Sony and one Canon capture had measurable native fields. Only the Sony
capture retained a measurable field in both JPEG re-encodes. A second run
reproduced every measurement and coverage count exactly, excluding
execution time. The two runs each processed 12 camera views. These are
camera-field coverage counts on original photos and their re-encodes.
Detection accuracy requires an edited-image holdout.

The predeclared transfer required at least three supported native sources
and two supported JPEG 75 sources. The next eligible local comparison would
use 24 v225 sources for development and 12 source-disjoint v262 LaMa object
removal pairs for confirmation, with exact source masks and existing JPEG
comparators. Those archives are ready for a method with stronger coverage.
[Development panel](image-bg-real-jpeg-v229-result.md) ·
[Confirmation panel](image-bg-real-backup-intake-v262-method.md) ·
[Confirmation JPEG comparison](image-bg-real-backup-jpeg-v264-result.md)

## Reproduce locally

Use the exact retained v239 RAW folder and pinned decoder wheel. The source
folder must be supplied as a direct absolute path under the local user's
control. The pilot reads those source files and writes only the new review
records.

```sh
review_dir=$(mktemp -d)
raw_dir=/absolute/path/to/image-rawnind-phase-v239-attempt02
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -m scripts.image_chromatic_field_pilot freeze "$review_dir/frozen.json"
backend/.venv/bin/python -m scripts.image_chromatic_field_pilot run \
  "$raw_dir" "$review_dir/frozen.json" "$review_dir/native.json"
```

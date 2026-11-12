# Image localization under shifted input grids

The previous PNG repeat pass reduced false highlights by checking whether the
same region survived a horizontal flip. This comparison moves the detector's
input grid while keeping every source pixel in its original orientation.
Recent document-forensics work uses displaced patch grids to reduce false
responses at tile boundaries. Here the shifts are 32 pixels, applied
independently to each axis, and the readout uses the existing photo detector.
[Previous flip test](image-regional-flip-transfer-2026-11-11.md) ·
[Document-forensics study](https://arxiv.org/html/2609.38391v1)

## Inputs and method

Reflection supplies the extra top or left pixels. After inference, discard
that margin and apply the existing native-area gate and condition mask to the
unaltered source rectangle. The readout compares each shift alone, their
pixelwise intersection, and original components with at least half their
pixels supported by both offsets at the same location or by either offset.
The two original-component arms preserve the original masks. This design
uses one checkpoint and measures repeatability without adding confidence
weight.

The first two cohorts reproduce 192 previously measured PNG views from 24
distinct photos and credited photographers. They include 48 sharp or
feathered known edits from two editors and 144 matched controls. The exact
original, flip and recompression masks are carried over. The third cohort
uses 30 views of six separately licensed 256-pixel photos with a different
editor, including three interior edits and three edge edits. Original, flip,
and recompression masks for these six source groups are measured again. The
photos had project exposure before this comparison. The ordinary controls
include untouched photos, local tone, blur and copy-move changes. An
original-pixel mask defines every known generated replacement.
[Earlier two-editor panel](image-background-patch-transfer-2026-11-11.md) ·
[Later two-editor panel](image-wavelet-region-transfer-2026-11-11.md) ·
[Separate-editor construction and controls](image-flux-photo-expansion-v285-result.md)

The frozen protocol uses intersection over union of at least 0.30 and recall of
at least 0.50 for a useful localization. The filtering arm must preserve every
useful flip location and reduce ordinary flags by at least 20% in both larger
panels. The direct intersection arm must add useful locations on at least two
photos per panel with no more ordinary flags. The six-photo transfer measures
behavior on another editor and source size.

## Results

The flip check found **4 of 54** known edits and flagged **75 of 168** ordinary
views. Requiring support from both shifted grids found **3 of 54** and flagged
**103 of 168**. The direct intersection added zero useful locations beyond
the flip. Neither method improved the tradeoff, so both stay outside the app.

| Check | Earlier edits /24 | Earlier controls /72 | Later edits /24 | Later controls /72 | FLUX edits /6 | FLUX controls /24 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original orientation | 1 | 47 | 2 | 62 | 1 | 18 |
| Horizontal flip support | 1 | 30 | 2 | 38 | 1 | 7 |
| Recompression | 0 | 11 | 0 | 12 | 0 | 6 |
| Horizontal offset | 0 | 49 | 2 | 55 | 1 | 17 |
| Vertical offset | 1 | 49 | 2 | 60 | 1 | 16 |
| Offset intersection | 0 | 42 | 2 | 55 | 1 | 14 |
| Original components supported by both offsets | 0 | 38 | 2 | 55 | 1 | 10 |
| Original components supported by either offset | 1 | 47 | 2 | 56 | 1 | 12 |

The Mac restart removed temporary masks. Running the preceding comparison
again reproduced its published journal SHA-256 and all outcome counts. The
first grid run then completed 192 views before a call to the wrong encoder
module stopped the FLUX transfer. The corrected call ran the 30 transfer
views. The recovery preserved the first 192 rows byte for byte and committed
both freezes. A separate replay verified all 222 delivered inputs, 444 shifted
inference inputs, 1,776 region measurements, and 576 prior mask matches.
These replays check the saved evidence and source geometry without rerunning
the new neural predictions. [Audited aggregate and frozen method](evaluation-panels/image-grid-origin-transfer-2026-11-11.json)

## Reproduce

From `backend`, supply the two licensed intakes and previous flip run.
The current local model registry provides the pinned weights. The six-image
editor-transfer sources use the sealed paths from the v285 preparation.
Keep the source files and masks in local storage.

```sh
export PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -B -m scripts.image_grid_origin_transfer freeze "$earlier" "$later" "$previous" "$run"
.venv/bin/python -B -m scripts.image_grid_origin_transfer run "$earlier" "$later" "$previous" "$run"
.venv/bin/python -B -m scripts.review_image_grid_origin_transfer "$run" "$earlier" "$later" "$previous"
```

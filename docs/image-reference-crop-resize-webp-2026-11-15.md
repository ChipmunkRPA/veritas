# Compare a cropped and downsized WebP with an earlier copy

In an image analysis, **Compare with an earlier copy** can now match a
submitted WebP that was cropped and then downsized. The evidence view
shows the part of the earlier image that was compared, the coordinates
in both original files, and an overlay marking retained brightness
differences. The PDF and JSON use those same native coordinates and
crops. Differences can come from ordinary editing. The measured areas
have zero weight in the automated AI assessment. The earlier image
outside the matched crop is unassessed.

The earlier image must be an upright opaque lossless PNG or WebP with
each side between 512 and 1,024 pixels. The matched crop must retain at
least 80% of each axis and remove at most 128 pixels per axis. The
submitted image must be a supported opaque lossy WebP with a shortest
side of at least 384 pixels and dimensions consistent with 72–95% of
the crop's size. The previous same-size, full-frame resize and unscaled
crop comparisons keep priority when they can explain the submitted framing.

Feature correspondence proposes a whole-pixel crop rectangle. Separate
pixel groups confirm its placement. At least two resize filters must
support both the comparison and the brightness differences. The
earlier crop boxes round outward to include every original pixel
projected into each submitted box. The existing regional comparison
rejects large or ambiguous differences and can miss faint changes.
[Earlier crop-only comparison](image-reference-cropped-webp-2026-11-14.md)
· [Earlier resize-only comparison](image-reference-resized-webp-2026-11-14.md)

## Reproduce the measurement

The evaluation uses 26 development and 24 separate held photo groups
from the [previous WebP comparison](image-reference-lossy-webp-2026-11-14.md).
It reconstructs sharp, feathered and small edits with exact changed-pixel
support before the WebP encode. A useful location must recover at least
half the measured change and reach 0.30 intersection over union, with
the correct earlier-image crop. Controls include unchanged delivery,
prior JPEG and WebP loss, local blur and tone, stretches, mirrors and
unrelated images. On a delivery where cropping removes all the changed
pixels, the receipt keeps the case and counts it separately. The held
carriers use Lanczos/WebP 70 and area/WebP 89, separate from the bicubic
development carrier. These are staged deliveries using prior source
banks. Their measured coverage applies to those files and settings.

The receipt reviewer uses standard-library Python to recount exact
crop, selected regions, misses and control flags. The native replay
rebuilds each delivery from the pinned local source banks and verifies
the submitted-image and mask hashes. No weights are needed for this
comparison.

## Comparison results

| Delivery | Prior useful locations | Useful locations with crop and resize | Controls highlighted |
| --- | ---: | ---: | ---: |
| Development, bicubic/WebP 76 | 0/104 | 90/104 | 0/208 |
| Held, Lanczos/WebP 70 | 0/96 | 60/96 | 0/192 |
| Held, area/WebP 89 | 0/96 | 69/96 | 0/192 |

All 312 accepted held comparisons used the exact earlier crop. They
include 138 staged edits and 174 controls. The other 264 held comparisons
were unassessed, including all mirrors, stretches and unrelated images.
Five accepted edits flagged differences with below-target overlap. Four
Lanczos deliveries lost the whole edit outside the crop and showed no
highlight. The other 54 staged edits were unassessed.
[Development measurements](evaluation-panels/image-reference-crop-resize-webp-development-2026-11-15.json)
· [Held measurements](evaluation-panels/image-reference-crop-resize-webp-held-2026-11-15.json)

## Local opt-in

Set `VERITAS_IMAGE_REFERENCE_LOSSY_WEBP_CROP_RESIZE=true` alongside
`VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_IMAGE_REFERENCE_LOSSY_WEBP=true`. Turn off the new setting to
restore the earlier behavior for new comparisons. Saved comparison
exports retain their original verification rules.

## App and export verification

Three local analyses used simulated USD 9.99 checkout. No job started
before payment. The small feathered edit was unassessed with the option
off. With the option on it retained a 52 × 52 region at (205, 231) in
the submission and the corresponding 58 × 58 crop at (282, 261) in
the earlier image. The recompression control found no differences and
the mirror remained unassessed. The original analysis results kept the
same hashes. Source replay recomputed both original crops and the
selected pixel mask, and the smoke test verified that the report
embedded the same crops and overlay as the JSON.
[Local app receipt](evaluation-panels/image-reference-crop-resize-webp-app-2026-11-15.json)

## Replay commands

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.image_reference_crop_resize_webp_transfer
panels=docs/evaluation-panels
python3 -B -S -m "$module" review "$panels/image-reference-crop-resize-webp-held-2026-11-15.json"
"$runner" -B -m "$module" held "$run/held.json" --development-sources "$development_sources"
cmp "$panels/image-reference-crop-resize-webp-held-2026-11-15.json" "$run/held.json"
"$runner" -B -m scripts.image_reference_lossy_webp_app "$run/app" "$run/held.json" \
  --crop-resize --development-sources "$development_sources"
```

# Locating photo edits after a rotated or tilted crop

The earlier-copy comparison can align two compatible JPEGs after a change
in perspective. It measures local differences on the submitted image and
shows the area compared, the difference mask, and a native crop from each
file. The PDF and JSON retain the same coordinate measurements and image
hashes. The comparison carries zero AI-authorship weight.

## Method

Mutual keypoint matches first estimate a transform from the submitted
pixel centers to the earlier copy. A robust consensus fit checks the
spread and projection error of those matches. A separate brightness
comparison looks for shared detail in at least ten of sixteen areas
spread across all four quadrants. The check estimates an overall color
adjustment using points that agree between images. It retains connected
local differences after allowing for resampling near image edges.
Pixels without supported geometry stay outside the measured area.
No external models or trained weights are used. [Keypoint matching research](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
· [Recent reference-based evidence research](https://arxiv.org/abs/2611.10908)

The local mode runs when the existing earlier-copy comparisons cannot
establish a match. Enable `VERITAS_IMAGE_REFERENCE_PERSPECTIVE_REVIEW=true`
with `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` in development.
In a completed image analysis, open **Compare with an earlier copy** and
upload the earlier file. The tested profiles use standard 4:2:0 JPEG
tables, an earlier quality-90 save, and a submitted quality-75 or
quality-90 save. Each image side is between 320 and 1,024 pixels.
The comparison addresses known two-copy differences. A single upload
still uses the separate image checks.

## Source-separated comparison

The development set includes 24 previously studied photographs with LaMa
removals and Stable Diffusion 1.5 insertions. It selects one setting for
each JPEG quality using 768 newly transformed deliveries. Both qualities
select the same value. The held sets supply 45 other photographs from two
earlier rights-reviewed LaMa panels. Each source produces a known edit
with sharp and feathered boundaries, an unchanged photo, two ordinary
whole-photo adjustments, a localized blur, an unrelated photo, and a
half-replaced control. Each delivery uses either a rotated crop or a
simulated perspective crop. Both copies passed through JPEG encoding.
These inputs simulate changes to the framing of an existing flat image.
Physical camera recapture, lens distortion, glare, and separately
photographed views of a three-dimensional scene remain outside this
evaluation.
[BG-REAL license and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Development source receipt](evaluation-panels/image-cross-quality-sources-2026-11-16.json)
· [Nine-photo held source receipt](evaluation-panels/image-removal-peer-sources-2026-11-16.json)

A correct location covers at least half of the known changed pixels and
at least 30% intersection over union. The mask is defined before JPEG
delivery and projected to the submitted grid for assessment. Ordinary
local blur is reported separately from whole-image processing and
incompatible pairs. The same photo contributes multiple deliveries.

| Submitted JPEG | Correct removal locations, previous → new | Ordinary highlights | Edited deliveries without a region | Unrelated or half-replaced pairs accepted |
| --- | ---: | ---: | ---: | ---: |
| Quality 75 | 0 → 133 of 180 | 2 of 270 | 27 of 180 | 0 of 180 |
| Quality 90 | 0 → 132 of 180 | 0 of 270 | 29 of 180 | 0 of 180 |

The earlier comparison was unassessed on all 1,440 held deliveries.
The new comparison assessed every removal and all but two ordinary
deliveries. It found 134 sharp and 131 feathered removals across both
qualities. Thirty-nine edited deliveries had a highlight without a
useful location. Both ordinary highlights followed a global color
adjustment on the same photo. The method also flagged two known local
blur deliveries out of 180. The largest observed discrepancy from the
authored geometry was 0.91 pixels across the nine tested reference
points per accepted delivery. [Numeric checks and per-delivery results](evaluation-panels/image-reference-perspective-2026-11-16.json.gz)

Four full image analyses verified the prepaid local flow with simulated
Stripe. Checkout and an unpaid event queued no jobs. A held removal had
690 highlighted pixels with the original crops reproduced in Chrome,
PDF, and JSON. An ordinary transformed copy and a whole-image tone
change showed no region. The unrelated input stayed unassessed. The
comparison preserved each saved result hash.

## Replay

Use the [public-source restore instructions](image-removal-peer-2026-11-16.md#replay)
to rebuild the nine-photo panel. The same restore command accepts the
development source receipt linked above. Rebuild the 36-photo transfer
set from the [earlier JPEG input recipe](image-region-ranking-2026-11-16.md#replay-and-compare-another-check).
The published numeric receipt includes input hashes, selected regions
and exact overlap counts. The receipt recount requires only Python's
standard library. A separate run on the rebuilt nine-photo set
reproduced all 288 delivered JPEG commitments, selected regions, and
overlap counts exactly. The local native replays used OpenCV 5.0.0 and
Pillow 12.3.0.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module=scripts.image_reference_perspective_panel
python3 -B -S -m "$module" verify docs/evaluation-panels/image-reference-perspective-2026-11-16.json.gz
backend/.venv/bin/python -B -m "$module" fit "$run/freeze.json.gz" --directory "$run" \
  --source "$development_jpegs" fresh
backend/.venv/bin/python -B -m "$module" assess "$run/held.json.gz" --directory "$run" \
  --freeze "$run/freeze.json.gz" --source "$small_removal_jpegs" fresh --source "$retained_jpegs" transfer
```

An individual comparison is replayable from its JSON download and the
two original JPEG files. The replay regenerates the full alignment,
checks the source hashes and native coordinates, and verifies every
exported PNG byte.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_perspective_export \
  comparison.json earlier.jpg submitted.jpg
```

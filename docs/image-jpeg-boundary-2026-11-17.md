# Can a JPEG boundary check locate small AI removals?

An AI removal can disturb texture and color at the edge of a filled area.
Ordinary texture, retouching, and JPEG compression can cause similar
changes. This comparison tests whether training on the same photographed
area before and after different edits helps locate the filled area.

## What the check measures

Each native 8 × 8 JPEG cell contributes measurements of three luminance
frequency bands, cross-channel residual energy and correlation, and
texture change relative to the local image gradient. A small convolutional
network estimates the filled region. The comparison fits the same network
twice. The first fit predicts changed pixels. The second also learns the
boundary and favors a higher response at the same position in the generated
edit than in an ordinary version. Three arms compare the first fit, the
second fit’s interior output, and a proposal score that uses both outputs
from the second fit. Each arm selects at most one connected area and
compares its response with nearby pixels.

Recent work motivates testing edge evidence separately from region evidence.
The implementation is project code trained on licensed source material.
No research checkpoint or inference code was copied for this check.
[Edge-contrast study](https://www.mdpi.com/2313-433X/12/10/480)
· [FUSED region-mask evaluation](https://arxiv.org/abs/2608.28302)

## Source separation and controls

Fitting uses 59 source photos. A separate 25 determine operating points.
The previous region-ranking and cross-quality studies supply 60 photos
with LaMa and Stable Diffusion edits. Another 24 supply full and partial
LaMa removals and matching traditional fills. The deterministic split
keeps deliveries from each photo together. JPEG75 and JPEG90 each have
four candidate points allowing up to four added ordinary source groups
during selection. The choice favors useful extra regions on calibration
photos, with misplaced regions and ordinary alerts counted separately.

The held comparison uses 20 source photos excluded from fitting and
selection. Their IDs, photographers, byte and pixel hashes, and close
image matches were checked against the preceding removal studies. A later
cross-check found seven of these photos in separate prior-grid studies.
Each picture supplies a small removal from LaMa
and MI-GAN. MI-GAN is absent during fitting and selection. Sharp and
inward-feathered edits are encoded at both JPEG qualities. The controls
include the untouched photo, local blur, local tone, local copying, and
traditional inpainting with OpenCV’s Telea and Navier–Stokes methods.
The JPEG deliveries share source photographs, and some ordinary saves
repeat across editor families. Counts represent 20 source groups.
[BG-REAL sources and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Editor provenance and licenses](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)

An edit is located if the selected region overlaps at least half of the
exact changed pixels, with an intersection over union of at least 0.30.
The comparator runs the current JPEG compression and regional checks,
including the serving repeat-compression confirmation for JPEG90.
An existing successful location is retained in the combined count.

## Held results

| Delivery | Correct locations, current → with candidate | Ordinary JPEGs flagged, current → with candidate | Misplaced candidate locations on edited files |
| --- | ---: | ---: | ---: |
| JPEG75 | 0 → 0 of 80 | 9 → 18 of 180 distinct files | 4 |
| JPEG90 | 0 → 0 of 80 | 6 → 15 of 180 distinct files | 4 |

The chosen JPEG75 point uses the first fit. The JPEG90 point uses the
second fit’s interior response. All three methods at all four tested
points found zero correct locations on either quality. At the chosen
points, the additional false alerts at both qualities came from the
same photograph, including four JPEGs made with traditional fills.
Counting repeated ordinary deliveries gives 10 → 20 of 200 at
JPEG75 and 6 → 16 of 200 at JPEG90. At these settings the number of
flagged ordinary source groups increases from one to two at each
quality. The leading region before thresholding overlaps four edits at
JPEG75 and two at JPEG90 at the required extent, all from LaMa. The
tested features and proposal ranking leave the small removals largely
unlocated even without a decision threshold. The candidate stays out
of the app.
[Frozen selection](evaluation-panels/image-jpeg-boundary-selection-2026-11-17.json.gz)
· [Exact held data](evaluation-panels/image-jpeg-boundary-held-2026-11-17.json.gz)

## Replay

The public results include source commitments, fitted weight hashes and
scalar region measurements. Rebuild the prior JPEG archives from their
published instructions, then use the reserved source plan and the two
pinned editor files. The archive variables below refer to the previous
region-ranking, cross-quality and frequency-band preparation folders.
[Earlier source preparation](image-jpeg-frequency-bands-2026-11-17.md#replay-and-independent-comparison)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
source_script=scripts.image_jpeg_band_sources
comparison=scripts.image_jpeg_boundary_transfer
stem=image-jpeg-boundary
"$runner" -B -m "$source_script" acquire "$panels/$stem-reservation-2026-11-17.json.gz" \
  "$retained/deliveries/manifest.json" "$run/intake" --previous-intake "$previous/intake/intake.json"
PYTHONPATH="$ort:backend" "$runner" -B -m "$source_script" prepare "$run/intake" "$editors" "$run/held" --partial
"$runner" -B -m scripts.image_jpeg_boundary_controls "$previous/intake" "$run/old-ordinary-full"
"$runner" -B -m scripts.image_jpeg_boundary_controls "$previous/intake" "$run/old-ordinary-partial" --partial
"$runner" -B -m scripts.image_jpeg_boundary_controls "$run/intake" "$run/held-ordinary" --partial
"$runner" -B -m "$comparison" fit "$previous" "$run" "$run/fit" \
  --input "$retained/deliveries" "$retained/reference.jsonl" \
  --input "$cross/confirmation/deliveries" "$cross/fresh-reference.jsonl"
for profile in held held-ordinary; do
  "$runner" -B -m scripts.image_region_rank_reference "$run/$profile" "$run/$profile-reference.jsonl" --repeat-jpeg90
done
"$runner" -B -m "$comparison" held "$run/fit" "$run/held.json.gz" \
  --input "$run/held" "$run/held-reference.jsonl" \
  --input "$run/held-ordinary" "$run/held-ordinary-reference.jsonl"
"$runner" -B -m scripts.export_image_jpeg_boundary_review "$run" "$run/fit" "$run/held.json.gz" "$run/audit"
python3 -B -S -m app.bench.image_jpeg_boundary "$panels/$stem-held-2026-11-17.json.gz" \
  --freeze "$panels/$stem-selection-2026-11-17.json.gz"
```

The export prepares exact native masks and source bindings locally for a
different detector to assess using the same truth. It independently
compares the pixel counts with the frozen receipt. The readback verified
all 560 truth rasters and 1,120 source and input byte commitments, and
replayed all 28 candidate highlights. The public package contains no
source photos, model weights or regional rasters.
[Reserved source plan](evaluation-panels/image-jpeg-boundary-reservation-2026-11-17.json.gz)
· [Source intake](evaluation-panels/image-jpeg-boundary-source-intake-2026-11-17.json.gz)
· [Binding and pixel-count receipt](evaluation-panels/image-jpeg-boundary-audit-2026-11-17.json)

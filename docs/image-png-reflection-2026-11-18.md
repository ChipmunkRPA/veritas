# Exact reflected areas in a PNG

Chipmunk Verify can show a pair of small areas in one PNG whose native
pixels match after a horizontal or vertical flip. The ordinary findings
view marks both locations and shows both original crops. The PDF and JSON
retain the same coordinates, reflection direction, and pixel counts. The
signed original-file verifier recalculates the match and the differences
around the crops. A local edit or a repeated graphic can produce this
pattern. The check contributes zero weight to AI authorship confidence.

Set `VERITAS_IMAGE_PNG_REFLECTION_REVIEW=true` for the local development
app. The check accepts upright, static 8-bit RGB PNGs with sides from
384 to 768 pixels. Candidate crops range from 16 to 56 pixels per side.
They must be disjoint and textured, and the three-pixel surroundings
must differ on at least three sides after alignment. It runs when the
other copy comparisons have cleared and can withhold a finding when
several plausible pairs compete. The evidence records when the exact
check ran, when the file was outside its profile, and when a preceding
result made the fallback unnecessary. It reuses the exact measurement
already evaluated for lossless WebP.
[Earlier WebP study](image-webp-reflection-2026-11-18.md)

## Local comparison

Twelve attributed camera photos from an earlier JPEG study supplied
new reflected edits on a predeclared PNG delivery plan. Their source
files and publisher identifiers are separate from the camera photos
in the WebP trial. The nine repeated-pattern photos supplied ordinary
controls only. The existing comparator included approximate reflection
and translation, geometric matching, and exact unrotated and rotated
PNG comparisons when enabled. A useful location required 0.50 minimum
intersection over union for both regions and at least half of the exact
changed pixels. Each photograph contributes several deliveries.
[Camera source roster](image-native-chroma-consistency-2026-11-18.md#replay)
· [Photo terms and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Repeated-pattern source inventory](image-small-affine-webp-2026-11-16.md#local-comparison)

| Staged edit | Current PNG comparison | With exact reflection |
| --- | ---: | ---: |
| Horizontal flip | 0/12 | 8/12 |
| Vertical flip | 0/12 | 9/12 |
| Feathered vertical flip | 0/12 | 9/12 |
| All staged reflections | 0/36 | 26/36 |

The 26 added locations came from nine photo groups. The fallback added
zero misplaced locations and zero highlights among 99 controls. These
controls included unmodified images, blur, tone changes, whole-raster
symmetry, straight copies, and quarter turns. The incumbent flagged
21 controls, including 17 staged straight copies or quarter turns. The
new exact comparison abstained on 20 of 21 symmetry layouts. These are
staged copy results on the specified format and dimensions.

## Local app verification

Ten simulated USD 9.99 payments preceded all analysis jobs in an isolated
run. Eight cases verified the browser, PDF, JSON, and signed original file,
including both directions, feathered edges, an unchanged image, whole-image
symmetry, competing pairs, an RGBA PNG, and the prior rotation result. The
horizontal and vertical examples produced the same automated assessment with
the PNG reflection setting off and on. Both original crops matched their
recorded source pixels in the browser and PDF.
[Local checkout and report receipt](evaluation-panels/image-png-reflection-app-2026-11-18.json)

## Independent replay

The compressed receipt holds attributed source commitments, delivery and
mask digests, candidate rectangles, and exact witness counts. It contains
no image files or crops. The standard-library recount validates the
geometry, fallback decisions, and aggregate counts.
[Numeric receipt](evaluation-panels/image-png-reflection-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_png_reflection \
  docs/evaluation-panels/image-png-reflection-2026-11-18.json.gz
```

The full native replay regenerated all 135 PNGs and their exact change
masks, recomputed the pixel overlaps, and reproduced every native
reflection result. Use the twelve-file `intake` folder from the
[camera transfer](image-masked-reconstruction-2026-11-17.md#reproduce)
and the nine-file `jpeg` pattern folder from the
[earlier WebP replay](image-webp-reflection-2026-11-18.md#offline-replay).
The receipt lists each expected source file and checksum.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_png_reflection \
  docs/evaluation-panels/image-png-reflection-2026-11-18.json.gz \
  --photographs "$camera_intake" --patterns "$repeated_patterns"
```

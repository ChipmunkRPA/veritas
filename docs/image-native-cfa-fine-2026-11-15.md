# Small interpolation shifts in a single PNG

The app can show a small area where the color interpolation pattern moves
to a different alignment from the rest of the picture. The evidence gives
the original pixel coordinates, an outline on the submitted image, the
unmodified 16-bit crop and the number of cells that support the change.
This indicates where to examine a possible local edit and carries zero
weight in the AI assessment. The file does not need an earlier copy.

This check covers bare 2048 × 1536 RGB16 PNGs with a repeatable interpolation
pattern. The existing rule establishes the normal phase from two spatial
folds formed by 32-pixel cells. The new check uses 16-pixel cells on four
grids offset by eight pixels. A region needs repeatable support for the same
alternate phase in at least two-thirds of its cells. The rule selects one
native 32- or 48-pixel-wide and -high crop. A region already selected by
the coarse, rotation or lower-depth check keeps priority.

## Held camera groups

The study used CC0 camera originals from three existing pinned RAW rosters.
All 26 camera models and source assets are distinct across roles. The six
development cameras set a single response point using ordinary deliveries.
The ten held and ten separate transfer cameras were scored at that point
with newly prepared edits. Their original RAW files previously appeared in
this project's source rosters. These results are from staged edits and a
controlled bilinear RAW export, with one scene per camera.

A useful location covers at least half the edited pixels and has at least
0.30 intersection over union with the changed-pixel mask. The comparison
uses the prior coarse phase check alone and with the new fallback.

| Delivery | Held, coarse → with fine | Transfer, coarse → with fine |
| --- | ---: | ---: |
| Sharp 32 × 32 | 0 → 9 / 10 | 0 → 7 / 10 |
| Sharp 48 × 48 | 1 → 8 / 10 | 1 → 8 / 10 |
| Sharp 64 × 64 | 6 → 8 / 10 | 6 → 7 / 10 |
| Feathered 32 × 32 | 0 → 0 / 10 | 0 → 0 / 10 |
| Feathered 48 × 48 | 0 → 0 / 10 | 0 → 0 / 10 |
| Feathered 64 × 64 | 6 → 8 / 10 | 5 → 7 / 10 |
| Useful locations across all edits | 13 → 33 / 60 | 12 → 29 / 60 |
| Ordinary controls flagged | 0 → 0 / 60 | 0 → 0 / 60 |

The ordinary controls cover unchanged photos, even-grid patches, local tone
or blur, fractional-pixel movement and a whole-image shift. Another 20
controls were JPEG-compressed and saved as 8-bit PNG. The check declined
those inputs. The fine check flagged 18 feathered 48-pixel edits, but none
of its chosen crops reached the required overlap. The box is evidence of
the measured pattern within it. It should not be read as an edit mask.
Small generated fills, JPEGs, and consumer AI editor exports need separate
testing.

[Development receipt](evaluation-panels/image-native-cfa-fine-development-2026-11-15.json) ·
[Held receipt](evaluation-panels/image-native-cfa-fine-held-2026-11-15.json) ·
[Transfer receipt](evaluation-panels/image-native-cfa-fine-transfer-2026-11-15.json)

## Local verification

Enable `VERITAS_IMAGE_NATIVE_FINE_REVIEW=true` alongside
`VERITAS_IMAGE_NATIVE_INTERPOLATION_REVIEW=true` in development for new PNG
analyses. The new setting defaults to off. The signed evidence records which
check produced the result and can verify it from the original file.

The full local worker processed a sharp edit and its matched tone control
with the option off and on. All four used simulated USD 9.99 checkout and
queued after payment. On the sharp edit, the evidence API, Standard PDF and
JSON show columns [480, 512), rows [416, 448) and support from all four
measured cells. The PDF embeds the same outline and crop, and the API
retains every original 16-bit sample in the crop. Both signed packages
replayed from the original files. The tone control showed no region, and
the automated assessment for each file agreed with the option off and on.
[Local checkout and export receipt](evaluation-panels/image-native-cfa-fine-app-2026-11-15.json)

## Feathered edit crops

The 18 flagged 48-pixel feathered edits had 32 × 32 crops entirely inside
the changed area. Each crop covered 44% of the changed pixels, below the
study's 50% criterion. The extent check starts with the accepted fine
crop. Along each side it tests pairs of 8 × 8 cells, one just outside the
crop and the other farther out. A side needs the normal interpolation
pattern to be weaker in the nearer cell and recover in the farther cell
in at least two-thirds of the pairs. The fixed limits are 0.75 in the near
cell, 0.80 in the far cell, and a rise of at least 0.20. The crop extends
eight pixels along each supported side when at least two sides qualify.
The report describes the inner region and the support for the larger crop.

The same six development camera groups were used to set the rule before
replaying the two panels of ten other cameras. The wider feather deliveries
use new nine-pixel blends from the same sources as the earlier six-pixel
blends. Each row counts useful crops against the changed-pixel masks.

| Staged delivery | Held, previous → extended | Transfer, previous → extended |
| --- | ---: | ---: |
| 48 × 48, six-pixel feather | 0 → 7 / 10 | 0 → 4 / 10 |
| 48 × 48, nine-pixel feather | 0 → 7 / 10 | 0 → 4 / 10 |
| 64 × 64, six-pixel feather | 8 → 9 / 10 | 7 → 7 / 10 |
| 64 × 64, nine-pixel feather | 8 → 9 / 10 | 7 → 7 / 10 |
| 32 × 32, each feather width | 0 → 0 / 10 | 0 → 0 / 10 |

Ten of the eleven useful 48-pixel crops matched the staged region exactly
at each feather width. All previously useful sharp crops still qualified.
The rule only extends a previously flagged crop and added no flags in the
120 supported ordinary controls. The test kept the same PNG scope and the
same fine detection point. Crop extent carries zero AI-confidence weight.

[Development extent receipt](evaluation-panels/image-native-cfa-extent-development-2026-11-15.json) ·
[Held extent receipt](evaluation-panels/image-native-cfa-extent-held-2026-11-15.json) ·
[Transfer extent receipt](evaluation-panels/image-native-cfa-extent-transfer-2026-11-15.json)

Enable `VERITAS_IMAGE_NATIVE_EXTENT_REVIEW=true` alongside the two PNG
settings above. The extent setting defaults to off. The local worker
replayed the feathered edit and its tone control with the setting off and
on. All four analyses queued only after simulated USD 9.99 checkout. On
the edit, the earlier 32 × 32 crop grew to 48 × 48. The API, JSON and
Standard PDF give columns [480, 528), rows [416, 464) and the basis for
all four supported sides. The original 16-bit crop, the outline and signed
source verification agreed. The tone control still showed no region, and
automated assessments for both files agreed with the setting off and on.
[Local extent checkout and exports](evaluation-panels/image-native-cfa-extent-app-2026-11-15.json)

The standard-library reviewer recalculates the supported extension and
the table. Native replay uses the same camera and decoder inputs as the
fine check, with `held` or `transfer` in place of `review` and the
`--cache` and `--archive` arguments described below.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_native_cfa_extent_transfer \
  review docs/evaluation-panels/image-native-cfa-extent-held-2026-11-15.json
```

Add `--extent "$run/extent-held.json"` when running the local app command
below to verify the new report with a replayed extent receipt.

## Reproduce

The numeric recount uses the standard library and the public receipts.
Pass the transfer receipt in place of the held receipt to recount the
second panel.

```sh
receipts=docs/evaluation-panels
PYTHONPATH=backend python3 -B -S -m scripts.image_native_cfa_fine_transfer \
  review "$receipts/image-native-cfa-fine-held-2026-11-15.json" \
  --calibration "$receipts/image-native-cfa-fine-development-2026-11-15.json"
```

Native replay uses the checked camera files and pinned decoder from the
[earlier RAW study](image-native-cfa-closure-2026-11-13.md#replay). The
`--archive` directory contains the exact `sample-{id}{extension}`
files listed in the three rosters. The `--cache` directory contains the
decoder wheel. Write generated receipts and app artefacts to a temporary
directory. The native scorer takes `development`, `held` or `transfer`.
For each held role, pass the replayed development receipt as `--calibration`.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_native_cfa_fine_transfer \
  development "$run/development.json" --cache "$cache" --archive "$archive"
backend/.venv/bin/python -B -m scripts.image_native_cfa_fine_transfer \
  held "$run/held.json" --cache "$cache" --archive "$archive" --calibration "$run/development.json"
backend/.venv/bin/python -B -m scripts.image_native_cfa_fine_app \
  "$run/app" --held "$run/held.json" --calibration "$run/development.json" --cache "$cache" --archive "$archive"
```

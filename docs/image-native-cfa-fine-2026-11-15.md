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

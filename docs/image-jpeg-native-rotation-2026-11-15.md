# Locate right-angle copies in larger JPEG uploads

The local image review searches for right-angle copies on the original pixel
grid of larger JPEGs. It compares native texture samples in different
orientations, groups nearby matches that imply the same displacement, and
checks how closely the aligned RGB pixels agree inside and outside both
crops. The finding links to both crops in the uploaded image. Copying
evidence can come from a manual or automated edit, so it carries zero
weight in AI authorship confidence. The check uses no new model weights.
[Earlier JPEG rotation check](image-jpeg-turned-copy-2026-11-15.md)

## Evaluation

The 149 held deliveries contain 60 staged edits, 80 matched controls and
nine repeated-structure photographs. Ten held camera groups are separate
from the four development groups. The camera RAW files had already been
used in other project evaluations. Crops use the sensor-derived RGB
pixel grid at 1,920 × 1,280 or 3,072 × 2,048, including portrait versions.
Each camera group uses one of two standard compression profiles. The
matched controls include unchanged photographs, local blur, straight
copies and mirrored copies. A useful result must overlap both known
regions at intersection over union of at least 0.50.
[CC0 source collection](https://raw.pixls.us/)
· [Repeated-structure credits](image-affine-native-scale-2026-11-11.md)

| Held edits | Existing geometry check | With rotation check |
| --- | ---: | ---: |
| All | 1/60 | 26/60 |
| 1,920 × 1,280, either orientation | 1/30 | 10/30 |
| 3,072 × 2,048, either orientation | 0/30 | 16/30 |
| Standard JPEG quality 90 | 0/30 | 12/30 |
| Standard JPEG quality 95 | 1/30 | 14/30 |

The new check added useful locations in all ten held camera groups. It
located six of 20 quarter-turn copies, 11 of 20 half-turn copies and
nine of 20 feathered quarter-turn copies. It missed 34 edited views, with
zero misplaced findings or unassessed views in this panel. The existing
geometry check flagged one repeated-structure control. The added rotation
check flagged zero of 89 controls. Development moved from 2/24 to 13/24
useful locations, with one misplaced location and zero new flags
across 32 controls.
[Held receipt](evaluation-panels/image-jpeg-native-rotation-held-2026-11-15.json)
· [Development receipt](evaluation-panels/image-jpeg-native-rotation-development-2026-11-15.json)

## Local app

Enable `VERITAS_IMAGE_JPEG_NATIVE_ROTATION_REVIEW=true` with
`VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW=true` in development. The larger
check covers upright RGB 4:2:0 JPEGs with exact standard quality-90 or
quality-95 quantization tables, a short edge of at least 1,024 px, and
a long edge of 1,025–3,072 px. Other table and color profiles within
this size range show an explicit coverage limitation. The conventional
and affine checks retain their previous scopes. The development switch
defaults to off elsewhere.

The app replay used three held positives and their matched ordinary
photographs, each with the option off and on. The three positives
showed both crop overlays in the browser. The API, PDF, JSON and signed
package agreed on the positions and both original crops. The three
controls produced no rotation finding. Earlier evidence remained
readable and overall assessments agreed across all six pairs. Twelve
simulated USD 9.99 checkouts queued zero unpaid jobs.
[Local app receipt](evaluation-panels/image-jpeg-native-rotation-app-2026-11-15.json)

## Reproduce

The numeric review uses standard Python, the committed source rosters
and the measurement receipt. Python 3.12 and 3.14 returned identical
counts. A second native pass rebuilt every held file and reproduced
all 149 measurements byte for byte with a different worker count.
Native replay needs the retained RAW sources, the credited Commons
files, the decoder wheel and the repository's image dependencies.
Keep source images and crops local. The browser replay uses the built
frontend and local Chromium.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.image_jpeg_native_rotation_transfer
receipt=docs/evaluation-panels/image-jpeg-native-rotation-held-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
"$runner" -B -m "$module" freeze held "$raw_archive" "$wheel" "$commons" "$run/inputs"
"$runner" -B -m "$module" capture "$run/inputs" "$run/held.json" --workers 2
cmp "$receipt" "$run/held.json"
"$runner" -B -m scripts.image_jpeg_native_rotation_app "$run/app" "$receipt" "$run/inputs" --browser
```

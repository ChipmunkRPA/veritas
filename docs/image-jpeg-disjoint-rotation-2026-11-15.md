# Find two independent rotated-copy pairs in a JPEG

Chipmunk Verify can show two separate sets of matched crops in the same
large JPEG. Each pair keeps its own rotation angle, original pixel
coordinates, inner color difference, and surrounding measurements. The
browser provides an independent outline and source link for each crop.
The Standard PDF includes both pairs, and JSON retains the measurements
in pair order. The signed verifier recomputes the matches from the uploaded
JPEG. Repeated detail and editing can produce a match, so this observation
carries zero weight in AI authorship confidence.

The optional native-grid check covers the existing upright RGB 4:2:0 JPEG
profile with standard quality-90 or quality-95 quantization tables. It
uses a short side of at least 1,024 pixels and a long side of 1,025 to
3,072 pixels. Existing texture proposals and color checks remain in use.
The new selection accepts up to two pairs whose crops are at least 16
pixels apart across pairs. Matches sharing a crop and files with three
distinct proposals remain ambiguous. Similar candidates with the same
rotation and overlapping crop locations represent one pair.
[Earlier native-grid check](image-jpeg-native-rotation-2026-11-15.md)
· [Multiple-copy benchmark design](https://www.vcl.fer.hr/papers_pdf/CoMoFoD%20-%20New%20Database%20for%20Copy-Move%20Forgery%20Detection.pdf)

## Held comparison

Ten held camera groups are separate from four development camera groups.
Each earlier camera JPEG contributes a new double-quarter-turn case, a
quarter-turn plus half-turn case, a shared-source case and a straight-copy
control. The image is re-encoded at the same JPEG profile after staging.
The method also replays the earlier single edits and genuine or ordinary
controls on their existing bytes. The camera originals and repeated
structure photographs have earlier project exposure. A correct pair
must overlap both labeled crops by at least 0.50 intersection over union.
[CC0 camera sources](https://raw.pixls.us/)
· [Earlier image source roster](image-jpeg-native-rotation-2026-11-15.md#evaluation)

| Held outcome | Earlier single-pair rule | Two-pair rule |
| --- | ---: | ---: |
| Both independent pairs located | 0/40 | 26/40 |
| Individual correct pairs | 10/80 | 63/80 |
| Misplaced pairs on the double edits | 0 | 1 |
| Useful earlier single edits | 26/60 | 26/60 |
| Ordinary control flags | 0/109 | 0/109 |
| Double-pair claims on shared-source copies | 0/20 | 0/20 |

The new rule located both pairs on nine held cameras, with 13 successes
for each staged rotation combination. It issued 27 double-pair results.
One contained a correct pair and a misplaced pair. Ten further double-edit
files showed one correct pair. One double edit remained
unassessed and two had no match. In development, the count for both
correct pairs rose from 0/16 to 12/16 with zero flags on 40 controls.
These results measure same-image copies under the tested conditions.
The source direction and origin of the content remain unresolved.
[Development measurements](evaluation-panels/image-jpeg-disjoint-rotation-development-2026-11-15.json)
· [Held measurements](evaluation-panels/image-jpeg-disjoint-rotation-held-2026-11-15.json)

## Local app and replay

Enable `VERITAS_IMAGE_JPEG_DISJOINT_ROTATION_REVIEW=true` alongside
`VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW=true` and
`VERITAS_IMAGE_JPEG_NATIVE_ROTATION_REVIEW=true` in development. The new
option defaults off. It inherits the same compression and size checks.

The isolated worker test uses two positives, an ordinary photograph,
a shared-source case, and an earlier-version result for the same mixed
rotation input. It verifies pair-specific browser outlines, source links,
and crop dimensions. Both positives include four exact original crops in
the Standard PDF and signed archive. The prior evidence remains available
and the overall assessment agrees across versions on the matched replay.
All five simulated USD 9.99 analyses queue only after payment.
[Local app receipt](evaluation-panels/image-jpeg-disjoint-rotation-app-2026-11-15.json)

Python 3.12 and 3.14 produced identical numeric reviews. The second
measurement pass reconstructed all held results byte for byte with a
different worker count. The receipt contains positions, measurements and
hashes. Image bytes and crops stay local. Reuse or recreate the earlier
[pinned original files](image-jpeg-native-rotation-2026-11-15.md#reproduce)
in a local input directory. Browser verification uses the built frontend
and local Chromium.

```sh
export PYTHONPATH=backend
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.image_jpeg_disjoint_rotation_transfer
held=docs/evaluation-panels/image-jpeg-disjoint-rotation-held-2026-11-15.json
python3 -B -S -m "$module" review "$held"
"$runner" -B -m "$module" capture held "$inputs" "$run/held.json" --workers 2
cmp "$held" "$run/held.json"
"$runner" -B -m scripts.image_jpeg_disjoint_rotation_app "$run/app" "$held" "$inputs" --browser
```

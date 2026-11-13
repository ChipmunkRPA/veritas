# Native PNG interpolation evidence across image dimensions

The current local interpolation check accepts one native landscape
geometry. This transfer changes only the eligible dimensions. It keeps
the same pixel grid, two-fold interpolation support, score, and
decision point. Source pixels stay in their submitted coordinates.
The result describes local interpolation and carries zero AI-confidence
weight.
[Existing check and calibration](image-native-cfa-png-transfer-2026-11-13.md)

## Frozen test

Ten distinct camera models outside the earlier native-image studies
supply checksum-verified CC0 RAWs. Use a fixed linear 16-bit render
with no resizing. Create a bare truecolor PNG in each of five
geometries from a centered crop. For portrait, a native quarter-turn
moves exact pixels before cropping. The resulting PNG contains
already-upright pixels and no orientation metadata. That design
tests output coordinates on the presented portrait raster.

Each camera and geometry includes an untouched source, even-aligned
copy, globally odd crop, local fractional interpolation and a resize
round trip. Positives are sharp and feathered 128-pixel copies
shifted by one source pixel horizontally, vertically, or both.
The exact changed-pixel mask and output PNG are regenerated on
review. A useful region needs at least 0.30 intersection over union
and 0.50 changed-pixel recovery. Advance a geometry only if both
edit types get at least seven useful locations, all 50 ordinary
views remain clear, and all ten resize controls decline.
[CC0 source roster](evaluation-panels/image-native-cfa-dimensions-cc0-roster-2026-11-13.json)

## Held results

| Submitted dimensions | Useful sharp | Useful feathered | Ordinary highlights | Resize abstentions |
| --- | ---: | ---: | ---: | ---: |
| 2048 × 1536 | 9/10 | 9/10 | 0/50 | 10/10 |
| 2304 × 1728 | 8/10 | 8/10 | 0/50 | 10/10 |
| 2368 × 1344 | 7/10 | 7/10 | 0/50 | 10/10 |
| 1792 × 1792 | 9/10 | 9/10 | 0/50 | 10/10 |
| 1536 × 2048 portrait | 6/10 | 6/10 | 0/50 | 10/10 |

The three additional landscape and square dimensions met the
frozen criteria. Portrait missed the minimum and remains outside
scope. Some edited inputs produced a box with insufficient overlap,
including two sharp larger-landscape, four wide and four portrait
cases. No ordinary view was highlighted. The native replay rebuilt
all 350 uploads and exact masks. The scalar reviewer independently
verified all geometry, overlap and advancement decisions.
[Held result](evaluation-panels/image-native-cfa-dimensions-transfer-2026-11-13.json)

## Local app

The optional development check accepts the three passing geometries
under the updated evidence schema. The previously supported landscape
keeps its original schema. Each report uses the actual source-grid
dimensions, shows the native crop when a region is selected, and
scales the context image to half of the submitted width and height.

Four held inputs ran through isolated Full checkout with simulated
$9.99 payment. The browser, API, PDF and JSON agreed on the large
landscape region, the unchanged source, the resized source without
a repeatable pattern, and the portrait outside scope. The selected
crop retained exact 16-bit samples. All four newly signed packages
replayed from their original bytes. Four prior version-one signed
packages also replayed with their original evidence.
[Local app replay](evaluation-panels/image-native-cfa-dimensions-local-2026-11-13.json)

## Replay

Use the retained RawNIND decoder wheel and the camera files from
the publisher links in the source roster. The native runner
reproduces lossless PNG bytes, mask commitments, and native
coordinates. The separate scalar reader uses only the Python
standard library and the [public measurements](evaluation-panels/image-native-cfa-dimensions-measurements-2026-11-13.json).

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze held review; do
  backend/.venv/bin/python -B -m scripts.image_native_cfa_dimensions_transfer \
    "$phase" "$rawnind" "$pixls" "$run" --receipt "$receipt" || exit 1
done
python3 -B -I -S backend/scripts/review_image_native_cfa_dimensions_transfer.py \
  "$run" docs/evaluation-panels/image-native-cfa-dimensions-cc0-roster-2026-11-13.json "$receipt"
```

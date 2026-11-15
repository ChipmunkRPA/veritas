# Locate rotated copies in additional JPEG profiles

Chipmunk Verify previously skipped right-angle JPEG copy comparisons
unless the file used standard quality-75 or quality-90 compression
tables. The native color comparisons can now find similar rotated
regions at standard quality 85 and 95 as well. Both displayed crops
remain in the orientation and pixel coordinates of the upload.

The detector searches for matching detail in a second orientation,
then measures the color difference between the aligned areas and around
their boundaries. This follows the approach of checking explicit
correspondence on the source grid. The experiment uses the project's
existing measurement code, search budget and decision rules. It adds
no pretrained checkpoint or third-party implementation.
[PatchMatch research](https://arxiv.org/abs/2308.04188) ·
[Earlier JPEG rotation check](image-jpeg-turned-copy-2026-11-15.md)

## Matched comparison

The input plan locked 404 deliveries before inference. Twenty-four
credited photo groups supplied quarter turns, half turns and feathered
quarter turns, with five matched controls per source. The controls were
unchanged images, global tone changes, local blur, straight copies and
mirrored copies. Ten architecture and building-exterior photos supplied
separate repeated-structure controls. The two qualities reuse the same
underlying edits and control images.

The edited sources come from a qualified commercially reusable photo
archive. Their encoded hashes and credited contributor groups exclude
recorded overlaps in the earlier published image copy panels. Some
photos may have prior project exposure. Exact source coordinates, file
digests and the input plan are bound to the evaluation receipt. A useful
location requires at least 0.50 intersection over union on **both**
known regions. The published comparison preserves earlier supported
locations and runs rotation only after the translation and block checks
record their outcomes.
[Photo archive terms](image-fresh-source-cache-v39-results.md) ·
[Repeated-structure photo credits](image-affine-native-scale-2026-11-11.md)

| JPEG profile | Useful locations before | Useful locations with rotation | New misplaced edited views | Added flags on matched controls | Added flags on repeated structures |
| --- | ---: | ---: | ---: | ---: | ---: |
| 85 | 1/72 | 50/72 | 2 | 0/120 | 0/10 |
| 95 | 8/72 | 50/72 | 0 | 0/120 | 0/10 |

Gains came from 19 and 16 distinct credited source groups respectively.
The test required gains in at least four groups, allowed up to two
additional ordinary flags and two misplaced edited views, and allowed
zero additional repeated-structure flags. One quality-95 candidate had
ambiguous matches. The two quality-85 errors selected a pair whose crop
boxes did not meet the required overlap. These are results for staged
right-angle copies. They leave arbitrary rotations, custom tables,
native high-resolution phone photos and non-copy edits outside the
evaluated scope. [Numeric receipt](evaluation-panels/image-jpeg-rotation-profiles-2026-11-15.json)

## Local app

The existing development option `VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW`
enables the extended check. It accepts upright RGB 4:2:0 JPEGs with
exactly one of the four tested table pairs and a short edge of at least
384 pixels and a long edge of 513–1,024 pixels. Quality 85 is checked
after the between-block translation comparison finishes without a pair.
Quality 95 remains outside that translation comparison. Its result
explicitly records the scope difference before reporting the separate
rotation observation. Historical evidence uses its original profile
rules when replayed. The visual comparison contributes zero weight to
AI confidence.

The browser and Standard PDF presented both original crops and coordinates
in the two positive cases. JSON carried the same pairs, angles and source
measurements. Signed verification reproduced both sets of measurements
from the submitted JPEGs. The two ordinary controls showed no rotation
finding. Both paired overall AI assessments matched their baseline, and
earlier evidence remained readable by its original identifier. Six
simulated $9.99 checkouts queued zero unpaid jobs.
[Local app receipt](evaluation-panels/image-jpeg-rotation-profiles-app-2026-11-15.json)

## Replay

The public numeric review uses standard Python and the checked-in receipt.
A second native run reproduced all 404 rows byte for byte. Python 3.12
and 3.14 produced the same numeric review.
Native replay needs the retained private source graph, photos and Commons
folder from the earlier studies. Continue to keep source photos and
derivative crops local. Build the frontend and install the repository's
Chromium dependency for browser verification.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
module=scripts.image_jpeg_rotation_profile_transfer
receipt=docs/evaluation-panels/image-jpeg-rotation-profiles-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
backend/.venv/bin/python -B -m "$module" freeze "$source_graph" "$run/inputs" --commons "$commons"
backend/.venv/bin/python -B -m "$module" capture "$run/inputs" "$run/results.json" --workers 3
cmp "$run/results.json" "$receipt"
backend/.venv/bin/python -B -m scripts.image_jpeg_rotation_profile_app \
  "$run/app" "$run/results.json" "$run/inputs" --browser
```

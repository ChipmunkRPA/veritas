# Small rotated and resized copies in JPEGs

The geometric copy check now has a development option for small rotated or
resized regions in a single JPEG. The search uses more low-contrast features
at the original pixel scale. The existing search still supplies the first
results. When it finds no supported pair, the additional search proposes one
pair, aligns the two source areas and checks agreement in their color and fine
detail. Both areas are shown with their original crops and selected pixels.

Patch matching at several scales motivated the separation between the
search for correspondence and the source-pixel comparison. The local
implementation uses OpenCV and requires no additional model or weights.
[Cross-scale patch correspondence research](https://arxiv.org/html/2308.04188) ·
[Earlier geometric copy check](image-affine-copy-transfer-2026-11-11.md)

## Method and source panel

The fallback proposes matches between distinct locations at least 48 pixels
apart. At least four distinct point pairs must support a local rotation and
uniform scale. The same dense luma check as before supplies both masks.
For the fallback, the interior must also have mean absolute RGB difference
of at most 8.5 levels and fine-detail correlation of at least 0.93 after
alignment. These are engineering selection rules for a pixel comparison.
The two areas are undirected, and matching detail can occur naturally.

Ten previously used camera photos supplied the initial trial. The held test
used 18 different public CC0 camera files selected by source hash across six
makes. It included nine previously used buildings with repeated structures
as additional controls. Each held camera file supplies three staged copies
and five ordinary variants at each compression setting. Ordinary variants
include the unaltered photo, local blur, local tone adjustment, local resizing
in place, and a prior whole-image JPEG save. The two delivered qualities reuse
the same photos and edits. The new test photos are disjoint from the pilot by
recorded bytes. Camera model and asset separation cannot establish contributor
independence or generalize to all repeated structures.
[Raw photo source and terms](https://raw.pixls.us/) ·
[Published held roster](evaluation-panels/image-small-affine-jpeg-cameras-2026-11-16.json) ·
[Repeated-structure credits](image-affine-native-scale-2026-11-11.md)

## Held results

A useful result overlaps both known areas with at least 0.30 intersection
over union and covers at least half their changed-pixel support. No delivery
was unassessed. The 306 delivered files are byte-distinct.

| Delivery | Current useful | With extra search and luma alone | With color and detail | Current ordinary alerts | Ordinary alerts with either addition |
| --- | ---: | ---: | ---: | ---: | ---: |
| JPEG75 | 5/54 | 15/54 | 13/54 | 1/99 | 1/99 |
| JPEG90 | 8/54 | 18/54 | 18/54 | 0/99 | 0/99 |
| Combined | 13/108 | 33/108 | 31/108 | 1/198 | 1/198 |

The selected search adds 18 useful results across 10 camera groups. The
strongest gain is on sharp 64-pixel copies, from 5/36 to 19/36. Sharp
44-pixel copies improve from 0/36 to 3/36 and feathered 80-pixel copies
improve from 8/36 to 9/36. Another 23 new highlights overlap both known
areas but fail the two-area localization criterion. The current check
already produced two such partial or misplaced outcomes. These counts
describe staged duplication and its localization. They carry zero AI
authorship-confidence weight.
[Per-delivery measurements and audit](evaluation-panels/image-small-affine-jpeg-held-2026-11-16.json.gz)

Color and fine detail removed a natural-repeat match in development. On
the held panel they removed two useful JPEG75 locations and no ordinary
alerts relative to the added luma search. The trial uses the color and
detail confirmation because the simpler rule flagged a known ordinary
case in development. The held evidence preserves both comparisons.

## Local app

Set `VERITAS_IMAGE_SMALL_AFFINE_COPY_REVIEW=true` together with
`VERITAS_IMAGE_AFFINE_COPY_REVIEW=true` in development. The fallback runs
on supported upright RGB 4:2:0 JPEGs using standard quality-75 or
quality-90 quantization, with a long side of 513–1,024 pixels and a short
side of at least 384 pixels. The held camera deliveries have long sides
of 768 and 1,024 pixels.

Eight simulated USD 9.99 checkouts ran two held JPEGs and their matched
originals with the fallback disabled and enabled. No job queued before
payment. Chromium showed both matching regions and their masks in the
ordinary view. The API, PDF, JSON and signed source replay used the same
coordinates and original crops. Both earlier evidence sets and ordinary
controls remained readable. The native run also reproduced all 306
measurements byte for byte with a different worker count.
[Local application receipt](evaluation-panels/image-small-affine-jpeg-app-2026-11-16.json)

## Reproduce

Download the CC0 RAWs listed in the roster to `$archive` under each
recorded `file` name and verify the SHA-256 values. Use the pinned RAW
decoder wheel and retained repeated-structure deliveries from the
[right-angle JPEG study](image-jpeg-native-rotation-2026-11-15.md#reproduce).
Use a fresh private output directory. The held receipt contains exact
source and mask hashes, candidate transforms, agreement measurements and
integer overlap totals. It contains no source photos or crops.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
start=docs/evaluation-panels/image-small-affine-jpeg
"$runner" -B -m scripts.image_small_affine_jpeg prepare \
  "$archive" "$wheel" "$start-cameras-2026-11-16.json" "$repeated" "$run/inputs"
"$runner" -B -m scripts.image_small_affine_jpeg capture "$run/inputs" "$run/held.json.gz"
python3 -B -S -m scripts.image_small_affine_jpeg review "$run/held.json.gz"
"$runner" -B -m scripts.image_small_affine_jpeg_app "$run/app" "$run/held.json.gz" "$run/inputs" --browser
```

# Locate rotated copies inside a JPEG

This local experiment looks for two regions of the same JPEG that match
after a quarter or half turn. It tests an additional copy comparison
when the existing translation and JPEG-block checks return no pair.
Existing geometric findings still count in both arms. The finding
measures repeated image content. A conventional edit or naturally
repeated detail can produce it.
[Earlier exact PNG check](image-exact-turned-copy-2026-11-15.md)

JPEG compression can change the decoded values when the same patch
moves to another block position. This experiment reuses the project's
local texture descriptors and correlation field. It looks for
correspondences on rotated views, then checks the color agreement
inside each disjoint pair and the differences just outside it. At
least three sides must provide contrasting surroundings. Competing
pairs with different positions or rotations leave the case unresolved.
Both reported boxes use the original JPEG's unrotated pixel coordinates.
The study uses no new weights or third-party detector code.
[JPEG copy-move research](https://library.utia.cas.cz/separaty/2017/ZOI/novozamsky-0483329.pdf)
· [Rotation-aware dense localization research](https://www.iris.unina.it/handle/11588/613096)

## Local comparison

The replay uses 37 distinct credited photos from the retained affine
comparison archive. Original bytes and decoded RGB hashes must match
the source roster. It excludes the source of the prior PNG application
example used during initial development. The fresh deliveries use
576 × 416 or 704 × 480 center crops and standard quality 75 or 90
4:2:0 JPEG encoding. The edits paste a 56 × 72 quarter turn, a 64 × 64
half turn, or a 72 × 64 clockwise quarter turn with a six-pixel blend.
Controls include unchanged content, a global tone shift, local blur,
an unrotated copy, and a mirrored copy. Each copy pair needs at least
0.50 intersection over union on **both** annotated rectangles, allowing
either box order. The controls test whether this particular rotation
check adds an incorrect highlight.
[Retained source comparison](image-affine-copy-transfer-2026-11-11.md)

| JPEG delivery | Existing useful | With the rotation check | Added control flags |
| --- | ---: | ---: | ---: |
| Quality 75 | 11/111 | 85/111 | 0/185 |
| Quality 90 | 18/111 | 82/111 | 0/185 |

At quality 75 the additional check found 26 quarter turns, 24 half turns,
and 24 feathered quarter turns that the existing checks missed. At
quality 90 the gains were 21, 21, and 22. Three staged edits returned
ambiguous candidates. The fallback produced no below-overlap crop
pairs. The existing checks flagged 31 and 36 controls, chiefly the
deliberate straight copies. The two quality profiles share the same
photos and staged changes. The rates apply to these delivery settings
and known copied rectangles. They leave coverage on native phone JPEGs,
other quantization tables, and arbitrary angles unmeasured.
[Source-coordinate receipt](evaluation-panels/image-jpeg-turned-copy-2026-11-15.json)

## Local app

Set `VERITAS_IMAGE_TURNED_JPEG_COPY_REVIEW=true` in development to
include this comparison. It accepts upright RGB 4:2:0 JPEGs with the
tested quality 75 or 90 tables, a shortest side of at least 384 pixels,
and a longest side of 513–1,024 pixels. It runs after the existing
non-mirrored, encoded-block, and between-block checks complete with
no supported pair. Trusted AI Content Credentials still skip these
comparisons. The printed rotation explains how the check compared
the areas. Both crops are displayed in their original orientation.
The finding carries zero weight in the overall AI confidence.

Four simulated USD $9.99 checkouts exercised the earlier quarter-turn
result, both supported rotations with the option enabled, and an unchanged
control. No analysis started before payment confirmation. Headless
Chromium verified the two positions on each original and the crop links.
The Standard PDF and JSON recorded the same coordinates, and the signed
package verifier recomputed the JPEG color measurements and checked both
crops against the upload. The quarter-turn assessment and earlier evidence
remained available, and the unchanged control returned no rotation finding.
[Local application receipt](evaluation-panels/image-jpeg-turned-copy-app-2026-11-15.json)

## Replay

The receipt contains only geometry, scalar comparisons and SHA-256
hashes. Reading it requires standard-library Python. The native command
re-encodes every case and repeats the existing geometric and JPEG
comparisons alongside the candidate.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/image-jpeg-turned-copy-2026-11-15.json
python3 -B -S -m scripts.image_jpeg_turned_copy_transfer "$receipt"
backend/.venv/bin/python -B -m scripts.image_jpeg_turned_copy_transfer \
  "$run/held.json" --archive "$archive" --workers 4
cmp "$run/held.json" "$receipt"
```

Build the frontend and install Playwright Chromium before the browser
option. The local smoke test creates its own database and simulates
checkout without contacting Stripe.

```sh
backend/.venv/bin/python -B -m scripts.image_jpeg_turned_copy_app \
  "$run/app" "$receipt" "$archive" --browser
```

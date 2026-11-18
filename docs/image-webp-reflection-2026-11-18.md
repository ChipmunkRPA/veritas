# Exact reflected areas in a lossless WebP

Chipmunk Verify can compare two areas of a lossless WebP after a
horizontal flip that exchanges left and right, or a vertical flip
that exchanges top and bottom. The reported crops stay in their
submitted orientation. The ordinary view marks both positions and
explains how to align Area B with Area A. The PDF, JSON, and signed
source replay retain the same coordinates and measured pixel counts.
Image editing and deliberate graphic symmetry can produce this
pattern. The comparison cannot determine the direction of an edit or
AI involvement. It contributes zero weight to the AI assessment.

The development option `VERITAS_IMAGE_LOSSLESS_WEBP_REFLECTION_REVIEW`
defaults to false. When enabled, the app keeps the existing lossless
WebP comparisons and checks for a reflected pair when those and the
other copy checks found no match. It uses the original opaque, upright,
single-image lossless WebP with both sides from 512 to 768 pixels. The
two textured RGB areas must have all corresponding pixels equal after
alignment. The three-pixel surroundings must differ on at least three
sides. A case with competing qualifying pairs or too many offsets
reports its coverage without choosing a pair. Existing evidence
versions keep their original behavior. No new model weights are used.
[Earlier lossless WebP comparison](image-exact-webp-copy-2026-11-18.md) ·
[WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)

## Held comparison

The trial reuses 24 BG-REAL camera photos from 24 credited
photographers. Their source digests are separate from the earlier
exact WebP experiment. The photos had appeared in other project
evaluations. Each contributes new staged horizontal, vertical, and
feathered vertical edits, with the pixel masks measured before the
lossless encode. Feathered edges leave a smaller exact inner
crop. The nine repeated-pattern photos from the earlier comparison
contribute ordinary controls only. Their source groups are separate
from the positive bank. [Credited photo sources](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls) ·
[Repeated-pattern sources](image-small-affine-webp-2026-11-16.md#local-comparison)

A useful match needs at least 0.50 intersection over union for both
regions and at least 50% recovery of the actual changed pixels. The
existing comparison includes approximate reflection, translation,
geometric matching, and exact unrotated or rotated WebP. The combined
method preserves every existing result.

| Input | Existing useful pairs | With exact reflection | Added misplaced regions |
| --- | ---: | ---: | ---: |
| Horizontal flip | 0/24 | 19/24 | 0 |
| Vertical flip | 0/24 | 19/24 | 0 |
| Feathered vertical flip | 0/24 | 19/24 | 0 |
| All staged reflections | 0/72 | 57/72 | 0 |

The new pairs span 21 source groups. The remaining 15 staged edits
produced no qualifying exact pair. Among 180 controls there were zero
additional highlights. That includes 33 whole-frame symmetry layouts
where the search abstained because too many offsets could match. The
other 147 controls include untouched photos, tone changes, local blur,
straight copies, and quarter turns. Five separately tested delivery
profiles are outside scope, including lossy, alpha and animated WebP.
The results measure deliberately staged reflected copies on these
lossless files. They give no rate for incidental reflections, generative
edits, or lossy re-encoding.

## Local application verification

An isolated app ran ten simulated paid analyses and queued zero jobs
before the payment webhook. Eight browser, PDF, JSON, and signed
source checks covered both reflection directions, feathering, an
unmodified photo, whole-frame symmetry, competing exact pairs, lossy
WebP, and the earlier quarter-turn behavior. The ordinary browser view
loaded the original crops and linked to their measured coordinates.
Running the same horizontal and vertical files with the setting off
and on produced identical automated assessments and signal values.
[Checkout and report receipt](evaluation-panels/image-webp-reflection-app-2026-11-18.json)

## Offline replay

The [frozen receipt](evaluation-panels/image-webp-reflection-2026-11-18.json.gz)
contains original and delivery hashes, known changed-mask hashes, exact
paired regions, and measurements. It contains no photographs or
original crops. The following command independently recomputes the
overlap and group counts from the recorded measurements.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_webp_reflection \
  docs/evaluation-panels/image-webp-reflection-2026-11-18.json.gz
```

The native replay uses the `intake` folder from the BG-REAL source
archive and the `jpeg` folder from the earlier WebP preparation.
It regenerates each file, verifies the staged bytes and changed masks,
then checks the serving candidate against the frozen locations and
measurements. Local replay matched all 252 deliveries and 72 changed
masks, including all 57 reported reflections. [Camera archive instructions](image-jpeg-frequency-bands-2026-11-17.md#replay-and-independent-comparison) ·
[Repeated-pattern archive instructions](image-small-affine-webp-2026-11-16.md#replay)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.review_image_webp_reflection \
  docs/evaluation-panels/image-webp-reflection-2026-11-18.json.gz \
  --photographs "$camera_intake" --patterns "$earlier_webp_jpeg"
```

# Local rotation evidence for camera-pattern loss

Local interpolation-pattern loss can accompany a generated edit or an
ordinary rotation. A genuine camera patch that is rotated can preserve
some of its Bayer-related periodic structure at a changed angle. This
test looks for that rotated structure inside a proposed region. It
uses the same submitted 16-bit pixels as the previous localization
check. The result carries zero AI-confidence weight.
[Previous four-grid comparison](image-native-generated-grid-transfer-2026-11-13.md) ·
[Resampling research](https://hfarid.org/downloads/publications/sp05.pdf) ·
[Recent spectral correlation research](https://doi.org/10.1109/TPAMI.2026.3721972)

## Frozen test

Measure the log of summed absolute horizontal and vertical curvature in
the red and blue channels. Remove the local mean and compare opposite
red/blue phase at both orthogonal frequencies near the original
period-two camera grid. Sweep angles from 1.75° to 6° in both directions.
Require the two horizontal halves to agree on phase. A location is
withheld when off-axis support reaches 0.08 and exceeds 1.35 times the
same-axis reading. This check runs after the frozen four-grid selection
and earlier global interpolation test. The preceding eight-camera panel
was used to choose the method and point.

The held panel uses eight new CC0 camera models and verified RAW source
bytes. A licensed local generator adds a purple paper hexagon to a
reference crop. The image receives a 128 × 128 native patch using a
frozen half-cell position, with sharp and feathered variants and a
separate linear 8-to-16-bit promotion. Source pixels outside the patch
remain unchanged. Ordinary controls include bilinear and bicubic
rotations at separately fixed angles, cubic translation, quantized
translation, blur, tone and aligned copy. The panel also includes
untouched, fully resized and 8-bit views.
[Held CC0 roster](evaluation-panels/image-native-rotated-stencil-cc0-roster-2026-11-13.json) ·
[Generator terms](image-flux-klein-assets-v271.md)

The advancement rule requires at least eight useful sharp or feathered
locations combined, preservation of every previous useful generated
location including promoted outputs, withholding of at least four and
at least half of the selected rotation controls, and zero unchanged
highlights or assessments on unsupported deliveries. A location needs
at least 0.30 intersection over union and 0.50 changed-pixel recall.

## Held results

| Delivery | Files | Prior highlights | With rotation explanation |
| --- | ---: | ---: | ---: |
| Generated sharp, useful | 8 | 5 | 5 |
| Generated feathered, useful | 8 | 4 | 4 |
| Generated linear promotion, useful | 8 | 7 | 7 |
| Ordinary rotations | 16 | 7 | 0 |
| Ordinary quantized translations | 8 | 3 | 3 |
| Other ordinary views including untouched | 40 | 0 | 0 |

The frozen rotation-specific rule passed. It withheld three
bilinear and four bicubic rotation controls while preserving all
16 previously useful generated locations. Ordinary highlights fell
from ten to three across 64 eligible control files. All eight
8-bit files were unsupported and all eight resized files had no
repeatable interpolation pattern. Two promoted generated cases
had an off-axis peak above 0.08 but stronger zero-angle support,
so the rule kept both.

Visual review confirmed a purple polygon intersected all eight
planted masks. Some polygons reached beyond the target and were
clipped by the fixed blend. The study covers one editor, deliberately
prepared native renders, and the tested small rotations. Remaining
quantized-translation highlights and missed generated edits keep the
combined phase-loss method experimental. The ordinary rotation result
qualifies the cue for further controlled use as an explanation.
A separate replay of the existing axis-aligned phase-shift check
assessed all 16 rotations and selected zero regions. The current
check therefore supplies no crop to which this explanation can
be appended.
The review rebuilt all 104 native cases, and the independent scalar
reader recalculated exact overlap and the frozen decisions.
[Audited results](evaluation-panels/image-native-rotated-stencil-transfer-2026-11-13.json)

## Replay

Use the retained RawNIND wheel, source RAWs from the roster and licensed
local Metal runtime. Scoring and review use the retained generated
bytes and CPU. The independent standard-library audit uses the scalar
ledger and source roster. The published scalar ledger includes source
IDs, input and mask hashes, coordinates, response values and integer
overlap counts. It contains no camera or generated pixels and no
local paths.
[Scalar measurements](evaluation-panels/image-native-rotated-stencil-measurements-2026-11-13.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_native_rotated_stencil_transfer freeze \
  "$run" --cached "$rawnind" --archive "$pixls"
"$cpython3126" -B -m scripts.image_native_rotated_stencil_transfer generate "$run" --runtime "$mlx_runtime"
for phase in score review; do
  backend/.venv/bin/python -B -m scripts.image_native_rotated_stencil_transfer \
    "$phase" "$run" --cached "$rawnind" --archive "$pixls" --receipt "$receipt" || exit 1
done
python3 -B -I -S backend/scripts/review_image_native_rotated_stencil_transfer.py \
  docs/evaluation-panels/image-native-rotated-stencil-measurements-2026-11-13.json \
  docs/evaluation-panels/image-native-rotated-stencil-cc0-roster-2026-11-13.json \
  docs/evaluation-panels/image-native-rotated-stencil-transfer-2026-11-13.json
```

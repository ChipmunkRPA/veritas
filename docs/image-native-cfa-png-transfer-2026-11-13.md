# Local interpolation shifts in native 16-bit PNGs

The camera-grid check located 54 of 60 conventional phase-shifted
128-pixel pastes on ten new camera models. It found 24 of 60 at 64
pixels and zero of 60 at 32 pixels. An ordinary edit can change the
interpolation phase. This observation carries zero AI-confidence weight.
[Measured results](evaluation-panels/image-native-cfa-png-transfer-2026-11-13.json)

The earlier method and 0.25 operating point stayed frozen. This test
encoded every submission as PNG and independently checked the decoded
16-bit samples against the original array. The check requires a single
2048 × 1536 RGB16 image in a noninterlaced PNG with only IHDR, IDAT and
IEND chunks. The reader checks CRCs, chunk order and exact length. Two
spatial folds must independently support the same repeatable red and
blue interpolation pattern before a region can be assessed. The check
uses submitted pixels alone. The panel uses controlled linear renders
of ten distinct, checksum-verified CC0 RAW captures. The analysis pipeline receives the developed PNG. The source RAW files remain local.
[Earlier calibration](image-native-cfa-closure-2026-11-13.md) ·
[Held source roster](evaluation-panels/image-native-cfa-cc0-png-roster-2026-11-13.json) ·
[Original publisher](https://raw.pixls.us/)

| Exact source size | Horizontal odd shift, sharp / feathered | Vertical odd shift, sharp / feathered | Both axes odd, sharp / feathered |
| --- | ---: | ---: | ---: |
| 128 × 128 | 9/10 · 9/10 | 9/10 · 9/10 | 9/10 · 9/10 |
| 64 × 64 | 6/10 · 1/10 | 6/10 · 0/10 | 7/10 · 4/10 |
| 32 × 32 | 0/10 · 0/10 | 0/10 · 0/10 | 0/10 · 0/10 |

A useful result covers at least half the exact changed pixels and
reaches 0.30 intersection over union. The smallest possible proposed
box is 64 × 64, so a 32 × 32 edit cannot meet that intersection
criterion. One 32-pixel edit produced a box that failed it. All six
128-pixel misses came from the same camera source. The result covers
the tested workflows, with known scene and position dependence.

The check highlighted none of the 60 even-aligned local copies or the
40 in-scope ordinary deliveries. Those ordinary controls comprised
untouched renders, global odd-pixel crops, affine gain changes and
localized fractional interpolation. It declined the 30 resized,
8-bit and alpha PNG controls. Even-aligned copies and classical
fills can leave the cue unchanged. The study supplies no performance
estimate for generated replacements or common JPEG uploads.

## Local application

Set `VERITAS_IMAGE_NATIVE_INTERPOLATION_REVIEW=true` on a development
instance and restart the worker. New PNG analyses show **Local pixel
interpolation** in Media Evidence. When a region is selected, the browser
and Standard PDF show its location and a source crop. The PNG asset
retains the original 16-bit samples. The JSON binds its coordinates and
the measured native image to the source. The signed package can replay
the observation and confirm the full-depth crop from the original file.
The report labels a completed scan with no match separately from an
incompatible PNG or an image with no repeatable interpolation pattern.

The local app test used four files from the held panel. The planted
128-pixel paste produced the exact region. The unchanged, resized and
8-bit inputs produced three separate coverage outcomes. The four analyses
queued after simulated checkout. Local Chrome, the PDF, the evidence API
and JSON showed the same results, and each signed package replayed against
its original PNG. The same positive input with the check disabled kept
the customer assessment and omitted the interpolation evidence. The test
isolated the new check and ordinary checkout, evidence, and report code
from the learned image checks.
The application detector separately reproduced all 310 frozen outcomes
and 79 native RGB hashes, including the oversized 32-pixel candidate.
[Local application receipt](evaluation-panels/image-native-interpolation-local-2026-11-13.json) ·
[Detector parity receipt](evaluation-panels/image-native-cfa-serving-parity-2026-11-13.json)

## Replay

Download the ten source files listed in the CC0 roster into the
archive as `sample-<id>.<lowercase extension>` and supply the pinned
RawNIND decoder wheel cache. Run the following phases with a fresh
scratch directory. The native review rebuilds 310 PNG deliveries and
verifies their exact input hashes, changed-pixel masks and coordinates.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze held review; do
  backend/.venv/bin/python -B -m scripts.image_native_cfa_png_transfer \
    "$phase" "$rawnind" "$pixls" "$run" --receipt "$receipt" || exit 1
done
python3 -B -I -S backend/scripts/review_image_native_cfa_png_transfer.py \
  "$run" docs/evaluation-panels/image-native-cfa-cc0-png-roster-2026-11-13.json "$receipt"
backend/.venv/bin/python -B -m scripts.review_image_native_cfa_serving \
  "$rawnind" "$pixls" "$run" "$detector_receipt"
```

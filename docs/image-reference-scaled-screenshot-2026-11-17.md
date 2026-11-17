# Comparing a resized photo inside a screenshot

In the local app, open **Compare with an earlier copy** on a completed image
analysis and upload the earlier photo. The comparison locates the full reduced
photo in an upright PNG or JPEG screenshot. It shows the area compared, the
pixel positions of the differences and crops from both files. The comparison
PDF and JSON use the same positions and images. Differences can come from edits
or interface overlays and carry zero weight in the AI assessment.

The check uses matching detail across the photo to propose its size and
position. It verifies placement throughout the photo and at its perimeter.
For earlier photos with an ICC profile, it converts display colors to sRGB
and preserves the original crop positions. The comparison tests two resize
methods. For JPEG it replays the submitted quantization and chroma sampling on
the screenshot grid. Local differences must survive both deliveries. The thin
border and everything outside the photo are unassessed. The check uses
existing image-processing dependencies and introduces no trained weights.
[Scale-invariant keypoint paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

## Local confirmation

The first frozen replay used 12 previously collected BG-REAL pairs held apart
from six development pairs. The full app replay found that several earlier
photos required ICC conversion. The first panel was rerun with the correction.
A second frozen panel used 12 more pairs with different photos, contributors
and original file hashes. Each source had a staged AI background edit.
Chromium composited the original and edited photos inside four page layouts.
The 864 deliveries span nine cases and four file encodings per source.
Known changed pixels come from the clean and edited PNG captures before
recompression. A correct location requires the exact screenshot photo
rectangle, at least half of the known changed pixels and at least 0.30
intersection over union. Each encoding reuses the same 24 sources.
[Source terms and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

| Delivery | Correct locations on the rerun | Correct locations on the second panel | Edits with unresolved placement | Correctly located ordinary tone edits | Clean captures with a highlight |
| --- | ---: | ---: | ---: | ---: | ---: |
| PNG | 10/12 | 10/12 | 3/24 | 21/24 | 0/48 |
| JPEG 75 | 10/12 | 10/12 | 3/24 | 21/24 | 0/48 |
| JPEG 85 | 9/12 | 10/12 | 4/24 | 21/24 | 0/48 |
| JPEG 90 | 10/12 | 10/12 | 3/24 | 21/24 | 0/48 |

The previous comparison localized zero of these pairs at any encoding. One
accepted edit per format had 42–48% pixel recall. Clean controls include 24
Chromium captures and 24 captures with a different resize filter per encoding.
Four clean placements per encoding were unresolved. All 480 photo-free,
unrelated, cropped, recolored and duplicate cases were unassessed. The replay
produced zero incorrect photo placements. The feature handles roughly 50–94%
proportional downscaling and requires an opaque screenshot in sRGB. These
results cover controlled desktop captures and single-encode screenshots at
the listed JPEG qualities. Mobile capture and multi-stage recompression need
separate tests.

Four simulated USD 9.99 purchases ran the full local analysis flow. Each
created zero analysis jobs before payment. In Chromium, two edited samples
showed the compared area, paired crops and overlay. An unchanged scene
produced zero regions, and a photo-free screenshot was unassessed. The
browser downloaded PDF and JSON for all four comparisons. The exported JPEG
example replayed both crops and all 2,397 marked pixels against both files.
[Local app receipt](evaluation-panels/image-reference-scaled-screenshot-app-2026-11-17.json)

## Replay

The [numeric results and input commitments](evaluation-panels/image-reference-scaled-screenshot-2026-11-17.json.gz)
include source attribution, photo and capture digests, per-case coverage and
localization scores. Raw comparison media stay local. For each panel write
its `source_qualification` record to `qualification.json` and place the
publisher assets at `<folder>/original.png` and `<folder>/edited.png` as listed
in the record. Run from the repository root with the backend dependencies and
Playwright Chromium installed.

```sh
export PYTHONPATH=backend
python=backend/.venv/bin/python
"$python" -m scripts.image_reference_scaled_screen_replay prepare "$run" "$sources"
node backend/scripts/capture_image_crop_multiscale_transfer.cjs "$run"
"$python" -m scripts.image_reference_scaled_screen_replay freeze "$run"
"$python" -m scripts.image_reference_scaled_screen_replay run "$run"
```

Run the export reviewer with the source files, the downloaded JSON and the
same comparison dependencies used to generate it.

```sh
backend/.venv/bin/python -I backend/scripts/review_image_reference_oriented_resize_export.py \
  comparison.json earlier.png screenshot.jpg
```

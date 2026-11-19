# Small pixel changes in an earlier PNG copy

When a submitted image and its earlier copy are upright, opaque, 8-bit RGB
PNGs on the same native grid, Chipmunk Verify can compare their original RGB
values directly. This is a fallback when the existing comparison displays
zero regions. The two PNGs must have matching dimensions and color metadata,
at least 98% exact pixel agreement, and unique matching textured areas in all
four quadrants. Neither original is resampled.

The comparison groups nearby changed pixels and shows up to five regions
with at least 16 changed pixels each. Each region includes the two original
crops and an overlay limited to the pixels whose RGB values differ. The
browser, comparison PDF, and JSON count changed pixels, including those
outside the displayed regions. The comparison measures changes between two
supplied copies. Its observations carry zero weight in the AI assessment.

## Local test

The panel used 12 retained photo source groups and 12 authored screenshot
layouts. Each photo supplied two new small placements using actual pixels
from a published AI-edited counterpart. Each screenshot supplied two ordinary
glyph changes with known positions. These are paired-source tests. The 24
source groups had been used in earlier experiments.

| Group | Pairs | Existing localized | New exact comparison localized |
| --- | ---: | ---: | ---: |
| Pasted AI-edited photo pixels | 24 | 0 | 24 |
| Ordinary glyph substitutions | 24 | 0 | 24 |
| Ordinary local blur | 24 | 2 | 24 |
| Same pixels with separate PNG serialization | 24 | 0 | 0 |
| Whole-image shift, tone change, or unrelated layout | 36 | 0 in this panel | 0 fallback admissions |

All 48 target edits matched their exact RGB-change masks. In one ordinary
blur pair the crops omitted small remnants and the JSON counted those pixels
separately. The shift, tone, and unrelated controls tested the fallback's
same-grid gate. This evaluation does not estimate false-alarm rates for
classifying AI content from a single image.

In Chromium, two simulated paid analyses led to three comparisons. The
photo and the screenshot highlighted 144 and 237 exact pixels. The
identical-copy control displayed zero regions. All three browser downloads
matched the server PDF and JSON, and the cropped evidence replayed against
the two source files. Both signed analysis packages verified against their
submitted originals. The comparisons created zero additional analysis jobs.
[Local comparison receipt](evaluation-panels/image-reference-native-png-2026-11-19.json)

## Verify a comparison

Download the comparison JSON next to the two uploads. From the repository
root, run the reviewer with the earlier and submitted files in that order.
It verifies both original digests, recreates the mask, and compares the
exported overlay and original crops pixel for pixel.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_export \
  comparison.json earlier.png submitted.png
```

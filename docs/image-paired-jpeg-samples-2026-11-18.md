# Local changes between matching JPEG copies

When the earlier-copy comparison finds no region, Chipmunk Verify now
checks brightness and color samples before the JPEG decoder converts them
to RGB. It only tries this on equal-sized, upright, baseline JPEGs with
matching standard quality-75 quantization and 4:2:0 color sampling. At
least 85% of complete 16-by-16 decoded tiles must match exactly, with
unique, detailed matching tiles spread across the image. A copy with
identical decoded samples, or one reproduced exactly by re-encoding the
earlier copy with the matching encoder settings, produces no fallback
finding. [Pillow JPEG decoding documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)

Each finding links the same coordinates in both copies and includes the
mask, both source crops, and the overlay in the web view, comparison PDF,
and JSON export. Retouching and JPEG re-encoding can produce differences.
These regions carry zero weight in the AI assessment. This comparison
requires an earlier copy with the same framing. JPEG compression can spread
differences beyond the exact pixels that changed before encoding.

## Local verification

The baseline was the existing earlier-copy comparison with the optional
JPEG comparisons enabled. The candidate supplies a result only when that
comparison supplies no region. For an edit to count as located, its mask
must cover at least half of the changed pixels, with an intersection over
union of at least 0.30 against the pre-encoding edit mask.

The development bank has 14 scene-text source pairs. The held banks use
16 other scene-text sources and 12 BG-REAL photos. They stage two edge
variants per edit and use the same source photos at both JPEG qualities.
The BG-REAL bank includes two removal methods per photo. Scene-text
reference images can have unknown origin. The experiment measures
changes between copies, rather than authorship or camera originality.
[Scene-text dataset and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [BG-REAL photo attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

| Source bank | Correct locations at quality 75 | Correct locations at quality 90, evaluation only | Conventional retouches highlighted at quality 75 |
| --- | ---: | ---: | ---: |
| Development scene text | 22 → 24 of 28 | 23 → 23 of 28 | 49 → 55 of 56 |
| Held scene text | 22 → 28 of 32 | 27 → 27 of 32 | 50 → 60 of 64 |
| Held BG-REAL removals | 43 → 46 of 48 | 46 → 46 of 48 | 70 → 72 of 72 |

The fallback added nine useful locations across the two held quality-75
banks and preserved the existing useful locations. It also exposed 12
more intentionally retouched copies with ordinary tone adjustment,
copying, or fill. Those are measured differences between the source
and submitted copies. Neither comparison flagged any of the 40 held
identical quality-75 pairs, which cover 28 distinct source JPEGs. The
fallback also produced no regions on 28 comment-only copies, 28 matched
decode-and-re-encode copies, or 28 one-pixel shifts. Shifts failed the
native alignment check. Different encoders and genuine independent
camera versions still need broader evaluation. Quality 90 added no
correct locations when used as a fallback, so the app uses quality 75.

A raw coefficient prototype returned the same aggregate counts but hit
a native allocator crash in a concurrent replay. The application uses
Pillow's existing JPEG decoder. The backend test sends an actual
comparison request and verifies aligned crops, the PDF exhibits and the
same PNG mask in the API and offline JSON review. It also verifies that
the reviewer rejects altered proof fields and mask pixels. An independent
replay of the application comparison on all 256 held quality-75 files
recounted the selected pixels and their overlap with the edit masks.

[Development ledger](evaluation-panels/image-paired-jpeg-samples-development-2026-11-18.json.gz)
· [Held scene-text ledger](evaluation-panels/image-paired-jpeg-samples-scene-2026-11-18.json.gz)
· [Held BG-REAL ledger](evaluation-panels/image-paired-jpeg-samples-bg-2026-11-18.json.gz)

## Reproduce

Prepare the scene-text banks and the `fresh` BG-REAL bank using the
published input recipes. The source files and individual masks remain
local. [Development scene-text intake](image-jpeg-pscc-localization-2026-11-17.md#reproduce)
· [Held inputs and recipes](image-masked-reconstruction-2026-11-17.md#reproduce)

From `backend`, give each delivery directory a new receipt path. The
first command uses the archived benchmark behavior with the fallback
disabled. The second checks source hashes and counts candidate pixels
against the same truth masks.

```sh
.venv/bin/python -B -m scripts.image_text_erase_localization_inputs compare-pairs \
  "$deliveries" "$baseline_json_gz"
.venv/bin/python -B -m scripts.image_paired_jpeg_samples_transfer \
  "$deliveries" "$baseline_json_gz" "$comparison_json_gz"
```

Verify a browser JSON export against the two JPEG files with
`scripts.review_image_reference_export`. Set
`VERITAS_IMAGE_REFERENCE_NATIVE_JPEG=false` to disable the fallback in the
earlier-copy endpoint.

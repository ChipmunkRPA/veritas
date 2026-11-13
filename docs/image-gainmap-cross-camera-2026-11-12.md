# Gain-map localization across camera formats

An expanded research reader admitted eight of ten original HDR JPEGs from
eight contributors and nine camera models. The earlier reader admitted
three. The five added files comprise one Samsung Galaxy S26 Ultra and
four iPhone files with ISO 21496-1 metadata. Two files with legacy Apple
metadata remain unassessed. The three Google Pixel files retain identical
base and gain-map pixels and localization responses. [Licensed original
source pool](https://commons.wikimedia.org/wiki/Category:HDR_gain-mapped_images)
· [Ultra HDR format](https://developer.android.com/media/platform/hdr-image-format)

The reader follows the JPEG Multi-Picture Format offsets and checks that
they bound one upright grayscale gain map after the original image. It
checks marker and metadata lengths, agreement of ISO and XMP values when
both are present, and the Samsung footer where present. It rejects
ambiguous or truncated layouts. The localization trial compares the stored
base and map code values. It performs no HDR rendering or origin attribution.
[Gain-map editing example](https://research.google/blog/hdr-photo-editing-with-machine-learning/)

## Fixed cross-camera result

The two operating points come unchanged from the prior Pixel 6 Pro panel.
Before scoring, the study committed the original bytes, extracted map,
derived base pixels, exact edit masks, and original points. Each admitted
source contributes its unchanged original, seven ordinary processing views,
two deliberately stale-map edits and a no-map control. Crop and rotation
apply to both renditions. The other variants use a controlled base rendition
with the same gain-map pixels. Derived variants are measured as pixel pairs
and are never claimed to be new native HDR carriers. [Previous panel and
frozen points](image-gainmap-localization-2026-11-12.md)

| Cue | Useful highlight edits / 8 | Useful other edits / 8 | Ordinary flags / 64 | Ordinary authors flagged / 7 | Edited views with a misplaced box |
| --- | ---: | ---: | ---: | ---: | ---: |
| Map edges absent from the base | 4 | 3 | 9 | 2 | 6 |
| Dark patches relative to nearby gain | 4 | 5 | 15 | 2 | 3 |
| Fixed union | 6 | 6 | 17 | 3 | 8 |

A useful box covers at least half the changed pixels and reaches 0.30
intersection over union. A union view can have both a useful and a
misplaced box. The union flagged two untouched originals and added useful
locations on both new carrier makes. All eight no-map controls remained
unassessed. No arm met the frozen limits for flagged ordinary authors and
misplaced edits. The reader and both cues remain experimental. The source
roster records per-file CC0, CC BY 4.0 and CC BY-SA 4.0 terms and attribution.
Only coordinates, scalar measurements and commitments are published.
[Public receipt](evaluation-panels/image-gainmap-cross-camera-2026-11-12.json)

## Replay

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_gainmap_cross_camera review \
  docs/evaluation-panels/image-gainmap-cross-camera-2026-11-12.json
```

To reproduce the exact pixel measurements, download each `original_url` in
the receipt to a private directory as `<key>.jpg`. Supply the roster and a new
scratch directory. Keep the downloaded and adapted images local.

```sh
receipt=docs/evaluation-panels/image-gainmap-cross-camera-2026-11-12.json
jq '{records: .freeze.source_roster}' "$receipt" > "$run/sources.json"
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_gainmap_cross_camera freeze "$run/freeze.json" \
  --archive "$archive" --sources "$run/sources.json"
backend/.venv/bin/python -B -m scripts.image_gainmap_cross_camera score "$run/freeze.json" \
  --archive "$archive" --sources "$run/sources.json"
backend/.venv/bin/python -B -m scripts.image_gainmap_cross_camera review "$receipt" \
  --archive "$archive" --sources "$run/sources.json"
```

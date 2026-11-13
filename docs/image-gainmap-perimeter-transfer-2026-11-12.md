# Dark perimeter support in single-file HDR JPEGs

A new check for a dark boundary in the base image localized eight of 16
known edits on the previously scored cross-camera panel. It flagged 38 of
64 ordinary views, including five of eight untouched originals. The earlier
union found 12 of 16 and flagged 17 ordinary views. The new check remains
experimental. [Earlier cross-camera comparison](image-gainmap-cross-camera-2026-11-12.md)

| Result on the same cross-camera inputs | Earlier union | Dark perimeter |
| --- | ---: | ---: |
| Useful high-gain edits | 6/8 | 5/8 |
| Useful arbitrary edits | 6/8 | 3/8 |
| Edited views with at least one misplaced flag | 8/16 | 7/16 |
| Ordinary flags | 17/64 | 38/64 |
| Untouched originals flagged | 2/8 | 5/8 |
| Ordinary authors flagged | 3/7 | 5/7 |
| New ordinary local tone adjustments flagged | 8/16 | 11/16 |

## Measurement and source roles

An HDR gain map carries local brightness corrections. Editing the base
image while retaining its map can introduce an SDR boundary without an
associated gain-map boundary. The fixed candidate tests that geometry
directly. For each rectangle on the earlier 384-pixel inspection grid it
measures signed darkening inside its four sides, relative to samples 4 and
8 pixels outside each side. It admits side samples whose mean absolute
gain-map step is at most 12 stored code values. The third-lowest of 12
positions measures each side, and the lowest three side responses set the
rectangle score. It reports the best rectangle only above the maximum of
24 ordinary views from the earlier four Pixel 6 Pro calibration photos.
These photographs share one contributor and device. The cue operates on
stored code values, with zero AI authorship weight. [Ultra HDR format](https://developer.android.com/media/platform/hdr-image-format)
· [How Google Photos regenerates gain maps after edits](https://research.google/blog/hdr-photo-editing-with-machine-learning/)

The transfer panel uses the same eight admitted cameras, base renditions,
gain maps and exact edited-pixel masks as the earlier comparison. Those
sources and earlier outcomes were known before the perimeter trial. The
new point uses only the separate Pixel 6 Pro calibration controls. The
comparison adds two ordinary local exposure adjustments per admitted
source, one brightening and one darkening, with the gain maps retained.
The native source files carry CC0, CC BY 4.0 or CC BY-SA 4.0 terms, recorded
per source in the earlier receipt. A useful box covers at least half of
the exact changed pixels and reaches 0.30 intersection over union. An edit
can count as both useful and misplaced under the two-check union.
[Calibration sources and methods](image-gainmap-localization-2026-11-12.md)

## Additional carrier intake

Ten additional Wikimedia originals matched their publisher byte counts and
SHA-1 hashes and their local SHA-256 commitments. The publishers declare
each as their own work under CC BY-SA 4.0. Nine carry three-component ISO
gain maps. Seven of those declare a primary MPF size different from the
primary JPEG's actual end. The other two have consistent offsets and lengths
and come from one contributor's Sony ILCE-6400 Adobe Camera Raw exports.
They remain unassessed in the grayscale localization comparison. A single
Pixel 9 grayscale file qualified. It is a different exact upload from an
already exposed contributor and scene. Both checks flagged that original
and its two local exposure controls. This gives no independent accuracy
estimate. Four additional requested originals were rate-limited and do not
enter the denominator. The receipt records each admitted or excluded
download's component count, MPF geometry and gain-map hash.
[Commons originals and per-file licenses](https://commons.wikimedia.org/wiki/Category:HDR_gain-mapped_images)
· [ISO gain-map specification](https://www.iso.org/obp/ui/#iso:std:iso:21496:-1:ed-1:v1:en)

## Replay

The standard-library replay recalculates the fixed point, rectangle
geometry, useful locations, false flags and carrier totals from the saved
records. Native replay recomputes the candidate, new control measurements,
source and mask commitments, and exact new carrier structure. The public
receipt contains scalar measurements, rectangles and commitments. Original
and adapted images stay local.
[Evaluation receipt](evaluation-panels/image-gainmap-perimeter-transfer-2026-11-12.json)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.image_gainmap_perimeter_transfer review \
  docs/evaluation-panels/image-gainmap-perimeter-transfer-2026-11-12.json
```

Use the original archives from the earlier studies for `pixel` and `cross`.
For `fresh`, extract `.freeze.sources` to `roster.json` and download each
`original_url` as `<key>.jpg`. Use a new directory for `run`.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_gainmap_perimeter_transfer run "$run/receipt.json" \
  --pixel "$pixel" --cross "$cross" --fresh "$fresh"
backend/.venv/bin/python -B -m scripts.image_gainmap_perimeter_transfer review "$run/receipt.json" \
  --pixel "$pixel" --cross "$cross" --fresh "$fresh"
```

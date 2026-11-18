# Linked-thumbnail evidence for 10-bit HEIC stills

Chipmunk Verify can compare a supported 10-bit SDR HEIC still with its
linked 8-bit or 10-bit thumbnail. The comparison uses the same decoded
8-bit RGB rendering as the image viewer and reports each crop in its
own pixel coordinates. The ordinary result, PDF and JSON include both
crops and the measured difference footprint. The comparison contributes
zero weight to AI confidence.

The existing option is `VERITAS_IMAGE_HEIF_THUMBNAIL_REVIEW=true`. Files
with a HEVC Main 10 still brand can now pass the normal upload and
readability checks. Thumbnail evidence requires a single suitable linked
thumbnail with a compatible color profile. Explicit PQ and HLG, HDR
display metadata, and 10-bit stills with auxiliary images are outside
the comparison. The usual image analysis can still run on a decodable
file that has no comparable thumbnail. Saved 8-bit evidence keeps its
existing replay format.

## Held comparison

The frozen study uses 11 different held photos and both edit methods from
the pinned [image removal study](image-removal-peer-2026-11-16.md). None
of these 11 photo sources was in the earlier HEIC thumbnail measurement.
The main image and saved thumbnail are encoded from prepared 8-bit JPEG
pixels, with 10-bit samples constructed before HEIC encoding. The transfer
tests this encoding and rendering path. It does not measure the added
precision of native 10-bit camera captures.

The 924 staged files cover two sizes, both edit severities, and controls
with a matching thumbnail after ordinary edits, refreshed edits, local
tone shifts, blur, or copy. A correct result covers at least half the
labeled edit with at least 0.30 intersection over union. The release
criteria require at least 90% retention of the useful 8-bit locations
and at most two additional control flags per profile. In each baseline
case the old and broadened readers returned the same measured evidence.
[Source terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Frozen plan](evaluation-panels/image-heif-depth-plan-2026-11-18.json)
· [Held measurements](evaluation-panels/image-heif-depth-held-2026-11-18.json.gz)

| Thumbnail maximum edge | Primary / thumbnail depth | Correct locations | 8-bit locations retained | Misplaced locations | Controls flagged |
| --- | --- | ---: | ---: | ---: | ---: |
| 192 px | 10 / 8 bit | 35/44 | 34/35 | 3/44 | 0/110 |
| 192 px | 10 / 10 bit | 35/44 | 34/35 | 3/44 | 0/110 |
| 320 px | 10 / 8 bit | 38/44 | 38/38 | 2/44 | 0/110 |
| 320 px | 10 / 10 bit | 38/44 | 38/38 | 2/44 | 0/110 |

All four profiles qualified. The earlier reader skipped 10-bit main
images for this comparison. The five pinned format controls produced no
false region. A separate run with the final HDR and auxiliary-image
guards reproduced the full held receipt byte for byte.

## App verification

Chrome opened the ordinary evidence view and decoded both source crops
and the footprint for each tested thumbnail depth. Both PDFs contained the
same crop pixels, coordinates and rendering note as the JSON export.
The two signed packages replayed against their exact uploaded originals.
Matching thumbnails, a 10-bit file without a thumbnail, and an HDR upload
produced no linked-thumbnail evidence. Disabling the check on the same
changed files preserved their AI assessment. The local checkout used a
Stripe test double for all eight $9.99 USD orders. The worker created no
analysis before a paid event and created one after confirmation. The UI
and export check used an isolated policy that omitted several heavier
learned detectors. An older 8-bit signed package also replayed against its
original source. [App receipt](evaluation-panels/image-heif-depth-app-2026-11-18.json)

## Replay locally

Run the numeric check with standard Python. The native comparison needs
the image removal study's quality-90 deliveries and masks, the pinned
[HEIC format controls](https://heic.fyi/heic-sample-files), and the
libheif shared library bundled with the tested Pillow HEIF 1.8.1 runtime.
Measurements used libheif 1.23.6. Keep the proof images and customer-facing
exports in local test storage.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
plan=docs/evaluation-panels/image-heif-depth-plan-2026-11-18.json
receipt=docs/evaluation-panels/image-heif-depth-held-2026-11-18.json.gz
python3 -B -S -m scripts.image_heif_depth_transfer review "$plan" "$receipt"
backend/.venv/bin/python -B -m scripts.image_heif_depth_transfer run \
  "$plan" "$removal_deliveries" "$format_controls" "$libheif" "$run/held"
backend/.venv/bin/python -B -m scripts.image_heif_thumbnail_app \
  "$run/held/proof-10-10-sharp.heic" "$run/held/proof-10-10-ordinary.heic" \
  "$format_controls/hdr-pq-10bit-bt2020.heic" "$run/serving-10-10" \
  --expected-carrier heif_10bit_sdr_thumbnail
backend/.venv/bin/python -B -m scripts.image_heif_thumbnail_app \
  "$run/held/proof-10-8-sharp.heic" "$run/held/proof-10-8-ordinary.heic" \
  "$format_controls/display-p3-10bit.heic" "$run/serving-10-8" \
  --expected-carrier heif_10bit_sdr_thumbnail
```

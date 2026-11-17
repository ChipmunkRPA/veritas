# Localizing differences in a linked HEIC thumbnail

One uploaded HEIC can contain two versions of a picture. The image worker
now compares the primary image with its linked thumbnail when the file
contains one of each. It aligns the two views and looks for compact
differences that agree across two renderings. The ordinary result, PDF
and JSON show the same main-image crop, thumbnail crop, footprint on the
aligned thumbnail grid and the corresponding pixel coordinates. The two
crops can show what changed visually. The file alone leaves the order and
cause uncertain. The comparison contributes zero weight to AI confidence.

The local setting is `VERITAS_IMAGE_HEIF_THUMBNAIL_REVIEW=true`.
The check requires an 8-bit RGB still with compatible color profiles
and a thumbnail between 64 and 512 pixels on each side. It skips ambiguous
thumbnail sets and files it cannot register. Coordinates follow each
item's decoded orientation. The serving worker now opens HEIC by the same
decoder as payment preflight. The update uses the public thumbnail API
in Pillow HEIF 1.8.1 and requires no model weights.
[Thumbnail API](https://github.com/bigcat88/pillow_heif/blob/master/CHANGELOG.md)

## Local measurements

The staged panel uses 33 source photographs with exact edit masks from two
BG-REAL selections. The main item is made from a pinned JPEG delivery.
Each source has a sharp edit, a feathered edit, a matching original, and
four matching-thumbnail controls with a refreshed edit, retouch, blur or
copy. Each case was delivered with two thumbnail sizes and compression
settings. The staged edits preserve the thumbnail from the original.
A correctly located result covers half or more of the changed pixels
with intersection over union of at least 0.30.
[Source terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) ·
[Insertions and removals](evaluation-panels/image-cross-quality-sources-2026-11-16.json) ·
[Additional removal sites](evaluation-panels/image-removal-peer-sources-2026-11-16.json)

| Edit and thumbnail size | Correct locations | Misplaced locations | Controls with regions |
| --- | ---: | ---: | ---: |
| Insertions, 192 px | 24 of 24 | 0 | 0 of 60 |
| Insertions, 320 px | 24 of 24 | 0 | 0 of 60 |
| Removals, 192 px | 30 of 42 | 2 | 0 of 105 |
| Removals, 320 px | 32 of 42 | 2 | 0 of 105 |

Eight mismatched files and seven controls lacked sufficient common
background to be assessed. The dataset variants share sources. The test
measures whether locations survive the thumbnail comparison and encoding
settings. An edited file with a refreshed thumbnail can have no discrepancy
to report.

Of four published camera originals, an iPhone 13 Pro Max file supplied an
eligible linked thumbnail and had no reported region. Its decoded primary
pixels matched the serving decoder, including orientation. The other
three supplied no linked thumbnail. A separate CC0 format panel covered
29 files. Its wide banner mismatch exceeded the alignment scope and
produced no region. The camera and format files stay out of the repository.
[Pinned camera samples and terms](https://github.com/ianare/exif-samples/tree/5332a9cf0220e9c6c93f88a187daa90808051e10#adding-images) ·
[Format panel](https://heic.fyi/heic-sample-files) ·
[Measured cases](evaluation-panels/image-heif-thumbnail-2026-11-16.json.gz)

### Single-JPEG comparison

A separate experiment tested whether frequency bands alone could improve
localization without a thumbnail or earlier copy. Two small networks trained
from scratch on fine and coarse detail, using a consistency term between
quality 75 and quality 90 views from the same image. The fixed comparison
measured each network separately and averaged their normalized scores.
Sixty original sources supplied fitting data. Twenty-four separate photos
set operating points and chose the fine-detail version for both qualities.
Nine other sources supplied held removals and ordinary controls.
[Multi-frequency inpainting method](https://pubmed.ncbi.nlm.nih.gov/42335053/)

The selected version added zero correct locations in 36 held edited
deliveries. Even before the operating points were applied, every best-ranked
rectangle missed the required overlap in all three variants. The wider
averaged check added 13 ordinary flags across qualities and two misplaced
edit locations. The networks remain experimental. Small edits in a single
recompressed JPEG remain difficult to locate.
[Frozen results](evaluation-panels/image-multiband-localization-2026-11-16.json.gz)

## Replay

The first command recounts regions and control flags using standard Python.
For the native replay, rebuild the exact input JPEGs and masks using the
two source receipts and `scripts.image_cross_quality_sources restore`.
Unpack the CC0 format panel and fetch the four pinned camera files using
the names and hashes in the measurement script. The camera inputs use the
original file from the pinned `heic/` or `heic/mobile/` path.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
python3 -B -IS backend/scripts/review_image_heif_thumbnail.py \
  docs/evaluation-panels/image-heif-thumbnail-2026-11-16.json.gz
"$runner" -B -m scripts.image_heif_thumbnail_panel "$libheif" "$format_panel" "$run/panel" \
  --cameras "$camera_panel" --input bgreal "$fresh_deliveries" --input removal "$removal_deliveries"
"$runner" -B -m scripts.image_heif_thumbnail_app \
  "$run/panel/deliveries/removal-fresh-000-feather-320.heic" \
  "$run/panel/deliveries/removal-fresh-000-ordinary-320.heic" \
  "$format_panel/basic-8bit-srgb.heic" "$run/app"
```

The app replay uses the locally built frontend and a simulated $9.99 USD
payment. The worker created no analysis before paid confirmation and one
after it. Chrome decoded both crops and the footprint at the recorded
dimensions. The PDF embedded those same pixels. JSON and the signed package
verified the exact crops and footprint against the uploaded bytes. The
same mismatched input with the option disabled had the same AI decision.
[Application receipt](evaluation-panels/image-heif-thumbnail-app-2026-11-16.json)

The source sets for replaying the experimental fitting are the retained
[region ranking](image-region-ranking-2026-11-16.md#replay-and-compare-another-check),
[JPEG cross-quality comparison](image-cross-quality-2026-11-16.md)
and [held removal comparison](image-removal-peer-2026-11-16.md).
The script uses their committed JPEGs, masks and baseline readings.

```sh
OMP_NUM_THREADS=2 "$runner" -B -m scripts.image_multiband_localization \
  "$cross_quality" "$ranking" "$removals" "$run/multiband"
```

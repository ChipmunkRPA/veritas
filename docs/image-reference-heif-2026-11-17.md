# Localizing changes between an earlier HEIC and a JPEG export

In a completed analysis of a compatible JPEG, open **Compare with an
earlier copy** and select the earlier HEIC or HEIF. Chipmunk Verify shows
both source crops, the measured area, and highlighted differences on the
submitted image. The comparison PDF and JSON preserve both source hashes
and the crop coordinates on each file's oriented pixel grid. The comparison
carries zero weight in the AI assessment. A local blur, for example, can
also produce a measured difference.

## Supported inputs and comparison

Enable `VERITAS_IMAGE_REFERENCE_HEIF_REVIEW=true` alongside
`VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` in development. The
earlier file must contain one 8-bit RGB image up to 16 megapixels with a
supported color profile. The submitted image must be a moderately compressed
4:2:0 RGB JPEG with sides between 320 and 1,536 pixels. The held deliveries
have maximum sides of about 680, 960, or 1,280 pixels. HDR, multi-image HEIF,
separate captures, and major changes in scene content are outside the tested
scope. The app reports when the pair cannot be assessed.

Matched features estimate the change in framing. A second check requires
shared detail across at least ten of sixteen comparison tiles, with support
in all four quadrants. The method converts embedded color profiles to sRGB,
reduces the earlier photo to the measured export scale before resampling, and
allows an overall color adjustment. It then compares local differences on
the JPEG's delivered pixel grid and maps each earlier crop back to its
full-size coordinates. The check uses no trained model or external service.
[Original feature-matching method](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

## Held-out comparison

The [pinned input recipe](image-cross-quality-2026-11-16.md#exact-input-and-overlap-replay)
provides 24 underlying photographs from prior image evaluations. Six source
photos select the operating point in 192 deliveries. The remaining 18 have
no overlap with selection and provide 576 newly rendered deliveries. The
earlier HEICs in this panel are converted from the reference photographs.
Each group includes offset and simulated tilted crops, two JPEG qualities,
sharp and feathered versions of dataset object removals or insertions, and
ordinary exports. A correct location covers at least half the labeled edit
pixels and achieves at least 30% intersection over union. Counts are by
delivery, so multiple encodes of a source photo are related.
[Photo attribution and dataset terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

| Held case | Assessed | Outcome |
| --- | ---: | ---: |
| Staged edited exports | 130/144 | 126 correct locations |
| Unedited reframed exports, whole-photo color changes, or whole-photo blur | 194/216 | 0 highlights |
| Local blur | 65/72 | 12 highlights |
| Unrelated or half-replaced photos | 0/144 | 0 pairs accepted |

The 126 useful locations include 62 of 72 quality-75 edits and 32 of 36
edits at each of qualities 85 and 90. The largest observed discrepancy at
nine native alignment points per assessed delivery was 1.42 pixels. The
previous endpoint rejected an earlier HEIC upload. These results describe
paired comparisons and do not estimate single-upload detection accuracy.
[Frozen development results](evaluation-panels/image-reference-heif-development-2026-11-17.json.gz) ·
[Held measurements](evaluation-panels/image-reference-heif-held-2026-11-17.json.gz)

A separate check compared three real phone HEICs with unchanged JPEG
exports made on macOS 26.5.2. All three aligned and produced zero regions.
Their earlier copies are approximately 8.5, 12.2, and 15.9 megapixels.
[Camera source terms](https://github.com/ianare/exif-samples#adding-images) ·
[Camera hashes, dimensions and results](evaluation-panels/image-reference-heif-camera-2026-11-17.json)

## Replay

The local app replay covered a held edit, a matched ordinary export, a
real camera export, and an unrelated reference. It queued zero jobs before
three simulated USD 9.99 payments. The browser showed both source crops and
the comparison coverage. It downloaded the PDF and matching JSON. A separate
readback regenerated each PNG byte from the original files and verified both
crop pixel arrays embedded in the PDF. The comparison preserved the saved
analysis result. [Local app receipt](evaluation-panels/image-reference-heif-app-2026-11-17.json)

From the repository root, set `run` to a writable scratch directory. The
receipt recount uses the standard library. Native replay used OpenCV 5.0.0,
Pillow 12.3.0, pillow-heif 1.8.1, and libheif 1.23.6.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
python3 -B -S -m scripts.image_reference_heif_panel \
  "$panels/image-reference-heif-held-2026-11-17.json.gz" --verify
"$runner" -B -m scripts.image_cross_quality_sources restore \
  "$panels/image-cross-quality-sources-2026-11-16.json" "$run/sources"
"$runner" -B -m scripts.image_reference_heif_panel "$run/sources" "$run/development"
"$runner" -B -m scripts.image_reference_heif_panel "$run/sources" "$run/held" \
  --freeze "$run/development/receipt.json.gz"
"$runner" -B -m scripts.review_image_reference_perspective_export \
  comparison.json earlier.heic submitted.jpg
```

To repeat the app test with the recorded camera input, use the pinned
[iPhone 13 Pro Max HEIC](https://github.com/ianare/exif-samples/blob/5332a9cf0220e9c6c93f88a187daa90808051e10/heic/mobile/iphone_13_pro_max.HEIC).
Build the frontend and use the repository's local image models and Chromium.
The exact exported JPEG hash is in the camera receipt. An ImageIO export
on another macOS release may produce different JPEG bytes.

```sh
sips -s format jpeg -s formatOptions high -Z 1120 "$camera" --out "$run/camera.jpg"
"$runner" -B -m scripts.image_reference_heif_app \
  "$run/held" "$run/app" "$camera" "$run/camera.jpg"
```

# Comparing an earlier image after width and height resizing

The earlier-copy comparison can now examine lossless images whose width and
height changed by different amounts. It tries a small set of resize filters
on the whole earlier image and checks them against the submitted pixels.
When every RGB pixel matches, the report states that result even for a
low-detail image. When pixels differ, the comparison requires exact matching
detail distributed across the image. The app shows retained differences in
submitted-image coordinates and displays both crops from their own originals.

The check is available locally with `VERITAS_IMAGE_REFERENCE_AXIS_RESIZE=true`.
It runs when earlier comparisons could not establish matching framing.
It accepts upright, opaque, single-frame RGB PNG and simple lossless WebP.
The earlier image is limited to 768 pixels per side and the submitted image
to 845. Each PNG side must be at least 160 pixels and each WebP side at least
512. Both axes stay within the existing bounded resize range. The feature
carries zero weight in the AI assessment. It compares the two supplied
files, whose authenticity requires separate evidence.
[Pillow resize filters](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)
· [WebP container specification](https://developers.google.com/speed/webp/docs/riff_container)

## Held results

The first 240-delivery trial found 12 of 24 staged removal variants in PNG
and 8 of 24 in lossless WebP, after earlier app checks had skipped both
sets. It missed the planned coverage target for clean copies. Requiring
textured support also left exact but low-detail copies unassessed. The
follow-up added a separate exact full-image test for clean copies and kept
the same rules for nonempty differences. The 512-pixel WebP minimum still
excluded six source groups in the earlier trial's smaller delivery layout.

The confirmation used 12 photos by 12 different photographers outside the
first trial, with identical tested image grids for both carriers. Each
format located 18 of 24 staged removal variants and 9 of 12 ordinary local
blur controls. It verified every one of the 24 unchanged resizes without
marking a region and declined all 72 incompatible pairs per format. That
includes unrelated images, a global tone adjustment, mirrors, partial
replacements and an unsupported resize filter. The two carriers reuse the
same photos. Location required at least half of the known changed pixels
and intersection over union of at least 0.30. The results apply to this
file scope and provide no estimate for arbitrary stretching or editing.
[BG-REAL source archive and credits](https://huggingface.co/datasets/anush-123/bg-real/tree/aaba6cd1ced4ce3c846be9641fed092adb4919d1)

The implementation reproduced all 511 frozen deliveries, including 93
paired native crops and 264 incompatible controls it declined. The numeric
receipt includes the source credits, committed inputs, comparison regions
and truth-overlap counts. It contains no photographs. An offline recount
checks the fixed confirmation roster and recomputes the native mapping and
location counts.
[Evaluation receipt](evaluation-panels/image-reference-axis-resize-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_image_reference_axis_resize \
  docs/evaluation-panels/image-reference-axis-resize-2026-11-18.json.gz
```

To verify the downloaded comparison JSON against both original files,
run the earlier-copy reviewer. It reproduces the selected resize, exact
pixel counts, the raw mismatch hash, retained mask and both native crops.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.review_image_reference_export \
  comparison.json earlier.png submitted.webp
```

An isolated browser replay covered both an edited and an unchanged file
in each format. Both unchanged comparisons verified all 393,537 submitted
pixels. The edited comparisons each displayed one region. Their PDF and
JSON exports preserved the same native crops and overlay. Four simulated
$9.99 payments cleared before the initial analyses started. The four
earlier-copy comparisons used the retained inputs without creating
another analysis job.
[Browser and export receipt](evaluation-panels/image-reference-axis-resize-app-2026-11-18.json)

## JPEG exports of an earlier PNG

The optional `VERITAS_IMAGE_REFERENCE_AXIS_JPEG_RESIZE` setting compares an
earlier opaque RGB PNG with an upright baseline JPEG that was resized by
different amounts in width and height. The feature is available through the
earlier-copy comparison when other checks leave the pair unassessed. It
supports standard quality-75 and quality-90 JPEGs with
4:2:0 color sampling. The earlier and submitted images have minimum sides of
256 pixels, and maximum sides of 768 and 845 pixels, respectively. Both
axes remain between 0.72× and 1.17×, with at least 10% relative axis drift.
[JPEG encoder options](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#jpeg)

The comparison reproduces the resize and one or two JPEG encodes using the
submitted quantization tables. A whole-image pixel match supplies an exact
decoded comparison without a highlight. Otherwise, detailed matching JPEG
tiles must corroborate alignment across the frame. The difference mask shows
measured brightness or color changes on the submitted grid, and each crop
uses the native pixels of its own file. JPEG compression and conventional
retouching can affect these samples. These results carry zero weight in the
AI assessment and require both copies.

The held test uses 12 archived photos from 12 authors outside the two earlier
axis-resize studies. Each quality reuses these photos with known sharp and
feathered local removals, including a second encode of the feathered version.
Controls cover unchanged exports, second encodes, local blur, unrelated
sources, and unsupported deliveries. The archived photos may have appeared
in other project studies. The test measures relative localization with the
matching earlier PNG. [Test photos and source attribution](https://huggingface.co/datasets/anush-123/bg-real/tree/aaba6cd1ced4ce3c846be9641fed092adb4919d1)

| Held measure | JPEG quality 75 | JPEG quality 90 |
| --- | ---: | ---: |
| Staged removals located, existing → combined | 0 → 12 of 36 | 0 → 12 of 36 |
| Distinct photo sources with useful additions | 4 of 12 | 4 of 12 |
| Unchanged exports reproduced without a region | 36 of 36 | 36 of 36 |
| Additional incompatible pairs accepted | 0 of 72 | 0 of 72 |
| Separate conventional blurs located | 3 of 12 | 2 of 12 |

An independent numeric check reconstructed overlap, crop projections,
counts, and the test decision from all 468 reading records. A useful region
covers at least half the known changed pixels and reaches 0.30 intersection
over union. The result applies to the specific source and JPEG profiles
tested here.

The application replay rebuilt all 312 held deliveries and matched their
frozen source, encoded file, mask, and crop coordinates. Four isolated
simulated checkouts exercised the browser, PDF, JSON, and original-file
replay. They covered two edits, an unchanged resave, and an incompatible
pair. Each comparison used the existing paid analysis job.
[Local browser and export receipt](evaluation-panels/image-reference-axis-jpeg-app-2026-11-18.json)

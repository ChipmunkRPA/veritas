# Compare an image with an earlier copy

If you have an independently obtained earlier copy, you can compare it with the submitted image on your own machine. Ordinary processing can produce differences. Establishing the earlier file's provenance or the cause of a difference requires separate evidence.

The [PNG opacity comparison](image-reference-png-opacity-2026-11-14.md)
also measures local transparency differences in a matching earlier copy.

## Local app

Set `VERITAS_IMAGE_REFERENCE_COMPARISON_ENABLED=true` in `backend/.env` and restart the local API. Open a completed image analysis in the normal view, select **Compare with an earlier copy**, then choose a single-frame PNG or JPEG. Each image may have up to 4 million pixels. The submitted source must still be available and match its recorded SHA-256.

The app shows both exact crops, their native positions and the submitted-image difference mask. Select **Highlight difference pixels** on a submitted crop to mark the exact selected pixels in amber. The separate comparison PDF shows the clean crops and the highlighted crop. The JSON download includes the same transparent overlay PNG, its SHA-256 and the crop's native coordinates. Verified integer crops and translations include both overlap areas. The earlier copy is staged temporarily for the request. The comparison carries zero AI confidence weight and requires a completed analysis.

The app can also compare same-framing copies at different pixel dimensions when alignment is supported. It measures differences on the submitted image's pixel grid and keeps the displayed earlier crop in its own native coordinates. [Resized-image evaluation](image-reference-resize-transfer-2026-11-11.md)

A single-frame RGB PNG can also be compared after a whole-image right-angle rotation or reflection. The app requires a unique orientation supported by distributed exact pixel matches. The result gives the matched orientation and keeps each crop in its source orientation. [Orientation evaluation and replay](image-reference-orientation-transfer-2026-11-12.md)

When those comparisons leave a pair unassessed, the app can verify a crop taken after a right-angle rotation or reflection. This path supports a single-frame RGB PNG and an earlier image with a longest side of up to 1,024 pixels. The submitted crop must retain 55–90% of the earlier area. The result shows the earlier overlap in native pixels and keeps both evidence crops in their own source orientation. The app measures differences inside the overlap. [Oriented crop evaluation](image-reference-oriented-crop-2026-11-12.md)

For the tested color JPEG delivery profile, the app can also compare a submitted crop that was subsequently resized. It shows the matched earlier area in its native coordinates and leaves earlier pixels outside that area unassessed. [Crop-and-resize evaluation](image-reference-crop-resize-transfer-2026-11-11.md)

The earlier-copy feature also supports a cropped and resized single-frame RGB PNG when a tested resize reproduces the unchanged pixels exactly across textured areas. This fallback covers earlier images up to 768 pixels on the longest side, submitted images up to 845 pixels, and proportional resizing from 0.72× to 1.17× after rounding. The report shows the matched earlier area and exact pixel and tile counts. Sparse texture, ambiguous geometry, and widespread differences can leave the pair unassessed. The downloaded JSON includes the resize method and a hash of the reconstructed RGB grid so the offline reviewer can reproduce the comparison. [PNG evaluation and replay](image-png-exact-crop-resize-2026-11-11.md)

The PNG fallback also covers a right-angle rotation or reflection followed by cropping and proportional resizing. It requires one exact crop and resize reproduction among the searched orientations. The plain view and PDF show the matched earlier area and the exact submitted-image differences. Both displayed crops stay in their original native grids. [Combined transformation evaluation](image-reference-oriented-resize-transfer-2026-11-12.md)

The tool can locate a native-scale photo inside a PNG screenshot when an earlier copy is available. The matched area in each file and the exact differences inside the photo appear in the normal view and the separate comparison PDF and JSON. Areas outside the matched photo are unassessed. [Screenshot evidence and evaluation](image-reference-screenshot-holdout-2026-11-12.md)

A local browser test used a cropped JPEG with a known edit and an earlier PNG. Simulated Stripe payment released the original analysis, and the supplemental comparison used its retained source. The browser loaded both crops and the mask and downloaded both exports. A separate HTTP test reopened the PDF and matched its embedded pixels and recorded hashes to the JSON. An identical pair returned no selected region. An unalignable pair returned an unassessed result.

The [offline reviewer](independent-testing.md#verify-an-earlier-copy-image-comparison) checks the downloaded JSON against both original images using Pillow. It checks native coordinates, pixel-exact crops, mask totals and the recorded SHA-256 values. When the overlay is present, it verifies every marked and transparent pixel against the mask at the submitted crop's coordinates.

Use the latest reviewer for a native screenshot photo or a rotated, cropped or resized copy. It also opens earlier comparison formats. [Offline reviewer](../backend/scripts/review_image_reference_oriented_resize_export.py)

```sh
backend/.venv/bin/python -I backend/scripts/review_image_reference_oriented_resize_export.py comparison.json earlier.png submitted.png
```

A production-path replay regenerated all 384 held resized PNG and JPEG75 comparisons. Both selection journals matched the earlier frozen hashes. The replay verified 85 overlays and 170 exact native crops across 74 comparisons with selected regions. The other 310 comparisons contained no overlay. [Frozen replay receipt](evaluation-panels/image-reference-overlay-audit-2026-11-11.json) · [Replay runner](../backend/scripts/review_image_reference_overlays.py)

## Command-line comparison

The offline command preserves the original source coordinates and writes a full-size difference mask, both original crops, and JSON with the source and artifact hashes.

```sh
cd backend
python -m app.cli compare-image earlier.png submitted.jpg --evidence-dir comparison
```

The destination must be a new directory. Omitting it prints JSON only. The standard comparison needs the same EXIF-oriented dimensions and measurable alignment. It checks images up to about 16 megapixels. The displayed pixels indicate localized differences that survive the modeled processing. The earlier file's authenticity, the origin of an edit and the full chain of custody require separate evidence.

## Integer crops and translations

```sh
python -m app.cli compare-image earlier.png submitted.jpg --allow-integer-alignment --evidence-dir comparison
```

This option requires corroborated whole-pixel alignment for changed inputs. It measures repeated local agreement before comparing the shared area. JSON records the offset, each image's overlap, and the measured tile locations. The full-size mask uses the submitted image's coordinates. Each paired crop uses its own original coordinates and pixels. Areas outside the overlap stay unassessed. The option supports verified crops covering at least 80% of each image and partial-overlap PNG translations. A JPEG with exposed pixels outside the overlap stays unassessed because its delivery grid cannot be reproduced from the reference alone. Resizing, rotation and fractional shifts may fail the alignment check. [Feature matching method](https://ieeexplore.ieee.org/document/6126544/)

The frozen transfer measured 180 views from 12 rights-qualified source photos already used in unrelated project tests. The crop, shift and control transformations were fixed before scoring. The two formats reuse the same source groups. The prior comparison could not assess the cropped or integer-shifted views in this test. The new option measured 109 exact offsets, with zero incorrect offsets. It declined all 48 fractional shift, rotation, resize and different-photo controls. [Source rights and exact patch construction](image-background-patch-transfer-2026-11-11.md) · [Aggregate](evaluation-panels/image-reference-alignment-transfer-2026-11-11.json)

| Condition | PNG useful locations or highlights | JPEG75 useful locations or highlights |
| --- | ---: | ---: |
| Cropped sharp generated insert | 9/12 | 8/12 |
| Cropped feathered generated insert | 9/12 | 9/12 |
| Cropped original | 0/12 | 0/12 |
| Crop after prior JPEG-only processing | 2/12 | 3/12 |
| Integer shift, sharp generated insert | 9/12 | — |
| Integer shift, original | 0/12 | 0/12, all unassessed |

A useful location overlaps at least 30% of the union with the edited area and covers at least half the known changed pixels. Ten of 12 unchanged crops in each format were comparable. A prior lossy encode can still create an ordinary difference worth checking. The source replay runner freezes input receipts, both implementation digests and the same geometric rule for every view. [Transfer runner](../backend/scripts/image_reference_alignment_transfer.py)

## Fixed comparison

The first comparison used native color differences after smoothing and a robust global tone fit. It located 12 of 12 feathered patches per format on its pilot, and also highlighted all 12 unchanged photos after JPEG75 delivery. The next comparison models the submitted JPEG's quantization and chroma sampling and adjusts large whole-image tone differences. On a separate twelve-photo panel it located 11 of 12 feathered patches in each format and removed all ten direct JPEG75 highlights from unchanged controls in that panel. It still highlighted ten of twelve JPEG75 files with an intermediate lossy encode.

The released local method keeps a local region when it also appears under at least two of three plausible earlier JPEG encodes. The frozen assumed qualities were 80, 88 and 96. The confirmation used eight other rights-qualified photo groups with publisher-derived Stable Diffusion 1.5 insertions, a newly feathered version inside each exact edit mask, and earlier qualities 83 and 92. The test has 128 views. Confirmation photographs were used in earlier unrelated project studies. They were outside both reference-comparison development panels. [Source rights and masks](image-bg-real-insertions-2026-11-11.md) · [Audited aggregate](evaluation-panels/image-reference-comparison-2026-11-11.json)

| Confirmation condition | PNG localized or highlighted | JPEG75 localized or highlighted |
| --- | ---: | ---: |
| Sharp generated edits | 8/8 | 8/8 |
| Feathered generated edits | 8/8 | 8/8 |
| Unchanged originals | 0/8 | 0/8 |
| Earlier lossy encode only | 1/16 | 5/16 |
| Whole-image contrast change | 2/8 | 2/8 |
| Whole-image autoencoder processing | 7/8 | 7/8 |
| Same-size three-pixel translation | 0/8 | 0/8 |

The translation controls were all unassessed. The previous one-delivery method highlighted 11/16 and 13/16 earlier-lossy-encode controls. A useful edit region needs at least 0.30 intersection-over-union and half the known changed pixels. An ordinary control counts if it has any displayed region. The two formats reuse the same source photos. Heavy processing can still create localized differences, and tiny or low-contrast changes may be missed. [Prospective runner](../backend/scripts/image_reference_compare_fresh.py)

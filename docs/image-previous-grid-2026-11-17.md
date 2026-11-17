# Local loss of an earlier JPEG grid after cropping

This comparison checks a single JPEG whose picture was cropped after a previous
save. An earlier quantization pattern can survive across much of the picture
and weaken near a later edit. The method selects a rectangle with less support
for that pattern than the surrounding cells. This can establish a difference
in local compression history. Blur, copying, and other ordinary edits can
produce the same cue. The experiment supplies no AI authorship score.
[Shifted-grid compression research](https://flore.unifi.it/handle/2158/830692)

## Measurements

The check searches all 64 possible origins of the earlier 8 × 8 blocks using
the decoded luma and the final file's actual quantization table. It excludes
coefficients whose earlier quantization spacing is small compared with expected
noise from the final table. It then requires the same previous grid and table
to have support in at least three image quadrants. Each 16 × 16 cell records
the support in two frequency directions. The comparison tests the contrast
alone, its combination with block-boundary evidence, and three region-scanning
variants. One variant also requires agreement from three sides of the region.
There are no additional model weights.

Development used 36 previously acquired BG-REAL photos with generated
removals. The stricter and wider comparison points were fixed separately for
each JPEG quality and region method. They use the second and fourth largest
ordinary source maxima among cases the existing checks left clear. The new
test held back 12 different photos with generated insertions. Exact decoded
and encoded identifiers and close duplicates were checked against earlier study
sources. Each intentionally edited region preserves all pixels outside its
label before the last encode. The sources use published edits with exact
original pixel and semantic masks. [Publisher dataset and terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

The study applies a known off-grid crop to a quality 50, 55, 60 or 65 JPEG,
then delivers quality 75 or 90 copies. It includes sharp and feathered
insertions, unchanged exports, local blur, color edits, local copies, resize
round trips, and direct originals and edits. A useful region needs at least
0.30 intersection over union and half the changed pixels. The comparator
uses the released aligned JPEG regions and the serving regional check when
the JPEG fits its source profile. The quality-90 check uses the same
re-encoding confirmation as the app.
[Existing regional check](image-cross-quality-2026-11-16.md#compare-the-existing-regional-check-on-a-jpeg90-upload)

| Held outcome | Current checks | Current plus stricter three-sided check where clear |
| --- | ---: | ---: |
| JPEG75 edits after an off-grid crop | 5/24 | 5/24 |
| JPEG90 edits after an off-grid crop | 4/24 | 11/24 |
| JPEG90 ordinary controls with a region | 5/72 | 8/72 |
| JPEG90 ordinary blur controls with a region | 1/12 | 4/12 |

The prior grid was correctly identified on all 84 staged JPEG90 deliveries
with an earlier save and offset crop. Direct JPEG90 deliveries produced
no prior-grid proposal. JPEG75 did not retain enough evidence for this
comparison. The stricter three-sided candidate added four sharp and three
feathered regions with no misplaced edit highlights. The wider variant
added one more useful region and two more ordinary highlights. All three
additional ordinary highlights at the stricter point are known local blurs.
This is one confirmation generator family under a staged JPEG history.
[Frozen counts and exact rectangles](evaluation-panels/image-previous-grid-2026-11-17.json.gz)
· [Development measurements](evaluation-panels/image-previous-grid-development-2026-11-17.json.gz)

## Second confirmation

The reserve test uses 12 unused photos from the same pinned publisher
roster. Acquisition excludes the first panel's source and photographer
keys, byte and pixel hashes, and close image matches. The frozen procedure
and operating point are the same. Both sets use the same insertion family.
[Second source roster](evaluation-panels/image-previous-grid-reserve-sources-2026-11-17.json)

| Held outcome | Current checks | Current plus the same three-sided check |
| --- | ---: | ---: |
| JPEG90 edits after an off-grid crop | 10/24 | 16/24 |
| JPEG90 ordinary controls with a region | 8/72 | 10/72 |
| JPEG90 ordinary blur controls with a region | 2/12 | 4/12 |
| JPEG75 edits after an off-grid crop | 9/24 | 9/24 |

Three sharp and three feathered edits gained a correctly located region.
Two deliberate local blurs gained a region. Direct JPEGs, unchanged
exports and whole-image resize controls gained no regions. The local
result describes weaker support for the older compression grid even
when the underlying change is known to be an ordinary edit.
[Second held results](evaluation-panels/image-previous-grid-reserve-2026-11-17.json.gz)

An additional control panel uses ten earlier CC0 camera-derived JPEGs.
Each supplied a routine off-grid crop, a crop after a coarser save, and a
whole-image contrast adjustment after that crop. The check returned zero
regions on these 30 quality-90 exports. It measured an earlier grid on
all 20 deliveries with the added coarser save. The ten crops directly
from the retained JPEGs produced no qualifying earlier grid.
[Camera control measurements](evaluation-panels/image-previous-grid-camera-controls-2026-11-17.json.gz)
· [Source preparation](image-small-affine-webp-2026-11-16.md#replay)

## Local analysis

In development, `VERITAS_IMAGE_JPEG_PRIOR_GRID_REVIEW=true` enables the
check in Full and Deep image analyses. It accepts the tested quality-90
4:2:0 tables in baseline JPEGs and in complete progressive JPEGs with
fixed quantization tables. The quality-85 profile below uses the same
formats. It can report a region when the aligned
JPEG and regional image checks complete without selecting one. Verified
AI credentials stop the analysis before this comparison.

The ordinary view shows a rectangular outline, the unmarked crop from
the submitted image, and a link that focuses the original on the same
pixels. The observation explains that the earlier pattern was weaker in
both measured directions and remained stronger on at least three sides.
The JSON retains the underlying readings and locations. The PDF uses the
same crops and coordinates. Signed source replay recomputes the measurement
and checks the displayed pixels against the original JPEG.

In 13 local checkouts with simulated $9.99 payments, the normal analysis
worker reproduced all eight proposed regions from the second panel. Six
located labeled insertions and two located deliberate blurs. The worker
kept an earlier selected region on the precedence control and reported no
additional region on the ordinary export, direct original, or JPEG75
control. Enabling the check left the automated assessment unchanged on the
paired replay. The queue held zero jobs before payment. The browser, PDF,
JSON and signed-source comparisons reproduced the native coordinates on
representative cases.
[Local checkout and browser receipt](evaluation-panels/image-previous-grid-app-2026-11-17.json)

### Progressive JPEGs and EXIF orientation

The detector measures the JPEG block grid in the stored raster. The app
maps the region to the image as displayed, including the eight supported
EXIF orientations. JSON keeps the grid offset in stored order and the
rectangle in displayed coordinates. Earlier signed reports still verify
under the file eligibility rules used when they were made.
[Pillow orientation mapping](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.exif_transpose)

The format transfer used all 108 quality-90 reserve inputs and 30
camera-derived controls. Each input was delivered three ways with a
lossless progressive scan conversion, an EXIF orientation tag, and both.
The tags cycle through orientations 2–8. All 414 deliveries preserved the
stored luminance, quantization tables, measured status and scalar readings.
All selected rectangles enclosed the same pixels in the new orientation.
Seven staged edits and two deliberate blurs yielded a region per delivery
variant when this check was run by itself. The 30 camera-derived controls
yielded no regions in any of the three variants. These results measure
format handling on the previously evaluated sources.
[libjpeg-turbo conversion tool](https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/doc/usage.txt) ·
[Format transfer measurements](evaluation-panels/image-previous-grid-deliveries-2026-11-17.json.gz)

Seven simulated $9.99 checkouts exercised the normal local worker. One
staged edit was submitted as progressive, rotated baseline, and rotated
progressive JPEGs. The browser, PDF, JSON and signed-source replay agreed
on the highlighted rectangle and the crop in each delivery. An ordinary
progressive rotation produced no highlight. The quality-75 control was
outside the comparison profile. The queue held zero jobs before payment,
and the on/off pair returned the same automated assessment.
[Local delivery replay](evaluation-panels/image-previous-grid-deliveries-app-2026-11-17.json)

### Quality-85 JPEGs

The check now also covers standard quality-85 luma and chroma tables.
It uses a separate operating point for that final compression profile.
Version 3 of the signed source record includes the detected final JPEG
quality when it measures an earlier grid. Earlier signed results replay
with their original eligibility rules and operating point.

The quality-85 point was set on 12 source photos and evaluated on 12
different photos from the reserve insertion panel. A separate transfer
uses 20 additional attributed photos. Acquisition excludes the earlier
quality-85 sources, source photographers, byte and pixel matches, and
near duplicates. LaMa and MI-GAN each supply removals on ten distinct
photos. The staged target replaces a small selected part of a subject.
Direct saves use no planned earlier grid. The staged histories crop a
coarser save before making the local edit and the final quality-85 save.
The same frozen point and 30% intersection over union and 50% recall
requirements apply to both held panels.

| Quality-85 inputs | Existing checks | With the earlier-grid comparison |
| --- | ---: | ---: |
| Staged insertions after cropping | 0/24 correct | 3/24 correct |
| Small staged removals after cropping | 0/40 correct | 10/40 correct |
| Ordinary controls in the insertion panel | 0/72 regions | 0/72 regions |
| Ordinary controls in the removal panel | 0/120 regions | 6/120 regions |
| Camera-derived crops, resaves and whole-image contrast changes | — | 0/30 regions |

All 13 useful regions matched staged edits from seven source photos.
No edited file received a misplaced region. Four deliberate local blurs
and two ordinary local copies from four source photos also showed a
weaker local grid. No unchanged ordinary export, direct original,
global resize or whole-image contrast adjustment was flagged. The
separate camera test applies the earlier-grid check itself. These results
support local compression-history evidence under the tested conditions.
The finding carries zero weight in the AI assessment. It can miss edits
after the earlier grid is lost or where the upload has only one save.

The serving code reproduced all 288 insertion and removal results from
the original JPEG bytes. Five lossless progressive conversions with
EXIF rotation preserved the result and the corresponding crop pixels.
Six simulated prepaid analyses reproduced all three insertion regions
and withheld evidence on the two controls. Two visible browser findings
matched the PDF, JSON and signed source replay. The queue held zero
jobs before payment, and the on/off pair returned the same automated
assessment.
[Serving and format readback](evaluation-panels/image-previous-grid-85-source-replay-2026-11-17.json)
· [Local browser and checkout receipt](evaluation-panels/image-previous-grid-85-app-2026-11-17.json)

[Quality-85 selection](evaluation-panels/image-previous-grid-85-selection-2026-11-17.json.gz)
· [Insertion results](evaluation-panels/image-previous-grid-85-insertions-2026-11-17.json.gz)
· [Removal results](evaluation-panels/image-previous-grid-85-removals-2026-11-17.json.gz)
· [Removal reservation](evaluation-panels/image-previous-grid-85-removal-reservation-2026-11-17.json.gz)
· [Removal source roster](evaluation-panels/image-previous-grid-85-removal-sources-2026-11-17.json.gz)
· [Camera controls](evaluation-panels/image-previous-grid-85-cameras-2026-11-17.json.gz)

To reproduce, restore the two published insertion source rosters as
`$run/selection-sources` and `$run/insertion-sources`. Prepare the
earlier removal-study archive and preserve its intake as `$previous/intake`.
The removal reservation binds the earlier JPEG boundary intake and both
insertion intakes by SHA-256. The two held replays verify 576 input and
parent bindings and independently recompute all candidate overlaps.
[Insertion bindings](evaluation-panels/image-previous-grid-85-insertion-bindings-2026-11-17.json.gz)
· [Removal bindings](evaluation-panels/image-previous-grid-85-removal-bindings-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
reader=scripts.image_previous_grid_panel
panels=docs/evaluation-panels
stem=image-previous-grid-85
"$runner" -B -m scripts.image_jpeg_band_sources acquire "$panels/$stem-removal-reservation-2026-11-17.json.gz" \
  "$retained/deliveries/manifest.json" "$run/removal-intake" \
  --previous-intake "$previous/intake/intake.json" \
  --previous-intake "$panels/image-jpeg-boundary-source-intake-2026-11-17.json.gz" \
  --previous-intake "$panels/image-previous-grid-sources-2026-11-17.json" \
  --previous-intake "$panels/image-previous-grid-reserve-sources-2026-11-17.json"
PYTHONPATH="$ort:backend" "$runner" -B -m scripts.image_jpeg_band_sources prepare \
  "$run/removal-intake" "$editors" "$run/removal-sources" --partial --as-pairs
"$runner" -B -m "$reader" measure "$run/selection-sources" "$run/q85-selection" --quality 85
python3 -B -S -m "$reader" freeze "$run/q85-selection/measurements.json.gz" "$run/q85-freeze.json"
for profile in insertion removal; do
  "$runner" -B -m "$reader" measure "$run/$profile-sources" "$run/q85-$profile" --quality 85
  python3 -B -S -m "$reader" assess "$run/q85-freeze.json" \
    "$run/q85-$profile/measurements.json.gz" "$run/q85-$profile/result.json.gz"
  "$runner" -B -m "$reader" bindings "$run/q85-$profile/result.json.gz" "$run/q85-$profile/bindings.json.gz"
done
npm --prefix frontend run build
"$runner" -B -m scripts.image_previous_grid_app "$run/q85-insertion" "$run/q85-app" --quality 85 --browser
```

## Independent replay

The public source roster contains the original and edited file commitments,
attribution, and the semantic mask paths. The portable manifest binds 216
JPEG deliveries and 432 input and parent file hashes. A separate readback
also checks the exact truth masks and recomputes each proposed rectangle's
overlap. A fresh download and replay reproduced all per-file measurements.
Reviewers can supply a different algorithm using these same input
files and native rectangles. The repository includes numeric receipts.
[Source roster](evaluation-panels/image-previous-grid-sources-2026-11-17.json)
· [Portable file bindings](evaluation-panels/image-previous-grid-bindings-2026-11-17.json.gz)
· [Portable bindings for the second panel](evaluation-panels/image-previous-grid-reserve-bindings-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
reader=scripts.image_previous_grid_panel
panels=docs/evaluation-panels
python3 -B -S -m "$reader" review "$panels/image-previous-grid-2026-11-17.json.gz"
"$runner" -B -m "$reader" restore "$panels/image-previous-grid-sources-2026-11-17.json" "$run/source-media"
python3 -B -S -m "$reader" freeze "$panels/image-previous-grid-development-2026-11-17.json.gz" "$run/frozen.json"
"$runner" -B -m "$reader" measure "$run/source-media" "$run/deliveries"
python3 -B -S -m "$reader" assess "$run/frozen.json" "$run/deliveries/measurements.json.gz" "$run/deliveries/result.json.gz"
"$runner" -B -m "$reader" bindings "$run/deliveries/result.json.gz" "$run/bindings.json.gz"
```

For the second panel, use `image-previous-grid-reserve-sources-2026-11-17.json`
in the restore command. The operating point comes from the same development
receipt. Run the serving replay against that panel's delivery folder.

```sh
npm --prefix frontend run build
"$runner" -B -m scripts.image_previous_grid_app "$run/deliveries" "$run/app" --browser
```

For the camera controls, use the ten `*-medium-ordinary.jpg` files and
`freeze.json` from the retained camera preparation described above.

```sh
"$runner" -B -m "$reader" camera-controls "$camera" "$run/camera-controls"
```

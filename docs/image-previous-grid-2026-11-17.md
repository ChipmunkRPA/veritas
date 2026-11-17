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

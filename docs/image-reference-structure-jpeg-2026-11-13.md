# Earlier-copy JPEG comparison after overall tone changes

The previous color comparison found more object removals after a global
adjustment. Ordinary grades and resize round trips also produced broad
highlights. This successor checks whether each retained color difference is
near a change in local image structure after accounting for JPEG delivery and
overall tone. It uses two supplied copies and has no effect on AI confidence.
[Previous paired test](image-reference-monotone-jpeg-2026-11-13.md)

## Method and sources

The earlier checks still run first. On an otherwise unassessed pair, require
matching framing from detailed spatial tiles. Fit monotone color responses on
separate checkerboard tiles, then test each fit on the opposite tiles. The
same sharpness check as before excludes widely softened deliveries. For both
modeled deliveries, compare the direction and energy of pooled RGB gradients
against the submitted image. A retained color difference must lie within nine
native pixels of a structural change supported by both deliveries. The check
withholds a result when the selected regions cover more than 10% of the image.
Relative local structure helps isolate real changes from broad photometric
differences. This implementation uses neither paper weights nor source code.
[Local image correspondence research](https://www.cs.cornell.edu/~rdz/Papers/ZW-ECCV94.pdf) ·
[Global nuisance modeling research](https://ira.lib.polyu.edu.hk/bitstream/10397/115980/1/Zhang_Unsupervised_Global_Difference.pdf)

The eight-photo sharpness confirmation from the previous experiment is now a
consumed development set. Before new scoring, the held freeze selected 25
other qualified LaMa photos, 14 from the primary v225 archive and 11 from the
v262 backup archive. It excludes photo IDs, contributor identities and original
asset hashes from development and the earlier 13-photo tone study. The photos
have previous exposure under other project methods. Each held group includes
sharp and feathered removals, matched ordinary local blur, tone and copy,
recompression, global grade, two resize kernels, whole-image blur, and three
incompatible pairings. The exact changed-pixel mask is calculated before the
final JPEG encode. A useful region recovers at least half of those pixels
with at least 0.30 intersection over union. The source photos carry CC BY 2.0
attributions, and the publisher separately documents the edited assets and
license terms. This method adds no model weights.
[Primary source qualification](image-bg-real-transport-v225-result.md) ·
[Backup source qualification](image-bg-real-backup-intake-v262-result.md) ·
[Publisher attribution and terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

## Held results

| Check | Useful sharp locations | Useful feathered locations | Ordinary delivery highlights | Incompatible pairs assessed |
| --- | ---: | ---: | ---: | ---: |
| Current paired check | 3/25 | 3/25 | 42/150 | 0/75 |
| Monotone tone fallback | 16/25 | 15/25 | 90/150 | 0/75 |
| Sharpness-checked fallback | 16/25 | 15/25 | 61/150 | 0/75 |
| Structure-supported fallback | 20/25 | 20/25 | 42/150 | 0/75 |

The new comparison added eight locations of each edit type on the 14 primary
photos and nine of each on the 11 backup photos. It preserved all incumbent
locations, added zero ordinary delivery flags, and kept the number of
flagged ordinary source groups at 16 of 25. Mean intersection over union
across all 50 sharp and feathered edits, including misses, rose from 0.078 to
0.671. It also located 20 of 25 ordinary local copies versus 4 of 25, with
little additional recovery on blur and isolated tone changes. These are
actual differences between files regardless of how they were made.
The eight-photo development rerun found 6 of 8 locations in each edit type
versus 1 of 8 and added zero ordinary flags. Separate native review rebuilt
all 462 JPEG pairs and verified the exact masks. The serving implementation
matched every research mask on 309 cases that reached the fallback.
[Development receipt](evaluation-panels/image-reference-structure-jpeg-development-2026-11-13.json) ·
[Held receipt](evaluation-panels/image-reference-structure-jpeg-held-2026-11-13.json)

The complete app comparison route, including the earlier alignment and JPEG
fallbacks, reproduced the same held improvement. On the 14 primary photos,
useful locations rose from 2 to 10 for each edit type, with 34 of 84 ordinary
deliveries highlighted under both settings. On the 11 backup photos, locations
rose from 1 to 10 for each edit type, with 8 of 66 ordinary deliveries
highlighted under both settings. All 75 incompatible pairs were declined.
The new route assessed 116 of 350 held pairs and left every successful
incumbent comparison intact. [App route replay receipt](evaluation-panels/image-reference-structure-jpeg-serving-2026-11-13.json)

## Local app and replay

Set `VERITAS_IMAGE_REFERENCE_STRUCTURE_JPEG=true` alongside the existing local
image comparison option. In **Compare with an earlier copy**, provide a
matching earlier JPEG. Supported pairs use the same 512–768 pixel dimensions,
upright RGB, standard 4:2:0 baseline compression, and quality tables 75, 90 or 95
for the earlier copy and 75 or 90 for the submitted image. The browser shows
both native crops, a full mask and a toggleable overlay. PDF and JSON use the
same source regions. Set the option to `false` to roll back. A comparison can
miss small changes and does not establish either file's provenance.

The authenticated endpoint tests confirmed scope and rollback, no saved
assessment mutation, original crops and PDF asset pixels, and rejection of a
modified JSON proof. The focused image suite passed 42 backend tests and 18
frontend tests. Chrome opened the held fixture through the plain assessment,
loaded the region and mask, and downloaded byte-matching PDF and JSON evidence.
The browser run used an isolated, already-completed analysis fixture.

The source runner accepts the two existing qualification directories and
retained earlier plans. Use a separate empty output directory for each stage.
The exact review reconstructs the native JPEG pair and truth from the retained
publisher assets.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_structure_jpeg_transfer "$phase" held \
    --source-roster "$v225" --backup-roster "$v262" --later-f5 "$later_f5" \
    --external "$new_editor_transfer" --output "$run" --publication "$receipt" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_reference_structure_route \
  --source-roster "$v225" --backup-roster "$v262" --later-f5 "$later_f5" \
  --external "$new_editor_transfer" --held "$run" --receipt "$app_receipt"
backend/.venv/bin/python -B -m scripts.review_image_reference_export "$export" "$earlier" "$submitted"
```

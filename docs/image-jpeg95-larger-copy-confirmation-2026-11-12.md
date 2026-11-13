# Larger copies in standard JPEG95 uploads

Extending the between-block copy check to standard JPEG95 added no useful
locations on 16 new sharp copies or 16 matching feathered copies. The
current checks located seven sharp and six feathered copies. The extension
added no flags on 56 ordinary deliveries. These results leave the JPEG95
extension outside the app. [Earlier JPEG95 comparisons](image-local-edit-candidates-2026-11-12.md)

| Held outcome | Current | With JPEG95 between-block comparison |
| --- | ---: | ---: |
| Useful sharp copies | 7/16 | 7/16 |
| Useful feathered copies | 6/16 | 6/16 |
| Ordinary views with a flag | 3/56 | 3/56 |
| Repeated-structure views with a flag | 0/8 | 0/8 |

The three ordinary views flagged by the existing geometric check come
from the same source photo. The existing routes also returned four
misplaced pairs for sharp copies and three for feathered copies. The new
path reached and withheld a finding on the other five sharp and seven
feathered views. In total it returned no pair on all 65 inputs that reached
it. This study does not show that enabling the new profile would recover
the smaller pairs from the earlier trial.

## Fixed comparison

The two standard quality-95 quantization tables, upright RGB 4:2:0 JPEG
delivery, minimum crop side of 56 pixels and existing candidate ranking
were fixed before measurement. The pre-existing translation and encoded
brightness-block routes ran first. Each of 16 source photos contributed a
sharp and an eight-pixel feathered copy of the same 88 × 96 pixel region.
Both use an integer shift off the JPEG block grid. The panel also includes
each source re-encoded without a copy and two feathered ordinary adjustments,
local tone and local blur, in the same target box. Eight retained licensed
Wikimedia photos with architectural or natural repetition add hard controls.

The 16 photos come from source and credited-author groups outside the six
earlier copy study freezes listed in the receipt. They were drawn from the
existing source cache and may have other project exposure. Commercial reuse
is permitted under the listed CC BY and CC BY-SA terms. All generated test
views, source bytes and exact changed-pixel masks were committed before
scoring. A pair is useful when both reported crops reach 0.50 intersection
over union with the known source and destination, and the destination
recovers at least half the exact changed pixels. The new profile needed
four added useful source groups, including two useful feathered locations.
At most one additional ordinary photo group, no additional repeated-structure
group and at most one incorrectly located edit were allowed.
[Source cache qualification and licenses](image-fresh-source-cache-v39-results.md)
· [Wikimedia control credits](evaluation-panels/image-affine-native-scale-2026-11-11.json)
· [Existing native comparison](image-jpeg-offgrid-quality85-2026-11-12.md)

## Replay

Native replay rebuilt the exact 88 JPEG deliveries and checked all 46
reported source crops. Moving a reported crop failed native verification
on both observed incumbent routes. The receipt includes scores, locations
and commitments. Original and derivative photos remain local.
[Evaluation receipt](evaluation-panels/image-jpeg95-larger-copy-confirmation-2026-11-12.json)

Use the retained cache group file, attributed Wikimedia JPEG directory and
six earlier study freezes identified by digest in the receipt. Set `run`
to a new private directory.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_jpeg95_larger_copy_confirmation \
    "$stage" "$cache" "$public" "$run" "${earlier_freezes[@]}" || exit 1
done
```

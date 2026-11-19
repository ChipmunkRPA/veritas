# Shifted-grid comparison for small edits in infographics

Two recent forensics studies tested overlapping image tiles and targeted
small edits that were difficult to locate. We tested one idea from that
work against our frozen paired residual classifier. The experiment asked
whether a proposed region stayed suspicious when the cell boundaries
moved. It used an independent implementation and reused no weights or
code from either paper. [Team MSU visual tampering report](https://arxiv.org/html/2609.38391) ·
[Evidence-guided text forensics](https://arxiv.org/html/2609.02097)

The original search uses a single grid of 16-pixel cells. The shifted
search scores the same rectangle in four views with grid origins
`(0,0)`, `(8,0)`, `(0,8)`, and `(8,8)`. It uses the lowest of the four
area-weighted scores. A separate control runs the original grid over
the same inner image area, so any gain from discarding edges is visible.
Each comparison uses the same fitted classifier and its own frozen
point from the same eight authored development-layout families.

The held intake found 11 eligible new English infographics from the
400-row MIT-licensed test split at publisher revision
`3e0c54f7a6aeca5fb3e12758cb93d0be14310f55`. It excluded all 32 source IDs
and original-image hashes from the previous fit and test sets. Before
any held score, a second intake considered explicit printed numbers and
dates as well as words. It added no eligible sources. The resulting
pilot includes published edit masks covering 0.25–12% of the image,
generated before images, matched ordinary blur and tint, and distinct
authored controls. [Dataset terms](https://huggingface.co/datasets/ReyChiaro/MaskEdit-10k#licensing-and-rights) ·
[Earlier residual comparison](image-paired-residual-transfer-2026-11-19.md)

A useful region covers at least half the known edit pixels and reaches
at least 0.30 pixel intersection-over-union. The comparison first
measures each method's best rectangle, then applies its frozen internal
point. Each carrier gave the same counts in all three arms.

| Delivery | Correct direct edit highlights | Misplaced direct highlights | Misplaced mask-confined highlights | Authored controls flagged |
| --- | ---: | ---: | ---: | ---: |
| PNG | 0/11 | 5/11 | 7/11 | 0/36 |
| JPEG quality 90 | 0/11 | 9/11 | 8/11 | 0/36 |

The shifted search changed 68 top rectangles in PNG and 48 in JPEG
compared with the same-area control. None qualified as a useful location
on a direct or mask-confined edit, even before score filtering. The
generated before and ordinary variant views still produced 21 alerts
out of 33 PNG deliveries and 24 of 33 JPEG deliveries in each arm.
This leaves the shifted check out of the app. The small panel covers
one publisher and a single type of image. Its authored ordinary controls
give no estimate of camera-image false alarms.

The second geometry pass recomputed mask overlap for all 546 selections
from the retained originals. The public [summary](evaluation-panels/image-shifted-grid-transfer-2026-11-19.json)
and [anonymized case records](evaluation-panels/image-shifted-grid-transfer-2026-11-19.jsonl.gz)
contain score and geometry counts for an independent arithmetic replay.

```sh
python3 -I -S backend/scripts/review_image_shifted_grid_transfer.py \
  docs/evaluation-panels/image-shifted-grid-transfer-2026-11-19.json \
  docs/evaluation-panels/image-shifted-grid-transfer-2026-11-19.jsonl.gz
```

# Can a longer list recover missed infographic edits?

The paired residual check missed every correctly labeled region at its
top rank in two earlier infographic tests. This diagnostic checked whether
the model had a useful region just below its first choice and whether
comparing each rectangle to its surroundings improved the ranking. Both
arms retained the same fitted classifier, rectangle sizes and carriers.
[Earlier comparisons](image-paired-residual-transfer-2026-11-19.md) ·
[Shifted-grid follow-up](image-shifted-grid-transfer-2026-11-19.md)

The second arm subtracts the positive average of a ring around each
proposed rectangle. The ring must contain at least as many grid cells as the
rectangle. It subtracts nothing if the ring gives a negative response.
The evaluator checked the top single rectangle and the top three
nonoverlapping rectangles, with the same internal point for both list
lengths. Each carrier and arm used the second-highest ordinary source
maximum across the eight existing development layouts as its own point.
The use of paired processing controls follows the evaluation question
raised by BG-REAL. The trial reused no code or weights from that paper.
[BG-REAL matched-control analysis](https://arxiv.org/html/2607.26232v2#S7)

The retained panel includes 31 publisher groups and 24 authored ordinary
layout groups, all evaluated before in this project. Each carrier has 31
direct edits, 31 edits confined to a known region, 93 generated-original or
same-site ordinary variants, and 72 authored controls. A useful result
covers at least half the labeled pixels with at least 0.30 intersection
over union. The test kept both the number of correctly located files and
the total number of wrong boxes on edited files.

| Carrier and selection | Correct direct edits | Wrong boxes on direct edits | Authored control files flagged |
| --- | ---: | ---: | ---: |
| PNG, original top one | 0/31 | 20 | 1/72 |
| PNG, original top three | 0/31 | 47 | 1/72 |
| PNG, local contrast top one | 0/31 | 4 | 15/72 |
| PNG, local contrast top three | 0/31 | 5 | 15/72 |
| JPEG90, original top one | 0/31 | 29 | 12/72 |
| JPEG90, original top three | 0/31 | 81 | 12/72 |
| JPEG90, local contrast top one | 0/31 | 26 | 21/72 |
| JPEG90, local contrast top three | 1/31 | 66 | 21/72 |

The three-region local contrast list put a useful rectangle among the top
three on three PNG edits before applying the point. All three scores fell
below it. Neither carrier located any of the 31 confined edits. Increasing
the list length alone keeps the same set of flagged files because the top
score still controls whether any region crosses the point. It increased
the number of wrong highlights per file without finding an additional
location. The contrast change recovered one direct JPEG edit and added nine
authored control alerts compared with the existing ranking. Neither result
qualifies for a fresh trial of this recipe or changes the app.

The local run reproduced the saved highest score and its exact rectangle
on all 454 carrier deliveries. A separate recount used the retained masks
to verify all 2,724 candidate intersections. The downloadable receipt
contains the anonymous counts, boxes and numerical responses. It excludes
source images and extracted text.
[Replay receipt](evaluation-panels/image-ranked-context-2026-11-19.json.gz)

```sh
python3 -I -S backend/scripts/review_image_ranked_context.py \
  docs/evaluation-panels/image-ranked-context-2026-11-19.json.gz
```

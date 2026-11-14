# Testing whether neighboring sentences drive the writing score

Discounting the effect of neighboring sentences found **no additional
rewrite locations** beyond the earlier sentence-omission check. Both new
rules flagged two additional ordinary eLife works and three wrong
sentences in edited eLife files. Neither rule enters the app.

## Fixed comparison

Earlier tests showed that a rewrite can shift the classifier response of
an unchanged sentence beside it. For each complete native center we use
six measured readings from the same licensed classifier: each of the
three sentences alone, the two adjacent pairs, and the whole triple.
The first rule reduces the smaller observed omission effect by the
difference between the two sides. The second also penalizes a disagreement
between the measured triple and the simple estimate obtained by adding the
pair scores and subtracting the isolated center. All arithmetic uses the
logged odds of the recorded response. The scores can depend on the same
language features, context length and wording. This tests context sensitivity
with the project's own scoring code. [Mixed-authorship segmentation research](https://arxiv.org/abs/2609.33787)

All scores and locations come from pinned existing receipts. Fourteen
older F1000Research work groups, each with two historical versions, set
each point to the second-highest genuine group maximum, floored at zero.
We fixed those points before evaluating the new selector on the two
previously exposed panels. Both exact input hashes and sentence offsets
must agree between the saved ledgers. The unit is the exact native changed
sentence. Both Granite and Qwen rewrite cases qualified. The old context
and position checks retain their previously reported location criterion.
[Prior omission scope and sources](writing-omission-attribution-2026-11-13.md)
· [Exact response binding and terms](writing-context-agreement-2026-11-14.md)

| Outcome | F1000Research, 18 edits and 9 ordinary works | eLife, 21 edits and 12 ordinary works |
| --- | ---: | ---: |
| Incumbent context and position checks, useful locations | 4 | 0 |
| With the earlier omission check, useful locations | 4 | 1 |
| With the side agreement rule, useful locations | 4 | 1 |
| With side and triple agreement, useful locations | 4 | 1 |
| Ordinary works flagged by incumbent checks | 2 | 2 |
| With the earlier omission check, ordinary works flagged | 3 | 2 |
| With either new rule, ordinary works flagged | 2 | 4 |

All 39 edited sentences received complete measurements. Before applying
any point, the omission check ranked 12 edited sentences first, the
side agreement rule ranked 11 first, and the side-and-triple rule
ranked seven first. The new rules reused the earlier useful eLife
location and added no unique useful work. This is a replay on retained
English scholarly excerpts. It used no new inference and gives no
standalone authorship accuracy estimate.

## Replay

The replay verifies each retained input commitment, refits the operating
points and recounts the decisions without loading the model or reading
article prose.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.writing_neighbor_interaction_transfer review \
  docs/evaluation-panels/writing-neighbor-interaction-freeze-2026-11-14.json \
  docs/evaluation-panels/writing-neighbor-interaction-2026-11-14.json
```

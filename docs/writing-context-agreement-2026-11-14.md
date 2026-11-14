# Agreement between three views of a rewritten sentence

The combined rule located **no additional rewrites** across the two journal
panels and flagged one additional ordinary work in each. It remains offline.
The app now preserves completed readings when its optional longer-passage
check runs only partly. It lists the unchecked sentence positions in the
evidence view, PDF and JSON.

## Comparison

Each eligible sentence gets three readings from the existing commercially
licensed checkpoint. These are the position score relative to its neighbors,
the response to the isolated sentence, and the smaller of the two response
drops after removing the sentence from either neighboring pair. The median
and median absolute deviation of ordinary work maxima put the readings on
comparable scales. The median of the three at each sentence requires two
readings to clear the same normalized point. The readings share an encoder
and can share errors. Recent segmentation work also examines misplaced
boundaries caused by changes in local detector responses.
[LA-CPD](https://arxiv.org/abs/2609.33787) · [Checkpoint terms](writing-checkpoint-agreement-2026-11-11.md#frozen-inputs-and-comparisons)

Both historical versions of 14 F1000Research works set the operating point
before applying the rule to the other two panels. This retrospective
comparison uses previously measured journal cases. Exact text hashes, context
hashes and sentence boundaries agree across the two measurement receipts.
The source ledgers document commercial permission and the dated ordinary
controls. [Position measurement](writing-long-position-transfer-2026-11-12.md) · [Omission measurement](writing-omission-attribution-2026-11-13.md)

| Outcome | F1000Research | eLife |
| --- | ---: | ---: |
| Edited files | 18 | 21 |
| Exact targets measured | 18 | 21 |
| Correct targets ranked first before cutoff | 9 | 3 |
| Useful locations from the incumbent checks | 4 | 0 |
| Useful locations with the new rule added | 4 | 0 |
| New rule flags at the wrong location in edited files | 4 | 0 |
| Ordinary works flagged by incumbents | 2/9 | 2/12 |
| Ordinary works flagged by the union | 3/9 | 3/12 |

The combined rule selected one already known F1000Research location. It
flagged the same unchanged passage in both historical versions and both
generator rewrites of work 12989. An ordinary eLife excerpt also exceeded
the point. The rule failed the documented comparison criteria.

## Replay

Replay the [saved outcomes](evaluation-panels/writing-context-agreement-2026-11-14.json)
with standard-library Python 3.12 and four pinned public receipts.

```sh
PYTHONPATH=backend python3.12 -B -S -m scripts.review_writing_context_agreement review \
  docs/evaluation-panels/writing-context-agreement-freeze-2026-11-14.json
```

Use `calibrate` with a new output path to reproduce the
[frozen operating point](evaluation-panels/writing-context-agreement-freeze-2026-11-14.json).

## Partial results in the app

The optional longer-passage check keeps a center after all three of its
inputs finish. It retries a rejected batch one center at a time. If the
check runs out of time, completed centers still count toward recorded passage
coverage. The check selects a sentence only after it can compare all eligible
centers. The source verifier checks each retained input hash and the full
list of eligible positions against the original. The check carries zero
weight in the overall authorship estimate.

An isolated local checkout used simulated payment and the installed classifier
on an authored three-paragraph fixture. An injected rejection at the third
center left two completed centers. The evidence API and JSON export matched,
the PDF showed the same unchecked character range `[889, 1013)`, and the
signed package verified against the original. The fixture queued no analysis
before payment. One separate writing check did not run within the local
memory budget.

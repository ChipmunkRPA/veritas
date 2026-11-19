# Can classifier attribution locate a rewritten sentence?

Neither of two positional checks recovered a known rewrite in the held
panel. The exact output decomposition ranked the edited sentence first
zero times across 15 assessable rewrites. It also flagged two additional
ordinary work groups. The application keeps its existing passage check
and confidence scoring. [Anonymous numeric receipt](evaluation-panels/writing-positional-attribution-2026-11-19.json.gz)

## Matched sources

Eight ordinary work groups supplied 16 development files. A separate held
set contained 49 files from 16 works. Each work supplied an unchanged
excerpt and an ordinary comparison. The Federal Register comparisons used
verbatim replacement sentences from elsewhere in the same source. The
Turing Way comparisons used dated revisions from the official history.
Two eLife works supplied four additional short earlier and later excerpts.
Their results are counted separately. [Federal Register selection](writing-sentence-omission-2026-11-19.md) · [The Turing Way archive](writing-neighbor-factorial-2026-11-19.md) · [eLife license terms](https://elifesciences.org/terms)

The held set contained 17 exact single-sentence rewrites from two local
generators. Seven preserved the checked claims and ten changed a claim
or qualifier. Every other source character stayed the same within each
constructed rewrite pair. Factual labels and exact half-open positions
were recorded earlier and kept fixed. These excerpts appeared in other
local studies. They were newly held for the two methods here and shared
no normalized 12-word fragments across work groups in this test.
Checkpoint pretraining exposure and the historical authors' writing
tools are unknown. [Earlier factual and source checks](writing-neighbor-factorial-2026-11-19.md)

## Methods and results

The first method integrated gradients of the classifier's log odds over
eight points between the original embeddings and fixed reference embeddings.
It kept token positions, attention and special tokens fixed. A numerical
check required the sum of contributions to recover the change in log odds.
The second method measured the contextual token positions that contribute
to the classifier's existing mean-pooled output. The classifier bias and
the entropy of its nonhuman class mix account for the remaining log odds.
The largest reconstruction error on an assessed file was under 0.000005.
Both methods used the same local checkpoint and withheld excerpts longer
than its complete 512-token input. [Integrated gradients paper](https://proceedings.mlr.press/v70/sundararajan17a.html) · [Checkpoint terms and revision](https://huggingface.co/wasitaigeneratedcom/ai-text-detector-small/tree/f1795c86806e6838d4afa33d0b1427f8430c9615)

Each candidate ranked the unique highest positive sentence after adjusting
for the square root of its word count. The gap to the next nonnegative
sentence set the internal selection value. Whole ordinary development
groups fixed each operating point before held scoring, with room for at
most one extra development alert group above the incumbent. The incumbent
used its existing isolated and neighboring-sentence readings. An exact
location needed to cover at least half the edited non-whitespace characters
and reach at least 0.30 intersection over union. [Mixed-writing localization research](https://arxiv.org/abs/2506.02959) · [Progressive editing benchmark](https://arxiv.org/abs/2606.06481)

| Primary held check | Exact rewrite locations / 17 | Ordinary work groups flagged / 16 | Wrong locations on rewrites |
| --- | ---: | ---: | ---: |
| Existing optional passage check | 0 | 1 | 1 |
| Integrated gradients | 0 | 2 | 0 |
| Exact output decomposition | 0 | 3 | 2 |

Integrated gradients failed the numerical check on all 14 structurally
eligible development files. In the primary held set it assessed three
ordinary files, declined 40 files after the numerical check and skipped
six over the token limit. All 17 rewrites were unassessed in this method.
The exact decomposition assessed 43 primary files and skipped six over
the same limit. On the 17 rewrites it ranked a neighboring sentence first
in two files and a farther sentence first in three. The other 12 produced
no ranked sentence, including two above the token limit. Its two shown
locations were both on changed-claim examples. Neither method flagged the
two short eLife work groups. All four eLife files passed exact decomposition
and failed the numerical gradient check.

As a separate diagnostic, comparing matched earlier and later readings gave
the known edited sentence the largest positive change in six of 15 assessable
synthetic pairs and four of 13 assessable ordinary edited pairs. This
diagnostic uses information unavailable for a single upload. Both the
one-file results and the paired control overlap limit its use for AI
authorship claims.

## Recount

The receipt contains anonymous work and file identifiers, sentence bounds,
non-whitespace positions, scalar readings, and local runtime and checkpoint
pins. The standard-library recount regenerates the development operating
points, exact held matches, ordinary alert groups and paired diagnostic.
The local runs used one CPU thread each. Development and held scoring took
about 14 minutes in total for integrated gradients and about three minutes
for exact decomposition. These timings describe the local run with its
shared workload. The public receipt leaves out all source and generated
prose.

```sh
python3 -B -I -S backend/scripts/review_writing_positional_attribution.py \
  docs/evaluation-panels/writing-positional-attribution-2026-11-19.json.gz
```

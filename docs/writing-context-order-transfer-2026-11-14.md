# Does swapping neighboring sentences improve rewrite localization?

We measured the same target sentence with its original neighbors, with
the neighbors swapped, and on its own. The classifier used the same
weights throughout. Each view applies its existing head to the target
sentence's token vectors only. Swapping neighbors changes the model's
context while preserving the sentence itself. Agreement across views
measures sensitivity to that change. The views share weights and cannot
provide independent votes on authorship.
[Checkpoint and terms](writing-context-pool-transfer-2026-11-14.md)
· [Editing-versus-generation research](https://arxiv.org/abs/2510.03154)
· [Peer-review polishing evaluation](https://arxiv.org/abs/2603.20450)

## Method and results

The frozen parent rosters supply 96 ordinary reference files and 72 separate
ordinary files for point selection. The two held panels contain 56 and 60
pairs of ordinary news or abstract excerpts and known one-sentence rewrites
from two generator families. Each source work belongs to a single panel.
All four panels had prior project exposure. The experiment reran 3,600 contexts with the existing
commercially licensed checkpoint on Apple GPU. Publication keeps source text
outside the receipts. [Pinned source and rights controls](writing-joint-tail-transfer-2026-11-14.md)

First rank each reading against one ordinary maximum per source work, genre
and length band. Then use the separate ordinary files to set points for
three empirical alert-rate targets. A rule can show only a unique highest
interior sentence. A useful location covers at least 50% of the marked edit
and reaches 0.30 intersection over union on non-whitespace codepoints.
Baseline counts use the existing research proxy: the classifier or the
writing-pattern check can produce a highlight. The exact worker reports
can differ from the proxy.

| Rule at the broader point | Held 56 pairs, ordinary flags / useful locations | Transfer 60 pairs, ordinary flags / useful locations |
| --- | ---: | ---: |
| Existing classifier and writing-pattern proxy | 52 / 29 | 55 / 21 |
| Existing raw agreement | 8 / 1 | 8 / 4 |
| Native context alone | 25 / 8 | 21 / 6 |
| Require original and swapped context agreement | 23 / 7 | 24 / 7 |
| Require both contexts and agreement in isolation | 28 / 12 | 23 / 7 |
| Require order agreement and contrast to the next center | 24 / 7 | 25 / 5 |

The three-view rule found three additional rewrite locations in each panel
but lost 20 previously useful locations in the 56-pair panel and 17 in the
60-pair transfer. Adding it alongside the original proxy would keep the
earlier locations and add two previously unflagged ordinary files to the
first panel. Its standalone flags on rewrites were also frequently misplaced,
with 19 in each panel. Every held file had at least one eligible reading.
The candidate remains offline. These are excerpt-level tests and cannot
estimate the reliability of a whole-document authorship claim.
[Measurements and outcomes](evaluation-panels/writing-context-order-2026-11-14/)

## Replay and independent evaluation

The standard-library replay rebuilds the ordinary-work references, the
pre-held cutoffs and the held comparisons. With the native rosters from the
parent procedure, it separately checks the exact context hashes and word
counts for all 3,600 inputs without model weights. Repeating the inference
uses the same checkpoint and `fit`, `point`, `freeze`, `held` and `transfer`
phases in that order. The first three measurement phases use the native
matched roster. Transfer uses the native earlier roster.

```sh
export PYTHONPATH=backend
bundle=docs/evaluation-panels/writing-context-order-2026-11-14
python3 -B -S -m scripts.writing_context_order_transfer review "$bundle" > "$run/outcomes.json"
cmp "$bundle/outcomes.json" "$run/outcomes.json"
for role in fit point held; do
  python3 -B -S -m scripts.writing_context_order_transfer audit-inputs \
    "$bundle/$role.json" --roster "$matched_roster"
done
python3 -B -S -m scripts.writing_context_order_transfer audit-inputs \
  "$bundle/transfer.json" --roster "$earlier_roster"
```

For independent experiments on new source files, the portable comparison
can now report transitions at a declared useful-location rule. This keeps
an unassessed source separate from an assessed source with no region and
reports when a method loses a useful region even if the mean IoU improves.
[Four-media location comparison](independent-testing.md#evidence-localization)

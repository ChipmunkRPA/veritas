# Can a writing check separate ordinary edits from AI rewrites?

The previous comparison added historical human revisions to the training
negatives. This experiment tests a different use for the same pairs.
Measure the change in the encoder's word states between an actual earlier
phrase and its replacement. Then remove the strongest directions of
ordinary editorial change from the word states before fitting the rewrite
check. Research on mixed writing motivates exact changed-word labels and
ordinary editing controls. This comparison reuses the pinned Apache-2.0
encoder and implements its own projection and training.
[Word-level editing research](https://arxiv.org/html/2609.32625v1)
· [Pinned encoder and terms](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

## Comparison

- **Existing experimental fit.** Keep the earlier AI-only word head.
- **Unrelated article control.** Pair each historical replacement with
  earlier wording from a different article using a fixed one-step rotation
  of the source order. Remove the 12 largest directions from those unit
  difference vectors.
- **Matched editorial projection.** Remove the 12 largest directions from
  true before-and-after word replacements using the same calculation.

For each projected fit, remove the learned components from both the word
state and its difference from the sentence mean. The three fits use the
same encoder, AI-changed words, ordinary words, head architecture and
training schedule. The comparison ranks each interior sentence by its
six strongest word responses and highlights the strongest 60% of its
words. Each head uses its own points from 36 ordinary calibration works
per genre. A useful result covers at least half the changed words and
has at least 0.30 word intersection over union. The submitted excerpt
alone supplies the inference input. [Earlier head and fit roster](writing-historical-editorial-negatives-2026-11-15.md)

## Sources

The retained evaluations include 112 news and report pairs, 84 separate
news and report pairs, and 120 pairs of historical Wikipedia revisions.
The Wikipedia articles used to fit the projection are separate from
those 120 works. News and reports include two AI generators. The additional
test uses 12 previously unreported Gemini essay work pairs, selected before
fitting, with up to six per interior sentence position. A further 36 essay
originals set the essay points. The source intake checks earlier project
work hashes and normalized 12-word overlap. Attempts at a larger essay
panel with separate GPT-5.4 Nano or GPT-5.4 did not supply unused works
with the required exact geometry. The resulting fresh test covers one
generator. Model pretraining overlap remains unknown.
[Publisher benchmark methodology](https://arxiv.org/abs/2606.06481)
· [Pinned essay shard and dataset terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench)

## Results

All 180 fit articles supplied eligible word replacements. The retained
AI fit used 297 work pairs, 1,690 changed words, and 15,624 other words.
The 12 matched editorial directions accounted for 27.35% of the squared
unit differences. The unrelated article directions accounted for 31.72%.

Each cell shows useful locations followed by ordinary flags at the wider
point. The Wikipedia row shows earlier and revised ordinary excerpts flagged.

| Held panel | AI-only fit | Unrelated article control | Matched editorial projection |
| --- | --- | --- | --- |
| Retained 112 news and report pairs | 1/112 · 31/112 | 1/112 · 32/112 | 1/112 · 33/112 |
| Separate 84 news and report pairs | 2/84 · 23/84 | 4/84 · 22/84 | 2/84 · 29/84 |
| 12 new essay pairs | 0/12 · 2/12 | 1/12 · 4/12 | 0/12 · 2/12 |
| 120 dated Wikipedia pairs | 5/120 · 4/120 | 4/120 · 2/120 | 6/120 · 4/120 |

All three arms assessed 106 of 112 work pairs in the first panel and
79 of 84 in the second. All 12 essay pairs had complete measurements.
At the tighter point, the matched projection reduced ordinary flags
from 19 to 15 on the 112-pair panel and lost its one useful location.
On the 84-pair panel it kept one useful location and cut ordinary flags
from 14 to nine. At the wider point the matched projection gained no
useful locations on either retained panel or on the new essays. The
unrelated control gained two useful locations on the 84-pair panel.
One of its four locations also flagged the corresponding ordinary
sentence. Its one useful essay location repeated on the same sentence
in the earlier ordinary essay. Both projections stay in evaluation.
[Numeric receipt](evaluation-panels/writing-editorial-direction-2026-11-15.json.gz)

## Replay

The receipt stores work hashes, original word coordinates, calculated
scores, selected words, and the results. The benchmark articles,
embeddings, and fitted heads stay local. The numeric replay checks source
separation within the project, recalculates the cutoffs, and measures
rewrite overlap alongside alerts on matched ordinary text.
Python 3.12 and 3.14 returned identical numeric reviews. A second native
run reproduced the private roster and published receipt byte for byte.

```sh
export PYTHONPATH=backend
module=scripts.writing_editorial_direction_transfer
receipt=docs/evaluation-panels/writing-editorial-direction-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

Recreate the private parent plan with the
[earlier historical editorial study](writing-historical-editorial-negatives-2026-11-15.md#replay).
Use the pinned `essays_gemini-2.5-flash.csv` test shard from the publisher
revision `072c03c7051409346aaba05168d183cf6d863ee9`. The runner checks its
SHA-256 against the pinned benchmark file.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$parent_plan" "$gemini_essays" "$run/plan.json.gz" --reference "$receipt"
"$runner" -B -m "$module" run "$run/plan.json.gz" "$encoder" "$run/result.json.gz"
cmp "$receipt" "$run/result.json.gz"
```

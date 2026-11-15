# Narrow writing highlights using unchanged words during training

This trial tests whether an individual upload can be given a useful
highlight over edited words. It measures word overlap against a known
earlier version. The overlap measures where the rewrite changed the text.
The scores do not establish who wrote individual words.

## Training and held inputs

The six-layer encoder from the previous word experiment stays frozen.
Three new linear heads start from the earlier word head and train on the
same 882 eligible government-report rewrite pairs from Gemini 2.5 Flash
and GPT-5.4 Nano. All use the existing word labels and compare rewritten
words with the six strongest words in the earlier version of the
sentence. The second head also penalizes score changes on exactly
aligned unchanged words. The third adds a comparison with the strongest
unchanged words inside the edited sentence. Paired original text is
used during fitting and scoring the evaluation. Each held inference
reads its own excerpt alone. The code reuses the pinned Apache-2.0
base and the earlier locally fitted checkpoint.
[Earlier fit and source roles](writing-word-localization-2026-11-14.md) ·
[Exact-word alignment research](https://arxiv.org/abs/2609.32625) ·
[Word-level coauthorship research](https://arxiv.org/abs/2607.21458)

Four pinned OpAI-Bench publisher development shards supply 72 ordinary
calibration works and 84 previously unused matched original and rewrite
works, half news and half government reports. The held works contain
48 single-sentence and 36 adjacent two-sentence edits, balanced between
Gemini 2.5 Flash and GPT-5.4. The frozen selection checks 216 earlier
project receipts for exposed works and excludes shared word sequences
as defined by the intake. Model pretraining exposure to the source prose
is unknown. A separate retained panel has 21 pairs of dated genuine
journal revisions, which stayed outside all three fits. Complete word
scores were available on 78/84 originals, 78/84 rewrites and 32/42
journal excerpts. The remaining inputs count as unassessed.
[Publisher data and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse) ·
[Publisher paper](https://arxiv.org/abs/2606.06481)

The earlier word rule ranks a sentence by its six highest word scores
and highlights its highest-scoring 60% of words. The new search ranks a
contiguous 2, 3, 5, 8 or 12-word range in one interior sentence. The
fragment variant can also show two equally wide 2–4 word ranges with
a gap of 1–5 words, within one or two adjacent interior sentences.
Separate ordinary maxima for news and reports set the scale for every
width. Two research points use leave-one-work-out maxima on those ordinary
files. A useful location covers half the changed words and reaches 0.30
intersection over union. Research points apply only to this evaluation.

## Results

| Wider research point | Useful rewrite locations | Useful with clear matched original | Ordinary flags | Journal revision panel flags |
| --- | ---: | ---: | ---: | ---: |
| Earlier word head, broad highlight | 4/84 | 2/84 | 17/84 | 6/42 |
| Earlier word head, narrow highlight | 0/84 | 0/84 | 16/84 | 6/42 |
| Earlier word head, with fragments | 0/84 | 0/84 | 16/84 | 6/42 |
| Refit, narrow highlight | 1/84 | 0/84 | 13/84 | 6/42 |
| Unchanged-word penalty, narrow highlight | 1/84 | 0/84 | 14/84 | 6/42 |
| Unchanged-word penalty and local margin, narrow highlight | 1/84 | 0/84 | 16/84 | 6/42 |
| Unchanged-word penalty and local margin, with fragments | 2/84 | 1/84 | 16/84 | 6/42 |

At the tighter point the earlier broad highlight locates two edits and
the strongest new rule locates one. Both flag six ordinary originals
and two journal excerpts. None of the journal flags has useful overlap
with the genuine revision itself. Both runs of the four word heads on
all 282 excerpts produced identical scores and calibration points.

Given the known changed words after inference, 53/84 edits could fit
one narrow range and 39/84 could fit two fragments under the same
overlap rule. Either shape can cover 61/84. The new head slightly
improves the median within-edit-sentence rank of changed words from
0.677 to 0.686. Unchanged words drift more between versions, from
0.262 to 0.275 in units of each head's ordinary median absolute
deviation. The new heads remain in evaluation. [Word score receipt](evaluation-panels/writing-word-anchor-transfer-2026-11-15.json.gz)

## Existing app replay

Eight matched work pairs were selected by source hash across genres,
generators and edit sizes. On the Mac's GPU, both current writing checks
completed on all 16 inputs. The app showed passages for review on all
eight rewrites and eight ordinary originals. Three rewritten excerpts
had an exact edited-sentence display, and one had useful changed-word
overlap. All overall assessments were inconclusive. A passage for review
and a file-level assessment are separate report fields. The replay
verified the evidence API and JSON source offsets, and processed 16
simulated USD 9.99 payments with zero unpaid jobs. A separate CPU
run completed the passage classifier on all 16 files and skipped the
writing-pattern comparison for memory on all 16. The GPU result above
uses both checks. [App receipt](evaluation-panels/writing-word-anchor-app-auto-2026-11-15.json)

## Replay

The numeric receipts retain source hashes, exact character offsets,
experimental scores and changed-word indices. They contain no source
prose or weights. The sentence ledger binds the inputs of the app replay
to the same frozen works as the word experiment.

```sh
export PYTHONPATH=backend
module=scripts.writing_word_anchor_transfer
panel=docs/evaluation-panels/writing-word-anchor-transfer-2026-11-15.json.gz
sentence=docs/evaluation-panels/writing-word-anchor-sentence-2026-11-15.json.gz
app=docs/evaluation-panels/writing-word-anchor-app-auto-2026-11-15.json
python3 -B -S -m "$module" review "$panel"
python3 -B -S -m scripts.writing_adjacent_range_transfer review "$sentence"
python3 -B -S -m scripts.writing_adjacent_range_app "$sentence" "$app"
```

For native replay, restore the earlier word fit's source roster, tokenizer,
and checkpoint. Use the pinned ONNX 1.24.3 environment and classifier
from the adjacent-sentence study. Keep the local plan and fitted weights
outside the public package.
[Earlier word fit and roster](writing-word-localization-2026-11-14.md#independent-replay) ·
[Sentence classifier environment and assets](writing-adjacent-range-2026-11-15.md#independent-replay)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
runner=backend/.venv/bin/python
"$runner" -B -m scripts.writing_adjacent_range_transfer freeze \
  "$run/plan.json.gz" --reference "$panel"
"$ort_python" -B -m scripts.writing_adjacent_range_transfer capture \
  "$run/sentence.json.gz" "$run/plan.json.gz" "$sentence_assets"
"$runner" -B -m "$module" fit "$run/fit" "$roster" "$encoder" "$checkpoint"
"$runner" -B -m "$module" capture "$run/result" "$run/fit" "$run/plan.json.gz" \
  "$run/sentence.json.gz" "$roster" "$encoder" "$checkpoint"
"$runner" -B -m scripts.writing_adjacent_range_app "$sentence" "$run/app" \
  --plan "$run/plan.json.gz" --device auto
```

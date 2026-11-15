# Locating wording by testing what changes a sentence score

An independent writing classifier selected the edited sentence more
often on one report panel, but its inherited word ranges rarely matched
the changed words. This follow-up asks whether the same classifier
reacts more strongly to removing the actually revised words than to
removing other words in the sentence. This is a measure of the classifier's
dependence on wording. A score change alone cannot establish authorship.
[Sentence and word baseline](writing-independent-range-transfer-2026-11-15.md)

## Fixed comparison

Use the same pinned, MIT-licensed 68M-parameter classifier and tokenizer.
The prior absolute-rank sentence selector keeps its original flags at
both ordinary-calibrated points. The new check can change which words
are highlighted inside a flagged sentence. It cannot create a new
sentence flag. For diagnosis only, also test the known edited sentence
in every labeled rewrite, including those that the selector missed.

Compare contiguous groups of 3, 6, 12 and 18 source words. Keep four
or more original words in the sentence. Sample starts every half-group
or three words, whichever is greater, and include the last possible
start. One reading removes the selected text and rejoins the remaining
parts. The other keeps the original token positions and substitutes
mask tokens for all token pieces inside the selected span. It declines
spans that share a token with neighboring non-whitespace characters.
Neither version modifies the source document.

Each score is the drop in the AI-minus-human logit divided by the
square root of the word count. The joint reading requires both drops
to be positive and uses the smaller score to select its highest span.
Equal highest scores withhold the selection. Each batch includes a
copy of the original sentence as its baseline, and the receipt records
its largest departure from the original unbatched reading.

Perturbed text can lie outside the classifier's training distribution.
Deleting words may break grammar, and mask tokens may elicit their own
response. For this reason the evaluation separately measures known word
overlap and ordinary inputs that would acquire word highlights. It also
checks how many documented rewrites the candidate grid could locate if
given the best possible span. [Feature removal and distribution shift](https://proceedings.neurips.cc/paper_files/paper/2021/hash/1def1713ebf17722cbe300cfc1c88558-Abstract.html)
· [Word-level evaluation of AI edits](https://arxiv.org/abs/2609.32625)

The held set uses the previous 628 excerpts, including 309 generated
rewrites and 319 ordinary original or revised files. The three native
sentence positions, edited word labels, source exclusions, and selection
points come from the earlier measurements. A correct word highlight
must cover at least half of the changed words and have at least 0.30
word intersection over union. The original source, modified inputs and
model weights remain local.

## Local results

The runner measured 482 nominated sentences from 419 excerpts. It
scored 10,223 deletions and an equal number of mask substitutions.
The remaining files had neither a wider sentence flag nor a labeled
rewrite requiring the separate known-sentence diagnostic.

| Wider sentence rule | Existing word range | Deletion | Masking | Require both |
| --- | ---: | ---: | ---: | ---: |
| Reports | 2/93 | 2/93 | 2/93 | 2/93 |
| Separate reports | 1/48 | 1/48 | 0/48 | 0/48 |
| News | 1/56 | 2/56 | 1/56 | 2/56 |
| Retained abstracts | 1/60 | 1/60 | 1/60 | 1/60 |
| Additional abstracts | 0/28 | 0/28 | 0/28 | 0/28 |
| eLife body excerpts | 0/24 | 0/24 | 0/24 | 0/24 |
| All rewrites | 5/309 | 6/309 | 4/309 | 5/309 |

Deletion found six different useful spans and lost all five of the
existing useful spans. Its two useful news ranges also highlighted the
same sentence in the matched ordinary files. Every arm supplied a word
highlight to all 110 ordinary files flagged by the fixed selector at the
wider point. Three of those ordinary files had no previous word range.

With the actual edited sentence supplied, at least one tested range
could meet the overlap rule in 231 of 309 rewrites. The previous word
range located 43, deletion located 31, masking located 26, and requiring
both located 25. The receipt reports the expected overlap when choosing
a random eligible starting position of the same selected width. This
diagnostic separates range coverage from ranking. The word scores did
not reliably distinguish the changed wording, so the check remains in
evaluation.

## Replay

The public data includes only the source hashes, original word bounds,
and numeric responses to each mask or deletion. The portable reviewer
reconstructs the eligible sentence nominations, validates each word
grid and reproduces the held word-overlap counts. A deterministic audit
scored 14 sentences individually, two from each panel. All 42 span
selections agreed with the batched calculations. The largest observed
difference in a logit drop was 0.0000126. The audit receipt binds the
batched source records and preserves the individual selections.
[Word-range receipt](evaluation-panels/writing-word-perturbation-2026-11-15.json.gz)
· [Batch audit](evaluation-panels/writing-word-perturbation-batch-2026-11-15.json)

```sh
export PYTHONPATH=backend
module=scripts.writing_word_perturbation_transfer
python3 -B -S -m "$module" review \
  docs/evaluation-panels/writing-word-perturbation-2026-11-15.json.gz \
  --batch-audit docs/evaluation-panels/writing-word-perturbation-batch-2026-11-15.json
```

Use the same [pinned model and native roster](writing-independent-range-transfer-2026-11-15.md#independent-replay)
and the same native arguments with `scripts.writing_word_perturbation_transfer`
and the `capture` or `audit-batch` subcommand. Use a fresh output path.
The comparison needs ONNX Runtime 1.24.3 for native inference. The
portable review reproduced identical output on Python 3.12 and 3.14.

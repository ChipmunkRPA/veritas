# Checking a writing range against nearby sentences

The earlier single-upload range checks often ranked ordinary sentences
above the actual rewrite. This comparison tests whether a high-scoring
range stands out from its two other eligible interior sentences and
whether a second check points to the same words. Each check sees one
upload. The original paired drafts establish evaluation locations. The
two checks share an encoder and therefore have related errors.
[Earlier range training](writing-paired-ranges-2026-11-15.md) ·
[Research on local score variation](https://arxiv.org/html/2609.33787v1)

## Comparison

- The baseline chooses the top exact range from the paired-training
  head at the previously measured absolute response point.
- The contrast rule subtracts the strongest response in another
  interior sentence. A second version requires both a strong absolute
  response and a strong contrast.
- The sentence-agreement rule takes the weaker of the two heads'
  absolute and contrast ranks in the same sentence. The word-agreement
  rule also requires three shared words and at least half of each
  proposal to overlap, then keeps only the shared words.

Ranks come from ordinary works at the same sentence position.
Each calibration work is ranked against the others in its genre.
Each rule uses its strongest unique interval and its own calibration
point, computed at the existing tighter and wider ordinary alert
budgets. The abstract and news results use their respective calibration
works. The article-body results replay both report and abstract ratings.
These are research comparisons without calibrated authorship probabilities.

A useful range covers at least half of the known changed words with
word intersection over union of at least 0.30. A matched alert means the
rule also flags the changed sentence's position in an ordinary version
of the same work. This identifies a failure to distinguish an edited
passage from its corresponding original even when the measured word
intervals differ.

## Results

Each entry gives useful rewrite locations followed by ordinary files
flagged. Dated journal pairs are a separate ordinary-only comparison.

| Panel and point | Paired range baseline | Contrast alone | Shared words |
| --- | ---: | ---: | ---: |
| Retained abstracts, wider | 1/60 · 25/60 | 2/60 · 25/60 | 2/60 · 8/60 |
| Additional abstracts, tighter | 1/28 · 1/28 | 2/28 · 4/28 | 1/28 · 7/28 |
| Additional abstracts, wider | 2/28 · 8/28 | 2/28 · 6/28 | 1/28 · 9/28 |
| Retained news, wider | 0/56 · 8/56 | 1/56 · 17/56 | 1/56 · 12/56 |
| Separate reports, wider | 0/48 · 8/48 | 0/48 · 11/48 | 0/48 · 13/48 |
| eLife body excerpts, wider abstract point | 0/24 · 19/30 | 0/24 · 12/30 | 0/24 · 7/30 |

The shared-word rule kept one earlier abstract location and added one
on the retained set at the wider point. Neither location appeared in
its matched original. The additional abstract holdout froze 28 works
before scoring, with 24 rewrites from one publisher-labelled generator
family and four from a second. The selection excluded known work hashes
in 206 repository receipts and normalized 12-word overlaps with known
originals in the two source shards. It also excluded overlaps between
admitted works. Prior checkpoint exposure is unknown.
[Benchmark methodology](https://arxiv.org/abs/2606.06481) ·
[Dataset and reuse terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

The shared-word rule located one additional-holdout edit at both
points. Its matched original also flagged the same sentence. The
contrast-only rule added one useful location at the tighter point with
three additional ordinary flags. At the wider point it traded one
correct location for another and flagged two fewer ordinary files.
It located neither of the four additional-holdout rewrites from the
second generator. The wider shared-word gains on the retained
abstracts did not transfer.

The eLife panel newly measures five-sentence excerpts of 15 retained
source works. Its 30 ordinary files contain 15 originals and 15
same-work substitutions with four unchanged neighbors. Four of the
54 files fell outside the supported word or tokenizer geometry. The prior direct
head located one of the 24 rewrites when using the wider abstract
point and flagged seven ordinary files. The new rules located none.
The separate two dated eLife revision pairs produced zero shared-word
flags and two baseline flags at the wider report point. The combinations
remain in evaluation.
[eLife sources, licenses and earlier cases](writing-elife-exact-sentence-confirmation-2026-11-12.md) ·
[Dated natural revisions](writing-edit-occurrence-transfer-2026-11-15.md)

## Replay

The three receipts contain source hashes, exact Unicode word offsets,
scores and outcomes. The portable comparison produced identical
outcomes on Python 3.12 and 3.14. Two native passes recreated the same
eLife and abstract readings byte for byte. The receipts contain neither
the source prose nor model weights.
[Combined comparison](evaluation-panels/writing-range-context-2026-11-15.json.gz) ·
[eLife readings](evaluation-panels/writing-range-context-elife-readings-2026-11-15.json.gz) ·
[Additional abstract readings](evaluation-panels/writing-range-context-abstract-readings-2026-11-15.json.gz)

```sh
export PYTHONPATH=backend
module=scripts.writing_range_context_transfer
panels=docs/evaluation-panels
python3 -B -S -m "$module" review "$panels/writing-range-context-2026-11-15.json.gz"
```

For native replay, use the earlier eLife `plan.json`, the two abstract
test shards and the fitted encoder and direct and paired range heads
from the linked studies. The Gemini shard is pinned to the publisher's
`072c03c7051409346aaba05168d183cf6d863ee9` dataset revision.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
runner=backend/.venv/bin/python
"$runner" -B -m "$module" capture "$elife_plan" "$encoder" "$encoder_weights" \
  "$direct_head" "$paired_head" "$run/elife.json.gz"
"$runner" -B -m "$module" capture-abstracts "$gemini_shard" "$gpt_shard" "$encoder" "$encoder_weights" \
  "$direct_head" "$paired_head" "$run/abstracts.json.gz" \
  --reference "$panels/writing-range-context-abstract-readings-2026-11-15.json.gz"
python3 -B -S -m "$module" review "$panels/writing-range-context-2026-11-15.json.gz" \
  --elife "$run/elife.json.gz" --abstracts "$run/abstracts.json.gz"
```

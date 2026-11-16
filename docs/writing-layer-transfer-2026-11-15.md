# Test intermediate writing-model layers on local rewrites

The earlier comparison of writing scores found little benefit from
combining the existing checks. This test asks whether a different
layer of the same small encoder carries a stronger signal about which
words changed. SV-Detect probes representations from several layers
for document classification. MixDetect uses source-aligned supervision
for word localization. This experiment reuses the project's encoder,
training pairs and word selector. It imports no code or weights from
either paper. [SV-Detect](https://arxiv.org/html/2606.07313v1#S3.SS2)
· [MixDetect](https://arxiv.org/html/2609.32625v1#S3.SS1)

## Method

Freeze the six-layer encoder from the earlier word study. Fit three
linear heads of the same size on its final layer, its middle layer,
and the final layer minus the middle layer. All three start from the
same seed and see the same 882 original-and-rewrite report pairs.
All use the same losses for changed words, stable scores on unchanged
words, and separation from nearby unchanged words. Each representation
averages pieces into source words and concatenates the word vector with
its difference from the sentence mean. The two blocks are normalized
separately before reaching the new heads. The existing experimental
head retains its original preprocessing.
[Earlier fit](writing-word-localization-2026-11-14.md)
· [Alignment and loss](writing-word-anchor-transfer-2026-11-15.md)
· [Pinned encoder and terms](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf)

Each head ranks interior sentences by its six highest word scores and
proposes the strongest 60% of words in one sentence. The same 36
ordinary calibration works per genre set two independent points for
each head before held inference. A useful location covers at least half
of the labeled changed words and reaches at least 0.30 intersection over
union. A separate diagnostic supplies the known edited sentence positions
after inference and applies the same 60% word selection inside them
without a score requirement. It measures word-ranking overlap with
correct sentence selection supplied externally.

## Held comparison

The held panel contains 112 matched English pairs, with 28 pairs in each
combination of news or reports and full GPT‑5.4 or Gemini 2.5 Flash.
Each is a native six-sentence excerpt with unchanged first and last
sentences. At most four pairs per genre and generator occupy the same
one- or two-sentence edit slot. The intake verifies the publisher's
character and sentence labels and aligns original and revised words.
It excludes the recorded local fit groups, previously exposed work
hashes and shared normalized 12-word sequences with earlier reconstructed
excerpts, identifiable exposed originals, or another held work.
The GPT‑5.4 works come from the publisher's training partition, while
the Gemini works come from the test partition. Both are held out from
this project's fit and prior measured works. Pretraining exposure is
unknown. [Dataset documentation and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

All counts below use 112 as the denominator. Six originals and six
rewrites lacked complete usable word coverage for all four heads.
The two operating points are internal evaluation settings.

| Readout | Useful locations / ordinary flags, tighter | Useful locations / ordinary flags, wider | Useful when supplied the edited sentence positions |
| --- | ---: | ---: | ---: |
| Existing experimental head | 2 / 14 | 2 / 23 | 42 |
| Refit on final layer | 1 / 16 | 1 / 24 | 35 |
| Refit on middle layer | 1 / 20 | 3 / 38 | 39 |
| Refit on final minus middle | 1 / 13 | 1 / 25 | 33 |

At the wider point, the middle-layer head gained two useful locations
and lost one. It added 24 ordinary flags and cleared nine. It flagged
the corresponding sentence in the original for one of its three useful
locations. All useful locations came from Gemini rewrites. No head
localized a GPT‑5.4 rewrite usefully at either point. When the correct
sentence positions were provided, the existing head still led. Even
then, it met the overlap rule on 42 of 112 rewrites, so word selection
within a correctly identified sentence is also a bottleneck under this
method. The three candidates remain in evaluation. The serving checks
and confidence calculation are unchanged.
[Numeric receipt](evaluation-panels/writing-layer-transfer-2026-11-15.json.gz)

## Independent replay

The public receipt includes source hashes, sentence and word offsets,
scalar scores, outcomes and the frozen source commitments. It contains
no source text, private uploads or fitted weights. A reviewer can
recalculate the calibration and all held decisions with standard Python.
Python 3.12 and 3.14 produced identical numeric results. The native
replay regenerated the same source plan, fitted heads and held receipt
byte for byte on the Mac CPU.

```sh
export PYTHONPATH=backend
module=scripts.writing_layer_transfer
receipt=docs/evaluation-panels/writing-layer-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
```

For native replay, restore the three earlier private plans from the
[prior source replay](writing-ratio-fusion-transfer-2026-11-15.md#replay)
and the [earlier word fit](writing-word-localization-2026-11-14.md#independent-replay).
The shard directory needs the two `default/train/*_gpt-5.4.csv` and two
`default/test/*_gemini-2.5-flash.csv` files from the pinned publisher
revision, stored under `train/` and `test/`. Preparation verifies each
file by length and digest. Fit and capture use Torch 2.13.0 and
Transformers 5.15.1.
[Pinned publisher tree](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9/default)

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" prepare "$shards" "$ordinary_plan" "$earlier_plan" \
  "$recent_plan" "$run/plan.json.gz" --reference "$receipt"
"$runner" -B -m "$module" fit "$fit_roster" "$encoder" "$word_checkpoint" "$run/fit"
"$runner" -B -m "$module" capture "$run/plan.json.gz" "$run/fit" "$encoder" \
  "$word_checkpoint" "$run/held"
cmp "$receipt" "$run/held/receipt.json.gz"
```

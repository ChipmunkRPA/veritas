# Consecutive word ranges for short AI rewrites

The earlier word proposal highlighted the strongest 60% of words in
one sentence. This trial checks whether accumulating adjacent word
responses can find a short rewrite more precisely. It adapts the idea
of charging for each label transition from a published segmentation
method. The comparison uses the project’s existing Apache-2.0 encoder
and fixed word heads. It uses no external implementation or new weights.
[Sequence segmentation research](https://arxiv.org/html/2607.27183v1#S4.SS3.SSS3)
· [Prior word heads and base encoder](writing-word-anchor-transfer-2026-11-15.md)

## Comparison

Compare the earlier broad highlight with one consecutive run, up to two
runs, and the two-run rule after subtracting positive excerpt-wide median
elevation. Each run uses 2–18 words in one interior sentence. A second
run can follow after 1–6 words in the same or next interior sentence.
The score sums standardized word responses clipped to ±3, charges
0.8 units per selected word and 1.5 per run, and takes the strongest path.
Exact ties select the shorter highlight and then the earlier source
coordinates. The same scoring and source rules apply to both fixed
word heads.

The 72 earlier original news and report excerpts calibrate the word
scores and whole-excerpt maxima separately by genre. The original and
refitted heads had usable scores for 35 news and 36 report inputs each.
The tighter and wider research points use leave-one-work-out ordinary
maxima. The source roster and points were sealed before native test
inference. Each inference reads a single excerpt. Afterwards, the
earlier version supplies the exact word-change labels. A useful
highlight covers at least half the changed words with word intersection
over union of at least 0.30.
[Prior calibration and source alignment](writing-word-anchor-transfer-2026-11-15.md)

## Sources and coverage

The four pinned files from the official OpAI-Bench test split supplied
84 separate work pairs. Each pair has one earlier ordinary excerpt and
one rewrite of one or two interior sentences with exact unchanged
neighbors. There are 35 news and 35 report pairs rewritten by Gemini
2.5 Flash and 12 news and two report pairs rewritten by GPT-5.4. The
geometry and overlap exclusions left only two eligible GPT-5.4 report
pairs. Five pairs had unavailable complete word readings and remain
in the reported denominators. The roster excludes known work hashes
from 219 earlier receipts and checks exact 12-word overlap with the
reconstructed prior development excerpt roster, identifiable known
works in the four test files and other selected originals. Training
exposure in the base encoder is unknown.
[Pinned publisher release](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench/tree/072c03c7051409346aaba05168d183cf6d863ee9/default/test)
· [Publisher terms and provenance](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

For comparison, the retained development bank contributes another 84
rewrite pairs and 42 dated journal excerpts across 21 works. Six
development pairs and ten journal excerpts had no complete readings.
The journal comparison uses the report point as an out-of-genre check.
Documented human editorial change is a useful control because polishing
can shift AI detector responses while preserving authorship.
[Editorial study](https://arxiv.org/html/2608.26710v1)
· [Dated source roles](writing-word-anchor-transfer-2026-11-15.md#training-and-held-inputs)

## Results

Counts show useful rewrite locations and genuine files flagged. The table
uses the original word head for all four approaches. Both heads and the
journal comparison are in the receipt.

| Selection | Fresh tight | Fresh wide | Retained tight | Retained wide |
| --- | --- | --- | --- | --- |
| Earlier broad | 1/84 · 6/84 | 3/84 · 15/84 | 2/84 · 6/84 | 4/84 · 17/84 |
| One consecutive run | 1/84 · 2/84 | 1/84 · 10/84 | 0/84 · 5/84 | 2/84 · 15/84 |
| Up to two runs | 4/84 · 7/84 | 4/84 · 11/84 | 2/84 · 10/84 | 2/84 · 17/84 |
| Up to two runs with excerpt median adjustment | 0/84 · 3/84 | 3/84 · 9/84 | 1/84 · 8/84 | 2/84 · 14/84 |

At the fresh tighter point, the two-run rule kept the earlier useful
location and added three Gemini rewrites. One added location repeated in
the same sentence of the matched ordinary excerpt. Five genuine flags
were new and four old ones cleared. At the wider point it gained three
locations and lost two, with ordinary flags down by four net. At the
retained tighter point it gained two and lost two, with four net
additional ordinary flags. The journal flags moved from 2/42 to 4/42
at that point, with ten unassessed under both rules. The refitted head's
tighter two-run outcomes were the same in aggregate on both rewrite banks.
[Complete paired results](evaluation-panels/writing-structured-range-transfer-2026-11-15.json.gz)

Keep both proposals available for further comparison. The fresh gains
support a separate test of how the broad and two-run checks complement
each other. The weak overall localization and limited GPT-5.4 coverage
leave the decoder outside serving and overall confidence.

## Independent replay

The receipt includes numeric word readings, source hashes, exact
character offsets, word-change labels, frozen calibration and paired
outcomes. It contains no passages or fitted weight files. Standard
Python reconstructs the points and both sets of decisions. Python 3.12
and 3.14 produced identical reviews. Two native runs reproduced all
168 score records and the receipt byte for byte. The reviewer rejected
a changed point, a shifted word offset, an exposed work and a forged
outcome.

```sh
export PYTHONPATH=backend
module=scripts.writing_structured_range_transfer
panel=docs/evaluation-panels/writing-structured-range-transfer-2026-11-15.json.gz
python3 -B -S -m "$module" review "$panel"
```

For native replay, download the four `SHARDS` from the pinned publisher
release to one directory. Rebuild the exact prior development plan with
the [adjacent-sentence procedure](writing-adjacent-range-2026-11-15.md#independent-replay),
using the earlier word-anchor receipt as the freeze reference. Use the
same encoder, trained checkpoint and refitted head from the
[word-anchor procedure](writing-word-anchor-transfer-2026-11-15.md#replay).
The source replay hashes each official shard and reconstructs the
entire private plan from the downloaded source files.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
runner=backend/.venv/bin/python
"$runner" -B -m "$module" freeze "$test_shards" "$prior_plan" "$run/plan.json.gz" --reference "$panel"
"$runner" -B -m "$module" verify-source "$test_shards" "$prior_plan" "$run/plan.json.gz"
"$runner" -B -m "$module" capture "$run/plan.json.gz" "$encoder" \
  "$word_checkpoint" "$refitted_heads" "$run/receipt.json.gz"
cmp "$panel" "$run/receipt.json.gz"
```

To reproduce the historical exposure snapshot after further work, keep
the receipts listed by name and hash in `freeze.exposures` at their
original bytes. In addition to those receipts, the verifier uses the
earlier word-anchor receipt and the source-shard hashes pinned in the
script.

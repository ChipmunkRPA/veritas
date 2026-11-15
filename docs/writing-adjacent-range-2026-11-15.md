# Can two adjacent sentences support a more precise writing highlight?

This comparison tests short, mixed English excerpts in one upload. A
whole-document classifier can miss a transition between ordinary writing
and an edited passage. A whole-sentence highlight can also include mostly
unchanged wording even when it selects the right sentence. The experiment
measures both problems separately. It uses the same frozen MIT-licensed
small classifier as the earlier isolated-sentence test. It adapts the
idea of testing contiguous score sequences without importing research code
or training new weights.
[Local change-point research](https://arxiv.org/abs/2609.33787) ·
[Word-level editing research](https://arxiv.org/abs/2609.32625) ·
[Pinned classifier and terms](writing-independent-range-transfer-2026-11-15.md)

## Frozen inputs

The retained publisher test shards had too few unused works with the
required layout. This run therefore uses four byte-pinned publisher
development shards for news and government reports, two per genre.
It selects separate calibration and held works before inference.
The intake checks 213 earlier numeric receipts for known work hashes and
excludes shared normalized 12-word sequences against known originals
within the shards and all admitted works. Exact publisher sentence
labels must agree with the inline character tags. The local app's
sentence reader must reconstruct the same six sentences in one
paragraph. Every sentence outside the target must match its original.
Each held work contributes one original and one exact rewrite.

The resulting set has 72 ordinary calibration works and 84 different
held works, half news and half reports. There are 48 single-sentence
rewrites and 36 adjacent two-sentence rewrites across two generators.
The targets include every interior position. Of 84 held edits, 59
come from the publisher's polish stage, 21 from paraphrasing, and four
from broader style edits. Checkpoint exposure to the original prose
is unknown. The evaluation package retains no prose or token IDs.
[OpAI-Bench method](https://arxiv.org/abs/2606.06481) ·
[Publisher data and Apache-2.0 terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench#licensing-and-reuse)

## Comparison

The isolated baseline chooses among four interior sentences. Each
experimental arm can also choose any adjacent two-sentence target.
One ranks a pair by the weaker of its two isolated readings. The other
reads both sentences together. Each measure uses its own median and
90th-percentile scale for each position and width on ordinary files.
The two whole-file operating points come from leave-one-work-out
ordinary maxima. Every arm withholds tied picks and returns at most
one range. These are research comparisons independent of public app
confidence.

Exact location means selecting all and only the publisher's edited
sentences. Useful changed-word overlap requires half the changed words
and at least 0.30 intersection over union. The changed words come from
aligning each labeled sentence with its ordinary counterpart.

| Wider operating point | Exact single sentences | Exact adjacent pairs | Useful changed-word locations | Ordinary excerpts flagged |
| --- | ---: | ---: | ---: | ---: |
| Isolated sentences | 4/48 | 0/36 | 2/84 | 18/84 |
| Also allow pairs, by their weaker member | 2/48 | 3/36 | 2/84 | 18/84 |
| Also allow pairs, read together | 4/48 | 1/36 | 2/84 | 16/84 |

All 84 held edits and 84 matched originals received complete readings.
Neither pair rule located both changed sentences in a government report.
Two of the weaker-member rule's three correct adjacent pairs also
flagged the same pair in the ordinary original. Reading both sentences
together selected one adjacent edit without that collision and cleared
two ordinary flags overall. At the tighter point that arm preserved
two of the baseline's four correct single-sentence locations and added
one correct pair. It still located only two useful changed-word ranges.
Both approaches remain in evaluation.

### Separate the ranking error from the highlight width

Even with the true sentence supplied after inference, the entire sentence
met the useful word-overlap rule on only **3/48** single-sentence edits.
The full two-sentence region met it on **13/36** adjacent edits.
Removing the cutoff gave the weaker-member rule five correct top pair
picks on the 18 news works and zero on the 18 reports. The joint reading
ranked three news pairs first and zero report pairs first. A narrower
word proposal and a better passage score are separate improvements to
test. These diagnostics use the known answers after inference.

## Existing app comparison

A separate local CPU replay selected eight matched works by source
hash, one per genre, generator and edit width, independently of
scores. Sixteen simulated USD 9.99 checkouts completed, and no analysis
started before payment. The API and JSON contained identical evidence
with offsets verified against each submitted excerpt. The app showed
review passages on three of eight original excerpts and three of eight
rewrites. None provided useful changed-word overlap. All sixteen
overall verdicts were inconclusive. The passage classifier completed
on every input. The writing-pattern ratio exceeded the local CPU
model memory budget and was skipped on every input. These counts
describe the existing app with that coverage.
[CPU app receipt](evaluation-panels/writing-adjacent-range-app-2026-11-15.json)

An initial automatic-device attempt exposed a separate runtime error.
When the storage volume filled during a model load, saving its measured
memory footprint could interrupt an otherwise successful load. The app
now retains the new measurement for its in-process memory accounting
and keeps the previous disk snapshot readable. A regression test forces
the disk write to fail, checks that the higher observation is still
available, and verifies that a later successful write persists it.
The upload storage reserve still applies.

After freeing storage, all 16 inputs completed simulated checkouts with
automatic device selection on the Mac's GPU. Both writing checks ran on
all 16 inputs. The app showed a review passage on seven of eight
rewrites and all eight ordinary originals. Three rewrites included a
review location that covered exactly the words of the labeled sentence or pair.
None met the changed-word overlap rule. All 16 overall verdicts remained
inconclusive. A review passage is a location for inspection and is
counted separately from a file-level verdict.
[Automatic-device app receipt](evaluation-panels/writing-adjacent-range-app-auto-2026-11-15.json)

## Independent replay

The standard-library replay verifies source commitments and native
Unicode positions, rebuilds the operating points from ordinary works,
and recounts exact locations, ordinary flags and matched-original
collisions. The diagnostic evaluates highlight geometry using the
same receipt. The portable commands need no data download, model or
running app. Python 3.12 and 3.14 produced identical results. A
second source intake reproduced the source roster byte for byte.
A second native inference run reproduced all 1,680 readings.

```sh
export PYTHONPATH=backend
module=scripts.writing_adjacent_range_transfer
receipt=docs/evaluation-panels/writing-adjacent-range-2026-11-15.json.gz
python3 -B -S -m "$module" review "$receipt"
python3 -B -S -m "$module" review "$receipt" --diagnostic
```

For native replay, use the pinned classifier assets and ONNX
Runtime 1.24.3 from the [earlier study](writing-independent-range-transfer-2026-11-15.md#independent-replay).
The freeze verifies each full publisher shard against its expected
byte count and hash, then releases its temporary copy. It also accepts
local shard directories through `--news` and `--reports`. The
`--reference` option uses the same prior exposure inventory as this
experiment. Source text stays in the local plan.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
backend/.venv/bin/python -B -m "$module" freeze "$run/plan.json.gz" --reference "$receipt"
"$ort_python" -B -m "$module" capture "$run/receipt.json.gz" "$run/plan.json.gz" "$assets"
cmp "$run/receipt.json.gz" "$receipt"
backend/.venv/bin/python -B -m scripts.writing_adjacent_range_app \
  "$receipt" "$run/app" --plan "$run/plan.json.gz"
backend/.venv/bin/python -B -m scripts.writing_adjacent_range_app \
  "$receipt" "$run/app-auto" --plan "$run/plan.json.gz" --device auto
python3 -B -S -m scripts.writing_adjacent_range_app \
  "$receipt" docs/evaluation-panels/writing-adjacent-range-app-2026-11-15.json
```

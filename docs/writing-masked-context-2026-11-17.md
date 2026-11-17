# Can masked words locate short AI rewrites?

This single-upload experiment asks whether a separate writing signal
adds correctly located passages to the current text-style checks.
Each test file supplies only its own text at inference. The comparison
with an earlier published version is outside this experiment.

The 45 MB masked-language checkpoint is pinned to the official
Apache-2.0 release. It predicts up to eight evenly placed words
per eligible sentence. For multi-piece words, each prediction hides
the target piece and all subsequent pieces of that word. The model
scores the same target in the same input positions twice. One reading
allows up to 24 surrounding tokens on each side. The other restricts
attention to the focus sentence and the two sequence boundary tokens.
[Pinned checkpoint and terms](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4/tree/387825ce42dbb39b87911cdf8e383ee3b25184f8)
· [Whole-word scoring method](https://aclanthology.org/2023.acl-short.80/)
· [Research on localized authorship evidence](https://arxiv.org/abs/2607.21458)

## Frozen comparison

Candidates span one to three sentences and 24 to 192 focus tokens.
Four readings compare the central words’ predictability, the loss
change when nearby text is removed, predictability relative to nearby
sentences, and a combination learned from paired development files.
Each reading can add at most one distinct passage to the existing
production interval selection. These are ranking measurements.

Eighty-six earlier ordinary files set within-length scales and the two
operating points using source-group maxima. The 32 paired rewrites help
fit the combined weight. Those 118 deliveries span 39 source groups.
A separate 118 doc pairs from 78 source groups selected the primary
method before any held inference. Selection values two points per added edit location,
minus two per additional ordinary alert and one per misplaced
addition. The stricter masked-word reading ranked first. It added
zero correct locations, three ordinary flags, and three misplaced
additions on the selection panel.

The held panels use 76 previously studied but source-separated
historical prose, economic report and GAO groups, another 48 retained
GAO groups, and 48 newly reserved GAO groups. The 76-group panel
keeps one document per source group before inference. The new
government reports exclude all prior report IDs, native text hashes,
and known group hashes from the recorded exposure list and the other
three input plans. Their originals match the government transcript
edition, and two published generator families supply the rewrites.
Checkpoint pretraining exposure is unknown. A correct match covers
half the labeled span and reaches 0.30 intersection over union on
non-whitespace character positions. Each labeled span can match once.
[Prior mixed-passages study](writing-entropy-prominence-2026-11-17.md)
· [GovReport terms and sources](https://gov-report-data.github.io/#license)
· [OpAI-Bench data and terms](https://huggingface.co/datasets/OpAI-Bench1/OpAI-Bench)

## Held results

The existing style checks located 14 of 499 labeled edits and flagged
11 of 172 ordinary files. The selected masked-word check assessed
342 of 344 deliveries, added zero correct locations and zero ordinary
alerts, and added one range that failed to reach the labeled edit.
The fresh 48-pair report panel had 162 labeled edits. The incumbent
found one, and the new primary check added zero.

| Additional reading | Correct additions, strict → wider | Added ordinary alerts, strict → wider | Misplaced additions, strict → wider |
| --- | ---: | ---: | ---: |
| Masked-word predictability | 0 → 0 | 0 → 12 | 1 → 11 |
| Less support from surrounding text | 0 → 3 | 2 → 13 | 2 → 13 |
| Predictability relative to nearby sentences | 1 → 1 | 12 → 18 | 11 → 13 |
| Paired combination | 1 → 5 | 6 → 35 | 6 → 26 |

The exact masked-word score reached 36 additional correct locations
when every top-ranked proposal was counted before the decision point.
Among the 485 edits missed by the current check, a suitable interval
exists for 436 somewhere in the eligible range list. The latter is
an oracle using the labels and the full range list. These measurements
point to poor ranking and weak separation from ordinary text.
The candidate stays in evaluation.
[Frozen operating points](evaluation-panels/writing-masked-context-frozen-2026-11-17.json)
· [Full results](evaluation-panels/writing-masked-context-outcomes-2026-11-17.json)

In a separate replay on the same fresh 48 report pairs, the app's
published-wording check matched a verified transcript for 74 of 96
files. It located 116 of the 162 known changes and flagged no
untouched reports. This check measures wording differences between
linked versions. [Fresh published-wording replay](evaluation-panels/writing-masked-context-archive-replay-2026-11-17.json.gz)

## Independent replay

The receipts preserve exact source fingerprints, native character
positions, sampled word offsets, sentence scores and current findings.
Standard-library Python 3.12 and 3.14 reproduce the scale fit and
results. A separate one-to-one matcher agrees on every count. A second
tokenization pass verified 143,276 sampled words in 18,040 sentences
across the 698 development and held deliveries.

```sh
export PYTHONPATH=backend
root=docs/evaluation-panels/writing-masked-context
date=2026-11-17
python3 -B -S -m scripts.writing_masked_context verify \
  "$root-development-$date.json.gz" "$root-frozen-$date.json" \
  "$root-earlier-transfer-$date.json.gz" "$root-retained-reports-$date.json.gz" \
  "$root-fresh-reports-$date.json.gz"
```

For the newest report panel, rebuild the raw plan from the pinned
OpAI-Bench shards, verified GovReport pack and three native input plans
from the earlier writing studies.
The public source roster records exactly which earlier receipts to exclude.
The replay regenerated the fresh input plan byte for byte.
[Earlier and retained mixed plans](writing-entropy-prominence-2026-11-17.md#independent-replay)
· [Retained additional reports](writing-archive-sentences-2026-11-17.md#replay-locally)
· [Fresh report roster](evaluation-panels/writing-masked-context-fresh-sources-2026-11-17.json.gz)

```sh
export PYTHONPATH=backend
runner=backend/.venv/bin/python
fresh="$root-fresh-reports-$date.json.gz"
"$runner" -B -m scripts.review_writing_masked_context prepare \
  "$shards" "$gao" "$out/new-report-plan.json.gz" \
  "$older/older-plan.json.gz" "$older/held-plan.json.gz" "$older/archive-confirmation-plan.json.gz" \
  --reference "$root-fresh-sources-$date.json.gz"
"$runner" -B -m scripts.review_writing_masked_context audit \
  "$out/new-report-plan.json.gz" "$fresh" "$checkpoint/vocab.txt"
python3 -B -S -m scripts.review_writing_masked_context export \
  "$out/new-report-plan.json.gz" "$fresh" "$root-frozen-$date.json" "$out/independent"
```

The export verifies the native bytes and their earlier transcripts for
all 96 deliveries. It produces files for the [independent evaluation
commands](independent-testing.md), with input bytes, location truth and
predictions in separate files. Its native character positions count
whitespace. Both readers reproduce the current check's one correct
location and zero addition on the fresh panel. Source prose and weights
stay local. [File and location verification](evaluation-panels/writing-masked-context-independent-2026-11-17.json)

To repeat the original inference on the Mac, place the three pinned
checkpoint files in `$checkpoint`. Use the current classifier's
verified snapshot as `$incumbent`. The freeze records the Torch,
Transformers and tokenizer versions and the selected compute device.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$runner" -B -m scripts.writing_multi_passage_panel score \
  "$out/new-report-plan.json.gz" "$incumbent" "$out/incumbent.json.gz"
"$runner" -B -m scripts.writing_masked_context capture \
  "$out/new-report-plan.json.gz" "$out/incumbent.json.gz" "$checkpoint" \
  "$out/masked-readings.json.gz" --device mps --freeze "$root-frozen-$date.json"
```

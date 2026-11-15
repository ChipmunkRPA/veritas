# Score exact word ranges in single-upload writing checks

The existing experimental word check scores each interior sentence and then
highlights a set of high-scoring words. This comparison trains a small head
to score contiguous ranges directly. It reads the submitted excerpt alone.
It uses the existing Apache-2.0 six-layer encoder and the same word features
from the full excerpt and the view with surrounding sentences hidden.
Span-based text localization motivated direct supervision, and research on
AI editing motivated keeping light rewrites separate from fully generated
documents.
[Direct span localization research](https://aclanthology.org/2026.findings-acl.213/)
· [Editing and generation study](https://aclanthology.org/2026.emnlp-main.1725/)
· [Pinned encoder and original checks](writing-context-response-words-2026-11-15.md)

In each of the three interior sentences, the candidate scores adjacent
ranges of 3, 5, 8, 12, 18 and 24 words and the whole sentence. It reads the
mean features inside the range, their difference from the sentence mean,
and the change on either side. An exact earlier draft supplies the labels
for fitting and evaluation. A useful range covers at least half the changed
words and has word intersection over union of at least 0.30. The head trains
on the qualifying ranges and the eight highest-scoring negative ranges per
file. All ranges in documented genuine revisions are negative examples.
The original fit roster supplied 652 eligible files, including 259
with useful generated-revision ranges. Only separate ordinary documents
set the two comparison points.

## Held and replication results

The strict and wider points below describe research comparisons. Each
cell shows useful rewrite locations or ordinary files flagged, as labeled.
All report test works were excluded from fitting and point selection.
The first report, news, abstract and journal panels had earlier project
exposure. The replication uses 48 different government report works.
It excludes recorded source commitments and matching normalized 12-word
phrases from the report study, and takes 16 unused works per interior
position by source hash before scoring. The replication includes 29
polishes, 13 paraphrases and six compressions. Both checks used their
existing report points. The exact paired originals stayed local.
[Publisher method and source](https://arxiv.org/abs/2606.06481)
· [Source pin and reuse terms](writing-matched-encoder-2026-11-14.md)

| Held outcome | Prior check, tighter → wider | Range check, tighter → wider |
| --- | ---: | ---: |
| Local Qwen report rewrites located | 1 → 6 / 32 | 5 → 5 / 32 |
| Original GPT-5.4 report rewrites located | 0 → 0 / 60 | 0 → 0 / 60 |
| Original ordinary reports flagged | 3 → 14 / 93 | 8 → 19 / 93 |
| Replication GPT-5.4 rewrites located | 0 → 0 / 48 | 0 → 2 / 48 |
| Replication ordinary reports flagged | 3 → 10 / 48 | 8 → 14 / 48 |
| News rewrites located | 0 → 1 / 56 | 0 → 0 / 56 |
| Ordinary news flagged | 9 → 13 / 56 | 10 → 18 / 56 |
| Abstract rewrites located | 2 → 2 / 60 | 1 → 2 / 60 |
| Ordinary abstracts flagged | 8 → 11 / 60 | 3 → 20 / 60 |

At the tighter report point the range check gained five useful local Qwen
locations and lost the prior check's one. It added six ordinary flags and
cleared one. None of the five detections also flagged the corresponding
ordinary sentence. At the wider point in the replication, both useful
GPT-5.4 locations also flagged the same sentence in the untouched original.
The separate journal comparison measured two dated revision pairs. The
range head flagged zero of their four versions at either point. The
earlier check flagged two of four at the wider point.

The target-sentence diagnostic distinguishes the available geometry from
the model's ranking. In the prior GPT-5.4 panel, five changes produced no
changed words under the lexical alignment. Of 52 remaining rewrites the
encoder could assess, 40 could be matched by at least one proposed range.
The top range reached the overlap target in seven when the target sentence
was supplied for evaluation. In the replication, 42 had changed words,
33 had usable candidate geometry and eight had a useful top range when
given the sentence. Improving placement alone would leave most misses.

## Combining the checks

The secondary analysis ranks each check against ordinary source works.
It gives each calibration work a rank against the other works, and uses
the largest of its two ranks to set one comparison point. The combined
rule reads both ranges and counts each work once. This adjusts for the
extra chance of an alert when running two checks. It was added after the
first results, before the replication was scored. The wider combined
point found seven useful local Qwen locations with 18 ordinary report
flags, compared with six and 14 for the prior check alone. It found one
useful GPT-5.4 location on replication with 13 ordinary flags, compared
with zero and ten. It added one useful abstract location with five net
additional ordinary flags and lost the prior useful news location.
The isolated gains and the alerts on untouched matches leave the range
check and the combined rule in evaluation.

[Report measurements](evaluation-panels/writing-supervised-ranges-report-2026-11-15.json.gz)
· [Genre measurements](evaluation-panels/writing-supervised-ranges-transfer-2026-11-15.json.gz)
· [Fresh report measurements](evaluation-panels/writing-supervised-ranges-replication-2026-11-15.json.gz)

## Independent replay

The portable replay checks exact offsets, corpus commitments, operating
points, gains and losses, matched ordinary alerts, and performance when a
target sentence is supplied. It needs only the repository and standard
Python. A second local model fit reproduced the checkpoint and report
responses. A second native replication reproduced the selected works
and all responses byte for byte. The corpus and fit files stay local.

```sh
export PYTHONPATH=backend
module=scripts.writing_supervised_ranges
panels=docs/evaluation-panels
report="$panels/writing-supervised-ranges-report-2026-11-15.json.gz"
transfer="$panels/writing-supervised-ranges-transfer-2026-11-15.json.gz"
replica="$panels/writing-supervised-ranges-replication-2026-11-15.json.gz"
python3 -B -S -m "$module" review "$report"
python3 -B -S -m "$module" review "$transfer" --report "$report"
python3 -B -S -m "$module" review-combined "$transfer" --report "$report"
python3 -B -S -m scripts.writing_range_report_replication review "$replica" --report "$report"
```

For a native run, use the pinned encoder, fitted word weights, report
roster and transfer inputs from the [prior native replay](writing-context-response-words-2026-11-15.md#independent-replay).
The replication uses the pinned `test-reports_gpt-5.4.csv` publisher
shard and the prior context head. The native preparation writes the exact
new source roster in a private scratch directory before scoring.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
"$runner" -B -m "$module" report "$run/report" --roster "$roster" \
  --encoder "$encoder" --encoder-weights "$word_weights"
"$runner" -B -m "$module" transfer "$run/transfer" --roster "$roster" \
  --encoder "$encoder" --encoder-weights "$word_weights" --fitted "$run/report" \
  --matched "$matched" --additional "$additional" --editorial "$elife"
"$runner" -B -m scripts.writing_range_report_replication run "$run/replication" \
  --fitted "$run/report" --roster "$roster" --encoder "$encoder" \
  --encoder-weights "$word_weights" --context-head "$context_head" --source "$gpt_report_shard"
```

# Estimating how much of a sentence was edited

The existing comparisons located **40 of 255** known edits. Two new heads
trained to distinguish or estimate the amount of edited wording added no
locations. The version that also reads writing features added two eLife
locations, from Granite and Qwen rewrites of the same source work, alongside
seven newly flagged ordinary work groups. All three remain offline.

EditLens uses measures of distance between original and edited writing during
training, then predicts editing intensity from a single submission. This
experiment independently applies that idea to exact native sentences. A
training target measures the share of words that overlap the publisher's
edit annotations and fall outside a monotonic exact-word alignment with the
paired original. At assessment time the heads read only one document.
The published EditLens weights and data carry noncommercial terms, so this
trial uses neither. [EditLens paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c7d936e696f57468b6b80a7b1463178c-Abstract-Conference.html) · [Checkpoint terms](https://huggingface.co/pangram/editlens_roberta-large/blob/main/README.md)

The binary and intensity heads use the same three features from the existing
Apache-2.0 classifier's saved native context responses. The third adds word
length, vocabulary variety, punctuation, and local wording continuity. Each
arm selects one exact sentence or a run of up to six sentences from one
paragraph. The heads fit 20 existing GAO works. A separate 24 short and 16
long GAO originals set each point before evaluating source-disjoint works.
The two short report panels each contribute 40 changed files. The longer
panel contributes 120. F1000Research contributes 18 rewrites and 18 dated
controls. Two eLife panels contribute 37 rewrites and 44 untouched or ordinary
same-article controls. Additional controls include 16 forum and 12 USWDS
versions. All these cohorts already had project exposure under earlier
methods. The comparator uses the first six positive native classifier
contexts and the existing longer position check where reported for journal
excerpts. [GAO reports](writing-interval-transfer-2026-11-13.md) · [Short edits](writing-word-edit-transfer-2026-11-13.md) · [F1000Research](writing-f1000-position-transfer-2026-11-13.md) · [eLife](writing-elife-long-position-transfer-2026-11-13.md)

| Assessment | Edits | Existing location | Existing plus binary | Existing plus intensity | Existing plus intensity and style |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short GAO A | 40 | 8 | 8 | 8 | 8 |
| Short GAO B | 40 | 12 | 12 | 12 | 12 |
| Longer GAO | 120 | 16 | 16 | 16 | 16 |
| F1000Research | 18 | 4 | 4 | 4 | 4 |
| eLife | 37 | 0 | 0 | 0 | 2 |

A useful location covers at least half the edited non-whitespace characters
and reaches 0.30 intersection over union. The intensity-and-style head added
wrong locations to eight eLife and 37 longer GAO edited files when counted on
its own. It added five ordinary eLife work groups and two ordinary longer
GAO groups to the union of existing checks. The two other heads added zero
ordinary groups and zero useful locations. Every file was assessed by each
arm. The frozen criterion required novel locations on both lengths and
both report generators, two additional journal work or passage results,
and no more than two added ordinary groups per cohort. A document's word
changes or degree of editing alone cannot establish who authored it.

The Python runner rebuilds measured features and gold labels from the
retained native rosters. The [text-free scalar receipt](evaluation-panels/writing-edit-intensity-transfer-2026-11-13.json)
contains the data needed to refit the frozen heads, reproduce the points and
recount the exact locations with standard-library Python. Published upstream
cases and model response hashes remain the source of the incumbent comparator.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.writing_edit_intensity_transfer review \
  docs/evaluation-panels/writing-edit-intensity-transfer-2026-11-13.json
```

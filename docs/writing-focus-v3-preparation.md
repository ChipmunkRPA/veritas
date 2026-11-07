# Writing evidence: prepare a comparison of focus and context

6 November 2026. Reconstructed the exact40 consumed mixed documents and674
sentence/context inputs before any new authorship classifier forward. All674
source positions, context hashes, retained-character fractions and extracted-text
hashes match the original records. The40 authors remain separate source groups.
No AI-labeled non-whitespace characters fall outside the prepared sentence units.
This is input/provenance preparation, not a detection improvement or fresh test.

## The question and its limits

A classifier reading surrounding sentences can react to AI wording outside the
sentence highlighted for the user. The next comparison must distinguish the
highlighted focus from the actual scored context, using exact Unicode codepoint
positions and complete model-native token coverage. Position accuracy alone does
not show that a classifier can determine the origin of individual words.

The publisher's `User`/`Bot` spans describe retained-character provenance.
Human-origin sentences inside an AI-assisted document are not independently
verified unassisted-human controls. Their highlight counts must be labeled as
human-origin highlights, not a SaaS population false-alarm rate. The current
four-class detector also covers AI rewriting/humanization; that broader response
must not be silently relabeled as the probability of retained Bot characters.

## Alternative model and source preparation

Acquired `Oxidane/tmr-ai-text-detector` at
`0ceddea903015ef99cbaa040a4d8a216aed9c683`. Its [pinned model card](https://huggingface.co/Oxidane/tmr-ai-text-detector/blob/0ceddea903015ef99cbaa040a4d8a216aed9c683/README.md)
declares MIT in metadata and prose. The499MB safetensors file matches publisher
SHA-256 `a74ea30f5db6a1094510be903ce374426191addaf7d860b766077de246d32358`.
Every downloaded asset matches its pinned Git blob or LFS checksum. Standard
RoBERTa construction strictly loads201 tensor entries and124,647,170 frozen
parameters; tokenizer is native/fast, label0 human and label1 AI. No custom remote
code, source-text inference or fitting was performed. No checkpoint is in Git.

The card declares RAID training; RAID cannot serve as presumed unseen validation
for this candidate. Published vendor measurements were inspected during source
review, so those small vendor panels would be development, not blind confirmation.
Unknown exact pretraining overlap remains unverified. Published model scores are
not substitutes for our local source-bound comparison.

The original AITDNA metadata/bundled-README license conflict remains recorded in
the [source audit](writing-source-preflight-v2.md). This preparation reads the
already acquired consumed copy; it grants no new training or raw-text/annotation
redistribution rights. The private input packet contains source wording and
labels. Only aggregate preparation and model-source metadata are published.

## Verified preparation and next stages

Twelve focused tests pass: Unicode/codepoint positions, whitespace normalization,
source bounds/conflicts, missing character attribution and refusal to treat an
empty or whitespace-only focus as evidence. Complete preparation uses current
three-sentence contexts, preserving all674 old inputs and charging uncovered
characters explicitly rather than silently omitting them. The
[aggregate receipt](evaluation-panels/writing-v3-focus-preparation.json) contains
no author names, text, queries, per-record labels or classifier scores.

```sh
PYTHONPATH=backend backend/.venv/bin/python -m scripts.prepare_writing_focus_v3 \
  /path/to/text-upgrade/manifest.json /path/to/sources/ait-span.parquet \
  /path/to/text-upgrade/passages-tropa.json /path/to/new-private-inputs.jsonl \
  --window-sentences 3
```

Next freeze a complete classifier comparison: rerun the actual current detector
on the prepared contexts, score the new model on the same contexts and separately
on the exact focus sentences. Report retained-character overlap, human-origin
highlights, pure-Bot recovery and complete coverage, grouped by source author.
Fix any diagnostic reference points before new scores; never report them as
public screening claims or calibrated confidence. Context scores remain evidence
about contexts. No model can be selected for precise focus evidence merely by
having a better context-level AUC.

Serving changes require genuinely unassisted-human controls, source-disjoint
authors and generator families, appropriate calibration, and retained support
for rewriting/email cases. This consumed mixed panel cannot satisfy those
requirements alone. Whole-document confidence remains withheld and current
service defaults are unchanged. Actual outside reviewers remain zero.

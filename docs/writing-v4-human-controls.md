# Human-only writing controls: source audit before inference

7 November 2026. Prepared a stronger false-alarm control inventory without
scoring text or changing a detector. The prior writing comparison measured
retained User-character highlights inside assisted documents. It could not
establish a false-alarm rate for unassisted human writing. This audit checks
unused human-only study documents against their original edit histories.

The institutional AITDNA archive contains the complete raw-history parquet as
well as the previously matched span parquet. All362 raw/span metadata records
match. Excluding the40 authors already consumed by the mixed-writing experiment
leaves53 human-only documents from49 other authors. None has an exact normalized
match or a near match to any of the680 previously consumed full documents under
the declared five-word-shingle rule. Shared topics, task prompts and unpublished
detector training overlap remain possible; author separation is not topic or
pretraining separation.

## What the raw history supports

The publisher's [study description](https://arxiv.org/html/2606.04906v2#S4.SS1)
distinguishes a human-only writing condition and reports filtering long external
copy-paste events. The [pinned source implementation](https://github.com/UKPLab/arxiv2026-aitdna/blob/28e3776681b563b5fb25ef137e33ba0e6147209a/aitdna/notions/AITDNotions.py)
replays source-ordered insert/delete operations with Python character indices;
formatting retains do not change text. That implementation's permissive slicing
is not evidence that every original editor coordinate is valid.

Our separate strict replay requires only User edit events, unique event IDs,
finite nondecreasing timestamps, in-bounds source operations and exact final
text correspondence. It never reorders by the repeated within-batch `order`
field, clamps invalid positions or inserts missing characters. Coordinates use
the publisher's Python codepoint convention; this is not independent validation
of original browser UTF-16 indexing. Synthetic astral and combining-character
tests distinguish those conventions.

Of53 controls, **45 from42 authors** reproduce their entire final source text
with valid histories. **Eight remain unresolved**: six have invalid insert/delete
positions, and two more have out-of-bounds formatting retains. An initial
diagnostic that checked only text-changing operations supported47; checking
formatting positions as well yields the stricter45. No classifier result
influenced this distinction. All53 remain in the private inventory, including
each unresolved history and the reason. There are zero observed AI requests or
Bot edits in the53 publisher-human-only histories; recorded edit consistency
cannot establish the absence of every possible off-platform assistance method.

The full inventory spans21 creative,16 argumentative and16 explanatory writing
tasks;36 are in the standard and17 in the scientific setting. These small,
related study samples do not represent all email, document, language or customer
populations. No population false-alarm rate or accuracy result is reported yet.

## Reproduce the audit

```sh
PYTHONPATH=backend python -m scripts.audit_writing_controls_v4 \
  /path/to/institutional/AITDNA.zip \
  /path/to/text-upgrade/sources/ait-span.parquet \
  /path/to/text-upgrade/manifest.json \
  /path/to/new-private-inventory.json \
  /path/to/new-public-aggregate.json
```

This needs PyArrow for the original data and standard Python for replay,
hashing, overlap analysis and reduction. It loads no classifier, decoder or
pretrained weights. The exact archive, raw39,360,952-byte member, span source,
old manifest, preparation source and private inventory are checksum-bound.
[Public aggregate receipt](evaluation-panels/writing-v4-human-control-source-audit.json).
Ten tests pass, including source-order replay, Unicode, invalid bounds, AI
interaction refusal, duplicate IDs, invalid clocks and a hand-constructed
near-duplicate example.

The source terms conflict remains: institutional CC-BY-SA4 metadata versus
bundled GPL2 prose. Only code, references, checksums and aggregate counts are
published. Source words, names, prompts, raw edit logs and per-document
annotations remain private. This audit grants no new training or source-text
redistribution permission, and no third-party code was executed.

## Next evaluation and boundaries

Freeze an incumbent-only control evaluation before classifier inference. Retain
all53 planned controls; unresolved provenance must be explicit unsupported
source, never an original-file vote or a silently dropped denominator. Report
false highlights and coverage separately for document, passage and author,
with grouped uncertainty, under fixed research operating points. A raw response
is not calibrated authorship confidence. Do not use these controls to rescue
the rejected TMR replacement or tune a point after results are seen.

Run neural evaluation after the active full-audio extraction, keeping the
resource-heavy tasks sequential. Genuine AI/mixed-document coverage, modern
generator families, email/rewriting behavior and later source-disjoint
confirmation remain necessary for any future replacement or calibration.
No fitting, serving change, report change or confidence promotion follows from
this source audit. Actual outside human reviewers remain zero; GCP stays last.

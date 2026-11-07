# Frozen design for the ModernBERT writing comparison

Use the same consumed40 mixed documents,40 author groups and674 exact
focus/context pairs as v3. Prepared input SHA-256 remains
`fb7dbfa725728926c86c6627e7871ba8fcc9e0faa50bba384b13992c624a9752`.
Do not score unused human or mixed authors at this stage. No training,
calibration or serving adoption is permitted by this development comparison.

## Model and input changes

Compare the actual unchanged current Tropa classifier with the permitted
ModernBERT candidate at `e047d9c56f51a7e5c5ef5b0dd556f519bb9dd624`.
The [acquisition review](writing-v4-modernbert-preparation.md) retains exact
cards, notices, tokenizer/config/weight digests, training overlap and saved
problem-type metadata. Source terms remain unresolved as documented; source
words, edit histories, author groups and per-passage labels stay private.

Retain all four arms: baseline/context, baseline/focus, candidate/context and
candidate/focus. The primary comparison is candidate/focus versus
baseline/context. Each classifier reads the exact focus sentence or the exact
three-sentence context that is recorded with it. Context responses are about
the scored context; projecting them onto the focus does not prove individual
word origins. All input positions are immutable Unicode codepoints, end exclusive.

Use CPUfloat32/four threads and eager attention. Candidate construction uses the
installed standard architecture, strict complete138-tensor/149,606,402-parameter
loading, native fast tokenizer and the publisher's positive index1. No remote
code, compiler fallback, temperature or publisher operating point is adopted.
Retain `problem_type=multi_label_classification`; supply no training labels and
use the publisher's two-logit softmax inference convention. The baseline still
uses `1 - p(human)`, including rewriting/humanization classes; the binary candidate
cannot inherit that broader target or a calibrated confidence claim.

Preserve512-token processing, batch4 and every stride0 tokenizer overflow window,
with content-token-weighted aggregation. Independently compare concatenated
overflow content IDs to the complete native tokenization, with their hashes.
No truncation-only shortcut, dropped token, guessed score or human/original vote
for a failed input is allowed. The native8192-position capacity is not tested as
a long-document detection claim by this comparison.

## Required runtime preparation and prospective freeze

Run the synthetic check only after the active heavy audio evaluation is finished.
The new loader rejects simultaneous audio/video/writing evaluation processes.
Three fixed synthetic inputs include Unicode and actual tokenizer overflow.
Require strict CPUfloat32 loading, identical repeated batch4 responses,
batch1/batch4 agreement and separate native single-window aggregation within
absolute1e-5, with exact complete token coverage. No project text is used for
this runtime check. It is execution evidence, not detection accuracy.

The comparison lock must bind that completed synthetic receipt, all candidate
and baseline assets, original prepared/historical input digests, project and
critical installed implementations, Python/package/platform versions, this
protocol and every comparison rule. Publish and verify the complete lock before
project classifier inference. A missing, failed, altered or stale synthetic
receipt cannot unlock project scoring. A preliminary code release is not a
completed runtime or execution freeze.

Retain every planned document and all674 positions. Processing failures are
written explicitly, private error traces are retained and the journal stops.
Only a valid prefix under the exact same lock can resume an interrupted journal;
never overwrite, merge failed attempts or omit difficult positions. Native replay
covers the first focus of every author in both views/models and every additional
overflow input, with exact input-token bindings and absolute score tolerance1e-5.
The actual baseline is rerun; old scores are used only for correspondence checks.

## Fixed analysis and continuation rules

Retain inclusive research references0.5 and0.8; they are not public screening
claims or calibrated confidence levels. At each reference report all pure-Bot
recoveries, pure-User-origin highlights, mixed highlights and complete character
counts/precision/recall/IoU. User-origin highlights inside assisted documents
remain distinct from false alarms on verified unassisted writing. Also retain
pure-origin rank AUC, rank correlation, focus/context shifts and complete token
coverage for every arm. Failed/unassessed inputs cannot become human evidence.

Bootstrap paired author-mean IoU by the40 original author groups,2000 replicates,
seed20261106 and percentile2.5/97.5 bounds. Preserve all nine v3 development
requirements: complete no-failure scoring; and at both references, more pure-Bot
recoveries, no decrease in Bot-character recall, pure-User-origin highlight rate
at most two percentage points above baseline, and strictly positive lower bound
for paired author-mean IoU improvement. The original arithmetic functions are
shared unchanged. No point or gate may be selected or relaxed after responses.

Passing numerical quality alone does not permit fresh confirmation or adoption.
Both native replay receipts, historical baseline correspondence within1e-5 and
separate complete arithmetic reconstruction remain necessary. A failed native
audit is preserved as failure. A complete rejection is a valid outcome.

Only a fully passing development study can lead to a separately frozen unused
author/generator comparison. The [53-control source inventory](writing-v4-human-controls.md)
retains45 supported and eight unresolved histories; none is consumed here to
rescue a rejected candidate or tune an operating point. Future calibration,
appropriate email/rewriting cases, modern-generator coverage and full UI/PDF/JSON
prepaid validation remain required before any serving or overall-confidence
change. Actual outside human reviewers remain zero. GCP remains last.

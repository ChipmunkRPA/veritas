# Layout-consistent and complementary passage evidence: development comparison

This follow-up implements the normalization direction discussed after the
[formatting audit](text-format-comparison-v94-result.md). The earlier results
are already known, including lost AI highlights after flattening. This is an
explicit **post-result development comparison**, not a blind confirmatory test.
No new sources, neural calls, learned parameters or customer confidence changes
are part of it. The existing records use the pinned Apache-2.0 checkpoints and
our own retained interval head; the source credit and CC BY 4.0 publisher terms
remain in the earlier result.

## Fixed methods and inputs

Retain all 288 controlled 160-word documents from 32 known source/prompt groups,
all five requested views and all recorded responses. Canonicalize a prose input
by replacing each whitespace run with one ordinary space and trimming outside
whitespace. Preserve every non-whitespace Unicode codepoint, punctuation, case,
token boundary and actual original source position. This can change the meaning
of layout-dependent material; it is a research prose method, not automatic
normalization of code, tables or email headers.

Require each canonical string **and its complete source correspondence** to
match exactly one effective native input already scored in v94. Reuse its
actual current/interval responses only. All five layouts must converge to the
same canonical input and source correspondence. A text mismatch is a refusal,
not permission to substitute a similar response or manufacture a model call.

Compare four methods at the unchanged, separately recorded v7 research points:

1. Current passage check on the submitted layout (baseline).
2. Current passage check on the canonical input.
3. Union of current and interval highlights on the canonical input.
4. Union of baseline and both canonical checks.

Methods 2 and 3 should be layout-consistent by construction. Method 4 retains a
layout-dependent baseline and must not be described as invariant. Unions combine
source-character masks, not probabilities. Overlapping responses and two views
of one classifier are not independent evidence. Missing components remain
unassessed; uncovered source characters remain uncovered.

## Measurements and interpretation

Report all methods and recipe strata: AI characters located/missed, human
characters wrongly highlighted, unscored AI/human characters, character IoU,
useful locations (IoU >= .3), known AI intervals at least half recovered,
genuine documents and source groups flagged, and AI-containing documents with
no AI characters located. Preserve each requested view's result; equal inputs
are not independent confirmations. Report useful locations gained and lost
against the same-view baseline and highlight changes from each method's
original view. These are internal research definitions, not public thresholds.

Do not tune points or choose a new method from one favorable subgroup. A
moderate increase in false alarms is not an automatic rejection: show gains,
both wrong directions, coverage and source-group behavior together. No automatic
serving admission is possible from this reused panel. Any deployment candidate
requires its own calibration and new natural articles, emails and mixed-text
confirmation with source/generator separation. No universal originality or
authorship probability follows from a stable or empty highlight.

## Execution and publication

Freeze this procedure, implementation, authored tests, prior packet, all eight
response journals and existing operating points before computing the new
comparison. Manually publish and verify that freeze. No prior study files are
modified. Exclusive directories/files preserve an attempted stage; no automatic
retry or overwrite. This is a single-process, non-neural replay: 900 seconds,
1.5 GiB self RSS, 3 GiB initial available RAM/disk, 1.5 GiB running reserves,
observed per document rather than continuously. Imports of neural/scientific
libraries are refused. Existing neural budgets remain unchanged.

The per-case journal stays private. Publish aggregate results and whole-artifact
hashes only. A separate arithmetic implementation is still required before
interpreting a candidate as qualified. Outside human review remains outstanding;
no outreach, workflow, real charge or GCP deployment is authorized by this study.

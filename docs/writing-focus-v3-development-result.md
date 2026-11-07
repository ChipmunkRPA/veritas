# Writing localization: keep the current context check

6 November 2026. **Reject the proposed candidate/focus replacement.** All 40
consumed documents and 674 foci completed across both models and both views,
with no omitted tokens or failed documents. The candidate recovers slightly
more retained Bot-origin sentences but highlights far more User-origin
sentences. Four of nine unchanged continuation requirements fail. No fresh
authors, calibration, serving replacement or whole-document confidence update
follows from this comparison.

## What the comparison found

The [prospective protocol](writing-focus-v3-protocol.md), implementation and
[freeze](evaluation-panels/writing-v3-focus-freeze.json) were published and
remote-verified as `7f19b24bbb6672cd5c7c52dd4713a748f7406452` before new
classifier forwards on project text. This 2×2 design separates the model change
from removing the neighboring sentences. The current check combines generation,
rewriting and humanization responses; the candidate is a binary human/AI
classifier. Their raw responses are not interchangeable calibrated probabilities.

| Arm | Bot-origin sentences highlighted / 322 | User-origin sentences highlighted / 153 | Bot-character recall | Character IoU |
|---|---:|---:|---:|---:|
| Current model, context |295|52|89.35%|73.53%|
| Current model, focus only |202|43|64.93%|55.44%|
| Candidate, context |303|143|94.99%|65.85%|
| Candidate, focus only |306|136|95.78%|67.14%|

This table uses the frozen **research raw-response reference 0.5**. At the other
fixed reference, 0.8, the same arms recover 276/150/291/287 Bot-origin sentences
and highlight 38/25/129/108 User-origin sentences respectively. The candidate's
extra 11 recoveries over the current context arm at each reference cost 84 and 70
additional pure-User-origin highlights. Neither tradeoff qualifies as better
localization under the predeclared allowance.

There are also 199 mixed-origin foci. The complete
[aggregate result](evaluation-panels/writing-v3-focus-development.json) retains
their highlights and character overlap, every character denominator, native
token coverage, ranks, focus/context shifts and all nine continuation decisions.
Below-reference scores mean this research check did not highlight a focus;
they do not establish human authorship or the absence of AI assistance.

## Why neither simple change should replace the current check

Removing context from the current model loses 93 pure-Bot recoveries at the first
reference and 126 at the second, while reducing pure-User-origin highlights by
only 9 and 13. The mean raw focus response is 0.200 lower than the context response,
and 177/674 foci are highlighted by context alone at the first reference.
This is evidence that neighboring text materially affects the current score.
It does not identify which specific neighboring words caused it. Preserve the
app/PDF distinction between the selected passage and what the check actually read.

The candidate is more sensitive on this consumed panel, but its rank separation
of pure retained origins is worse: focus AUC 0.718 versus the current context
arm's 0.878. Its focus response highlights 88.9% of pure-User-origin sentences
at the first reference and 70.6% at the second. These are **human-origin highlight
rates inside AI-assisted documents**, not population false-alarm rates for
unassisted human writing. They nevertheless undermine the proposed precise
focus-localization replacement.

Paired mean author IoU gain is −0.00587 at the first reference, with a 95%
author-bootstrap interval [−0.04606, 0.03585]. At the second it is +0.01056,
interval [−0.02548, 0.05011]. Both intervals cross zero. Resampling preserves the
40 author groups; the 674 related passages are not 674 independent people.
The fixed gates reject both human-origin highlight increases and both author-IoU
interval requirements. The remaining five requirements pass. No response-driven
threshold search or gate relaxation was performed.

## Execution and arithmetic checks

All 2,696 model/view inputs were scored with complete native-token coverage.
The earlier current-context records reproduce within 1.1921e−7 on all 674 inputs;
none differ by more than 1e−5. The actual unchanged serving classifier is rerun
for the baseline, rather than substituting those historical scores.

Both predeclared direct replay audits pass: 80 input/score bindings per model,
including the first focus of every author in both views. Independent unbatched
model forwards and reduction differ from recorded scores by at most 2.325e−6
for the current model and 1.148e−6 for the candidate, within the frozen 1e−5
tolerance. There are no actual overflow inputs; synthetic tests cover missing,
duplicated and reordered overflow tokens. Do not claim the panel empirically
validated long-input behavior.

The separate arithmetic reviewer reproduces all four arms' counts, rank AUC,
rank correlation, character rates, context/focus shifts, paired bootstrap
intervals and decisions. It uses separate rank/count/reduction code with the
declared NumPy random generator. Six hand-computed arithmetic tests pass.
The 45 original selected tests and 15 publication-scanner checks also pass.
These are local execution and numerical checks, not outside human validation.
[Numerical review receipt](evaluation-panels/writing-v3-focus-numerical-review.json).

The current model retains its Apache-2.0 release and the candidate its MIT
declaration, pinned assets and revisions. No checkpoint parameters are changed
or published. The source-data terms conflict remains recorded: private source
wording, author groups, queries, passage annotations and response journals stay
local. The public result contains aggregates and artifact checksums only.
See [execution and reproduction](writing-focus-v3-execution.md).
Exact execution requires separately obtained source files and the historical
local input/response artifacts. The public aggregates alone cannot replay neural
responses or source positions. No outside reproduction has been completed, and
this release does not claim to be a self-contained writing evaluation package.

## Next methodological step

Keep the current context-based inspection evidence and its limitations. Test
genuinely unassisted controls separately, preserve rewriting/humanization and
email cases, and acquire source-disjoint authors and generator families before
calibrating or selecting a future replacement. A future localization method
must improve the alignment between the claim and scored input while retaining
useful AI recovery. This consumed comparison supplies a rejection and a clearer
failure diagnosis, not fresh customer accuracy, word-origin proof or a legal
authorship conclusion. Actual outside human reviewers remain zero.

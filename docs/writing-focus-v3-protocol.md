# Frozen writing focus/context development comparison

6 November 2026. This prospective protocol will be published with the complete
implementation and source/model/runtime freeze before a new classifier scores
project text. No fitting, calibration, fresh confirmation or serving adoption
is permitted by this experiment.

## Question and fixed inputs

Use the exact prepared 40 consumed mixed documents, 40 declared author groups
and 674 sentence foci. Input SHA-256 is
`fb7dbfa725728926c86c6627e7871ba8fcc9e0faa50bba384b13992c624a9752`.
All original three-sentence contexts, focus Unicode codepoint positions and
retained User/Bot character provenance are bound before new inference. Zero
Bot-labeled non-whitespace characters are omitted. No unused authors are scored.

The four predeclared arms separate model and input changes: current Tropa on
context, current Tropa on focus, TMR on context, and TMR on focus. The primary
development comparison is TMR/focus versus current Tropa/context. All arms and
all documents remain in the result; no convenient subset selects a winner.

Use unchanged `FourClassTextClassifier.predict` for the current baseline, with
its combined generation/rewriting response `1 - p(human)`. TMR uses softmax
class 1 without temperature fitting. Both run on CPU float32, four threads,
batch four and maximum 512 tokens including native special tokens. Keep every
non-overlapping tokenizer overflow window, weighted by content-token count.
Independently verify each input's full token IDs against concatenated overflow
content IDs; retain token counts, window counts and exact input hashes.
Processing failure is retained and blocks continuation, rather than becoming a
human verdict or disappearing from denominators. Interrupted journals resume
only if all prior identities and scores still match the same freeze.
Replay the first focus of every author on both views/models, plus every input
with tokenizer overflow. Use independently assembled direct single-window
forwards and content-token aggregation, with absolute score tolerance `1e-5`.
This verifies the recorded local execution; it is not outside human review.

## Models and permitted use

The installed Apache-2.0 Tropa release is pinned at
`f1795c86806e6838d4afa33d0b1427f8430c9615`; its six serving assets remain exact.
The MIT TMR release is pinned at
`0ceddea903015ef99cbaa040a4d8a216aed9c683`; all eight downloaded assets are
bound, including card and weight checksums. Only standard locally installed
Transformers architectures load, strictly, without custom remote code or
pickle. No pretrained parameters are changed or redistributed.

TMR was trained on RAID. Public vendor measurements were read during source
selection, and exact pretraining overlap remains unknown. This consumed panel
cannot establish fresh accuracy. The AITDNA source terms conflict remains
recorded in the source audit: no new training or source-text/annotation
redistribution is authorized by this experiment. Only aggregates and checksums
will be public. Private records retain exact text positions and source binding.

## Analysis fixed before responses

Use diagnostic raw response references 0.5 and 0.8 with inclusive `>=`.
These are research operating points, not public claims or confidence levels.
For each arm and reference report all pure-Bot recoveries, pure-User-origin
highlights, mixed highlights, located Bot characters, highlighted User
characters, character precision/recall/IoU and coverage. Scores of surrounding
contexts remain context responses; projected focus overlap is a localization
diagnostic, not proof of the origin of individual characters.

Also retain pure-origin rank AUC, correlation with retained Bot-character
fraction, paired focus/context response shifts, and comparison with the earlier
recorded current-context scores. None are calibrated confidence measurements.
Bootstrap paired documents by their 40 author groups, seed 20261106, 2,000
replicates, percentile 2.5/97.5 interval. Do not treat 674 correlated foci as
independent people. For each arm and reference retain the mean author IoU;
primary intervals compare candidate/focus with baseline/context.

Development continuation requires complete no-failure scoring and, at **both**
references: more pure-Bot recoveries, no decrease in Bot-character recall,
pure-User-origin highlight rate at most two percentage points above baseline,
and the lower interval for paired author-mean IoU improvement strictly above
zero. Freeze these rules before scoring; no threshold search or post-result
revision. Passing permits only further validation, never automatic adoption.

## Interpretation and next decision

User-origin passages in AI-assisted documents are not verified unassisted-human
controls; label their counts accordingly. Tropa also detects rewriting and
humanization, while source labels describe retained characters. Those targets
are not interchangeable. Even an improved localization result must retain
appropriate rewriting and email support and pass separately calibrated tests
with genuinely unassisted controls and source-disjoint authors/generators.
Whole-document confidence, customer evidence, payment and serving defaults stay
unchanged. Actual outside human reviewers remain zero until someone participates.

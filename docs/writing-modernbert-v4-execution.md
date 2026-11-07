# Writing comparison: synthetic runtime verified and prospective lock created

7 November 2026. Added a separate runner and arithmetic reviewer for the
[prospective ModernBERT design](writing-modernbert-v4-protocol.md). Original v3
code, results, inputs and model assets remain unchanged. The actual synthetic
check has now passed and its prospective comparison lock has been created.
This is runtime preparation, not a detection-quality result.

## Actual synthetic runtime and lock

The138-tensor/149,606,402-parameter standard model loaded completely as
CPUfloat32 with eager attention. Repeated batch4 responses are identical;
single/batch and separate native response differences are at most
3.5390257835388184e-8, below the original1e-5 requirement. All three authored
synthetic inputs retain every native token, including the1,081-token probe
processed in three windows. No project text, fitting or calibration was used.
[Complete synthetic receipt](evaluation-panels/writing-v4-modernbert-synthetic-runtime.json).

The [prospective public lock](evaluation-panels/writing-v4-modernbert-freeze.json)
binds the successful receipt, all original model/input/historical assets,
14 source files,11 critical installed implementations, runtime and every
unchanged comparison rule. Private lock SHA-256:
`0a239741c9820301b678178f66534afa593d305c2ce09ebba511800413e1dade`.
Its recorded project classifier forwards are zero. Publish and remote-verify
this complete record before starting the actual consumed-document comparison.
No calibrated authorship probability, new-source performance or serving
admission follows from the synthetic pass.

The new implementation keeps the original40 documents/674 focus-context pairs,
four arms, both research references and all nine quality requirements. Seven
coverage/native/reduction function bodies match the original abstract syntax
exactly after schema-name substitution. Shared denominator/input/gate functions
are unchanged. The new candidate loader requires its exact architecture,
parameter count, saved label/problem-type metadata and complete CPUfloat32
state; it rejects missing/mismatched/unexpected keys and unresolved tensors.

Seventy-three selected tests pass:31 new strict-state/runtime/guard tests and42
existing input/coverage/denominator/numerical tests. The live full-audio process
was also observed by the new concurrency guard, which refused writing-model
loading. No pretrained writing model was loaded by that check.

## Complete evaluator compatibility check

Replayed the known old baseline/TMR responses through the new evaluator as a
private fixture. All40 documents,674 foci and four arms are retained. All counts,
ranks, rates, focus/context shifts, author-bootstrap intervals, historical
baseline correspondence and all nine requirements match the old aggregate
exactly. Separate arithmetic reconstruction also passed with the old native
metadata rebound only for fixture validation. All12 original project source
pins were checked unchanged.

This checks evaluator wiring and preserves the known rejection; it is not a
ModernBERT inference, native audit or quality result. The fixture rows, source
words and native metadata remain local. Only the explicit
[compatibility receipt](evaluation-panels/writing-v4-evaluator-compatibility.json)
is public. No old TMR response is used as the new candidate's actual score.

## Execute after the audio work

Finish or review the current heavy audio phase first. Keep the existing offline
model/cache/data directories and commercial policy. Use CPUfloat32/four threads.
Run every stage separately, retaining its output and error log. Begin with the
fixed synthetic native-token/batch/strict-state test:

```sh
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 smoke \
  /path/to/writing-v4-modernbert /path/to/new-smoke.json
```

It must exercise actual overflow, reproduce repeated batched outputs exactly,
retain all tokens and match single/batch/direct-native responses within1e-5.
A failed or missing receipt blocks the prospective project-text lock.

```sh
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 freeze \
  /path/to/writing-v3/inputs.jsonl /path/to/text-upgrade/passages-tropa.json \
  /path/to/current-baseline-snapshot /path/to/writing-v4-modernbert \
  /path/to/new-private-lock.json /path/to/new-public-freeze.json \
  --smoke-receipt /path/to/new-smoke.json
```

Publish and verify the completed prospective freeze before project inference.
The freeze binds the successful synthetic receipt, exact source/model/runtime
and critical installed code. Source paths are private; only hashes and the
synthetic-only receipt are projected publicly. Then run the complete stages:

```sh
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 score \
  /path/to/private-lock.json baseline /path/to/new-baseline.jsonl
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 score \
  /path/to/private-lock.json candidate /path/to/new-candidate.jsonl
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 native \
  /path/to/private-lock.json baseline /path/to/new-baseline.jsonl /path/to/new-baseline-native.json
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 native \
  /path/to/private-lock.json candidate /path/to/new-candidate.jsonl /path/to/new-candidate-native.json
PYTHONPATH=backend python -m scripts.run_writing_modernbert_v4 summarize \
  /path/to/private-lock.json /path/to/new-baseline.jsonl /path/to/new-candidate.jsonl /path/to/new-summary.json
PYTHONPATH=backend python -m scripts.review_writing_modernbert_v4 \
  /path/to/private-lock.json /path/to/new-baseline.jsonl /path/to/new-candidate.jsonl \
  /path/to/new-baseline-native.json /path/to/new-candidate-native.json \
  /path/to/new-summary.json /path/to/new-numerical-review.json
```

Every actual native receipt must pass, baseline/historical responses must match
within1e-5, and separate full arithmetic reconstruction must agree. Preserve
failed attempts and all outcomes; no source omissions, point changes or relaxed
quality requirements may rescue a failed candidate. Review the complete gates
and execution evidence together before any separately frozen unused-author
comparison. Raw responses remain uncalibrated, with context/focus limitations.
Serving models, overall confidence, customer reports, billing, outside human
reviewer count and GCP infrastructure remain unchanged.

# Writing comparison: implementation ready, runtime checks pending

7 November 2026. Added a separate runner and arithmetic reviewer for the
[prospective ModernBERT design](writing-modernbert-v4-protocol.md). Original v3
code, results, inputs and model assets remain unchanged. No ModernBERT model
has been constructed or scored; there is no completed synthetic runtime check,
execution freeze or new detection-quality result yet.

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

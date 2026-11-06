# Audio v9 development verification supplement

This is additional verification of the [existing execution protocol](audio-identity-v9-execution-protocol.md).
The original freeze was published as `4e1c59c` before extraction. This supplement
is prepared and published during consumed extraction, before fitting or
development comparison. It changes neither the original 51 implementation pins,
statistical learner, source selection, internal operating points nor sixteen
advancement gates. No confirmation inputs are scored by these tools.

## Predetermined native coverage

Use the lexicographically first development input in each distinct
dataset/generator/truth/condition cell. The fixed source plan has 99 such cells
across eleven dataset/generator combinations and nine conditions. Selection uses
metadata only, with no score, applicability or candidate-conclusion filter.
Publish the [exact IDs](evaluation-panels/audio-v9-native-development-plan.json)
before fitting and comparison. The plan field `required_before_fitting_and_comparison`
refers to publishing that selection; actual result replay follows comparison.

After complete extraction/comparison, re-decode each selected original waveform
through the production decoder, checking byte/PCM commitments. Re-execute VAD,
window selection and the frozen CPU model/head. Require exact status, source
spans/windows, raw response and representation digest. Replay any computed
stability probe for those same predetermined inputs, including its filtered PCM
digest. A failed probe remains a failure. Preserve all 99 cases, write failures
to the audit receipt and refuse a passing audit when any case disagrees.

This is broader than the separate nineteen successful cache checks, which are
preserved. It is a sample of native development forwards, not neural replay of
every extracted or probe window. Complete extraction binding and numerical
checks apply to every required input independently of the native sample.

## Complete numerical verification

The numerical reviewer checks all 4,248 consumed representations and original
checkpoint-head arithmetic. It reconstructs worst-case connected-group/truth
guard ranks using only the 873 calibration inputs, including unassessed variants.
The fitted object must match its saved fitting-only artifact. With `--refit`,
rerun the unchanged declared learner on only the 1,926 fitting inputs and compare
the complete projection, coefficients and fitting metadata at 1e-10 tolerance.
This refit shares the published learning implementation; it is not a separately
implemented optimizer or outside review.

For all 1,449 development outcomes, separately reconstruct candidate probability,
actual-service probability, class guards, probe requirements, raw and calibrated
stability changes and final conclusions. Failed or unassessed checks cannot
establish an original verdict. Independently count operational outcomes with
complete denominators. Reuse the unchanged declared proper-score/bootstrap/gate
implementation and require every summary field to agree, with numerical tolerance
1e-10 and exact discrete outcomes. Wrapper schema identity remains v9.

These reviews are prerequisites for considering separately frozen confirmation,
not permission to enable the candidate. A failure is preserved; do not relax a
rule, change a point or omit a case to make it pass. Paid-flow, evidence/export,
rollback and any overall-confidence admission remain separate requirements.

## Commands

Use the existing pinned environment, explicit model/data/HF paths and offline
commercial-weight policy. From the repository root:

```sh
PYTHONPATH=backend backend/.venv/bin/python backend/scripts/audit_audio_identity_forwards_v9.py \
  plan /path/to/v9-lock.json /path/to/new-native-plan.json

PYTHONPATH=backend backend/.venv/bin/python backend/scripts/audit_audio_identity_forwards_v9.py \
  run /path/to/v9-lock.json /path/to/new-native-audit.json \
  --plan /path/to/published-native-plan.json --receipt /path/to/representations.jsonl \
  --responses /path/to/development-responses.jsonl

PYTHONPATH=backend backend/.venv/bin/python backend/scripts/review_audio_identity_numerics_v9.py \
  /path/to/v9-lock.json /path/to/representations.jsonl /path/to/trained/head.json \
  /path/to/development-responses.jsonl /path/to/summary.json \
  /path/to/new-numerical-review.json --refit
```

The selected source/identity/execution/learner/guard/independent-review suite
passed 115 tests before use, including sixteen arithmetic/native units and a
summary-wrapper regression. Tests use fixtures and do not establish accuracy.
The [supplemental code commitments](evaluation-panels/audio-v9-result-review-freeze.json)
bind the actual verification implementation. No serving, payment/auth or GCP
configuration is modified, and outside reviewer participation remains unestablished.

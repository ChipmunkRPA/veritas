# Corrected multi-generator audio comparison: fewer false alarms, less useful coverage

6 November 2026. **Rejected under the unchanged sixteen development requirements.**
The candidate makes substantially fewer false AI calls and gives better probability
scores, but returns fewer correct conclusions, especially under noise, music and
equal-level overlapping speech. It passed eleven requirements and failed five.
The running calibration and overall confidence remain unchanged. None of the
1,080 reserved confirmation inputs was scored.

## Scope and frozen design

This is consumed-source development, not fresh accuracy testing or outside review.
The [identity correction](audio-source-identity-v9.md) joins known genuine/cloned
parent voices and retains the original failed v8 design. The [execution freeze](audio-identity-v9-execution.md)
was published as `4e1c59c` before extraction. The [verification supplement](audio-identity-v9-review-protocol.md)
was published as `f9496a7` before fitting/comparison; it changed no learner, source,
operating point or acceptance rule.

Completed all 4,248 four-second consumed inputs: 1,926 fitting, 873 independent
calibration and 1,449 development. They represent 214/97/161 original foregrounds
in 169/82/106 canonical connected groups. Nine variants per original are correlated
observations, not 1,449 independent people or customer uploads. Some genuine and
cloned voices correctly share a group within a role. Unknown biometric and
pretrained-corpus overlap remains unverified.

All 4,248 feature records are terminal: 4,186 assessed, 62 unassessed, zero failed.
There were exactly 1,846 verified exact-input reuses and 2,402 new extraction
records. Development has 1,420 assessed and 29 unassessed inputs. All planned
cases remain in operational denominators; neither system turns skipped speech
into an original-file verdict. Verified commercial-permission checkpoints and
the original runtime were retained; no additional pretrained weights were used.

## Full development outcome

The [complete aggregate JSON](audio-identity-v9-development-evaluation.json)
includes every condition, generator, background and paired uncertainty result.

| Measurement | Current guarded service | Candidate |
|---|---:|---:|
| Correct AI calls / 756 AI inputs | 648 | 574 |
| Correct original calls / 693 genuine inputs | 356 | 298 |
| Genuine inputs incorrectly called AI | 58 (8.37%) | 3 (0.43%) |
| AI inputs incorrectly called original | 0 | 1 (0.13%) |
| AI inputs without an AI conclusion, including abstentions | 108 | 182 |
| Inconclusive / 1,449 planned inputs | 387 (26.71%) | 573 (39.54%) |
| Total correct directional conclusions | 1,004 | 872 |
| Source/class-balanced Brier error; lower is better | 0.094859 | 0.059157 |
| Source/class-balanced log loss; lower is better | 0.317112 | 0.201372 |
| Assessed-input AUC | 0.969312 | 0.979674 |

The Brier difference is -0.035702, with a paired connected-group bootstrap 95%
interval of [-0.048027, -0.024225]. Log-loss difference is -0.115740, interval
[-0.149519, -0.083521]. These describe this consumed panel conditional on the
fixed fitting/calibration and reused backgrounds. They exclude fitting and
background-population uncertainty and are not per-upload confidence intervals.

Better probability scores and fewer false accusations are useful findings.
They do not establish a passing replacement: 132 fewer correct conclusions and
186 more inconclusive results violate the predeclared retention/usefulness
requirements. The extra inconclusive results are not counted as detection gains.

## Background-specific outcome

Each row contains the same 161 foreground originals, including 77 genuine and
84 generated originals. Values show current service → candidate. Background SNR
is a benchmark mixture setting, not a supplied predictor or a customer certainty.

| Condition | Correct conclusions | False AI | AI called original | Inconclusive |
|---|---:|---:|---:|---:|
| Clean |154 → 147|0 → 0|0 → 0|7 → 14|
| Noise, 0 dB |97 → 87|2 → 0|0 → 0|62 → 74|
| Noise, 5 dB |112 → 98|0 → 1|0 → 1|49 → 61|
| Noise, 15 dB |139 → 120|0 → 0|0 → 0|22 → 41|
| Music, 5 dB |95 → 75|0 → 0|0 → 0|66 → 86|
| Music, 15 dB |141 → 112|0 → 0|0 → 0|20 → 49|
| Genuine background speech, 0 dB |75 → 51|11 → 0|0 → 0|75 → 110|
| Genuine background speech, 5 dB |93 → 79|18 → 1|0 → 0|50 → 81|
| Genuine background speech, 15 dB |98 → 103|27 → 1|0 → 0|36 → 57|

Across overlapping-speech variants, false AI calls fell 56 → 2 and correct
original calls rose 32 → 47, but correct AI calls fell 234 → 186. The resulting
net loss is 33 correct conclusions. Strong-noise and music usefulness also
worsened; those conditions cannot be hidden by the pooled Brier improvement.
Clean generated-speech calls were retained, but seven correct clean original
calls became inconclusive.

Failed gates: clean retention, per-generator AI retention, strong-condition
usefulness, combined background-speech usefulness and background-speech class
retention. Both error-direction limits, proper-score gains, complete cases,
matching assessed sets and absence of extraction/probe failures passed. No
threshold or rule was relaxed after seeing the outcome.

## What withheld conclusions

The [descriptive diagnostic](evaluation-panels/audio-v9-development-diagnostic.json)
accounts for all candidate outcomes: 876 directional calls (872 correct, four
wrong), 292 withheld by non-unique class guards, 234 between the internal research
operating points, 18 withheld by stability checks, and 29 unassessed. Of the
292 guard-withheld cases, 245 are genuine and 47 generated. These counts explain
which rules withheld conclusions; they neither prove a physical cause nor
justify weakening a guard. Research operating points are not a public SaaS
accuracy guarantee.

The [guard-only upper bound](evaluation-panels/audio-v9-guard-only-bound.json)
shows that five generator families cannot meet retained AI-call requirements
even if every guard and stability veto is removed while keeping the fitted
probabilities and internal operating points fixed. For example, the optimistic
Chatterbox ceiling is 72 against a required 76; PflowTTS is 98 against 113.
This is descriptive arithmetic, not a new candidate or permission to drop guards.
It rules out a guard-only repair of this fitted result.

Next work needs to improve the response mapping or representation as well as
calibration, potentially using observable signal quality. Any new method needs
its own prospective design, separate calibration sources and the unchanged
complete-case/error/usefulness requirements, including clean/generator retention.
Ground-truth generator or mixture SNR labels must not become serving predictors.
Preserve this result and reserve fresh confirmation for a passing method.

## Verification and public reproduction

All nineteen predetermined cache replays matched before reuse. All 99 separately
predetermined development cells subsequently matched exact native source/VAD/
window/raw-response/representation outputs, including fifteen computed probes.
The complete numerical audit reconstructed both systems' probabilities, guards,
raw-plus-calibrated stability, operational counts and statistical summary.
CPU refit matched the complete fitting artifact at the declared 1e-10 tolerance,
with zero coefficient error.

The [public numerical capsule](evaluation-panels/audio-v9-development-review.json.gz)
is 44,852,410 bytes, SHA-256
`0546940a656ce0f718318975491ee041f2a429be15ac57d316e8e5e3ea6345c3`.
It includes all 4,248 representations, 1,449 outcomes and fitted arithmetic.
No source audio, transcripts, raw provider identifiers, private paths or full
pretrained checkpoints are included. Original source terms and attribution
continue to apply; see the neighbouring source catalogue.

An actual review with an empty model/data cache reproduced the original
93,382,641-byte feature JSONL exactly (SHA-256
`315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d`),
both original fitting artifacts, and all probability/guard/stability/count/gate
arithmetic. Sanitized outcome/summary bytes differ from their private originals;
their original digests are lineage commitments, not claims of byte equality.
Per-background strata use source digests and inherited bootstrap descriptions
are corrected to consumed-development scope without changing numeric results.

From the repository root, with the existing Python dependencies and all four
referenced public capsules beside the result:

```sh
PYTHONPATH=backend OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python backend/scripts/publish_audio_identity_results_v9.py review \
  docs/evaluation-panels/audio-v9-development-review.json.gz \
  /tmp/new-audio-v9-result-review.json --refit
```

The [portable review receipt](evaluation-panels/audio-v9-development-review-reproduction.json)
records its actual scope. Native waveform/model execution is not performed by
this numerical command. [Native](evaluation-panels/audio-v9-native-development-review.json)
and [numerical](evaluation-panels/audio-v9-development-numerical-review.json)
local receipts are separate; none constitutes an outside human review. No app,
payment/auth, overall-confidence or GCP deployment change follows from this study.

All 124 selected source/learner/guard/execution/export/independent-review tests
passed. Eight deliberate alterations to the actual complete capsule were refused,
including a changed vector, role, fitted coefficient, probability, unassessed
original verdict, denominator, missing native case and hidden local path. The
[tamper receipt](evaluation-panels/audio-v9-development-tamper-review.json) is
reproducible with the same command's `tamper` stage and a new output path. These
checks verify implementation and evidence integrity, not predictive accuracy.

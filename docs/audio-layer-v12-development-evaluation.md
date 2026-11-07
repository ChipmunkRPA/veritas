# Fixed speech layers: reject both response mappings

6 November 2026. **Neither arm qualifies for fresh confirmation or replacement.**
Earlier/middle/later frozen representations slightly improve the matched final
representation control. Both still lose too many useful AI conclusions against
the current service, especially with overlapping speech. Keep the serving model,
its calibration and rollback unchanged. No reserved confirmation was scored.

## Method and complete outcome

The [protocol](audio-layer-v12-protocol.md) and
[prospective execution freeze](audio-layer-v12-execution.md) preceded project
forwards and fitting. All 64 frozen implementation/protocol files remain
unchanged. Compare unprojected final 1,024 coordinates with the concatenated
masked means of hidden states 8/16/24 (3,072 coordinates). Both use fitting-only
normalization at the same variance budget, weighted objective, regularization,
independent calibration, dual stability checks and sixteen adoption gates.
Unlike the earlier v11 experiment, neither arm removes the head direction.

All 4,248 consumed inputs completed: 4,186 assessed, 62 unassessed, zero failures.
Roles remain 1,926 fitting, 873 calibration and 1,449 development inputs from
214/97/161 originals in 169/82/106 connected groups. Nine acoustic variants of a
source are correlated observations. Known genuine/cloned speaker connections
remain grouped; unknown speaker and pretraining overlap remain unverified.
This is reused development, not fresh accuracy or outside human validation.

| Measurement | Current service | Final control | Three layers |
| --- | ---: | ---: | ---: |
| Correct AI conclusions / 756 AI inputs | 648 | 524 | 526 |
| Correct original conclusions / 693 genuine inputs | 356 | 389 | 394 |
| Genuine inputs incorrectly called AI | 58 | 2 | 2 |
| AI inputs incorrectly called original | 0 | 0 | 0 |
| AI without an AI conclusion, including abstentions | 108 | 232 | 230 |
| Inconclusive / 1,449 planned inputs | 387 | 534 | 527 |
| Total correct directional conclusions | 1,004 | 913 | 920 |
| Inconclusive rate | 26.71% | 36.85% | 36.37% |
| Balanced Brier error, lower is better | 0.094859 | 0.070791 | 0.066568 |
| Balanced log loss, lower is better | 0.317112 | 0.238027 | 0.223569 |

Three layers gain seven correct conclusions over their matched control, with
unchanged false calls. Against service, false AI falls 58 to 2 but correct AI
falls 648 to 526 and inconclusive rises 387 to 527. A lower probability error
and fewer false alarms do not establish better useful detection. Both pass
twelve gates and fail the same four: per-generator AI retention, strong-condition
usefulness, background-speech usefulness and background-speech class retention.
Clean retention passes. No acceptance rule was changed after results.

Three-layer Brier difference versus service is -0.028291, connected-group
bootstrap 95% interval [-0.037055, -0.019901]; log-loss difference is -0.093543,
interval [-0.119255, -0.068293]. These intervals condition on consumed development,
fixed fitting/calibration and selected backgrounds, excluding fitting and
background-population uncertainty. They are not customer confidence intervals.

## Background interference

Each row has the same 161 originals. Values are correct directional conclusions,
with inconclusive counts in parentheses. False calls account for the remainder.

| Condition | Current service | Final control | Three layers |
| --- | ---: | ---: | ---: |
| Clean | 154 (7) | 154 (7) | 154 (7) |
| Noise, 0 dB | 97 (62) | 84 (77) | 85 (76) |
| Noise, 5 dB | 112 (49) | 103 (58) | 102 (59) |
| Noise, 15 dB | 139 (22) | 134 (27) | 135 (26) |
| Music, 5 dB | 95 (66) | 79 (82) | 80 (81) |
| Music, 15 dB | 141 (20) | 130 (31) | 132 (29) |
| Background speech, 0 dB | 75 (75) | 52 (109) | 53 (108) |
| Background speech, 5 dB | 93 (50) | 73 (87) | 74 (86) |
| Background speech, 15 dB | 98 (36) | 104 (56) | 105 (55) |

Across background-speech conditions, three layers reduce false AI 56 to 2,
but correct AI falls 234 to 174 and total correct calls 266 to 232. Both wrong
calls remain genuine overlapping-speech mixtures. All clean conclusions match
service. Neither background tags nor waveform shape establishes foreground
authorship. Internal research decision points are not public threshold claims.

## Verification and publication scope

Every source forward reproduced the original raw score, final-vector bytes,
activity/window result and status. All 508 native replays passed: the union of
99 predeclared cells and every one of the 422 computed filtered probes. These
checks validate extraction/provenance; they do not supply independent accuracy.

Separate numerical review rebuilt fitting-only normalization, linear probability
arithmetic, calibration ranks, both stability decisions, operational outcomes
and summaries. Exact fitting-only refits reproduced both complete fitted objects.
The declared statistical gate/bootstrap implementation remains shared. Public-only
replay also passed with empty model/data caches, including both exact refits.
Nineteen focused codec/arithmetic tests passed. Thirteen deliberate alterations
of the actual public packet were refused after the unchanged packet passed.

The [published aggregate](audio-layer-v12-development-evaluation.json) contains
complete condition/generator-group results, uncertainty and all adoption gates.
It excludes file-level outcomes, voice features and fitted parameters. The full
lossless reproduction package is prepared and verified locally; its public
disclosure awaits the separate user decision. Public aggregate access alone
does not reproduce neural outputs or permit the exact numerical refit.

## Next decision

Reject both arms without scoring reserved confirmation. Test the separately
frozen [noise-aware backbone](audio-wavlm-v13-execution.md) with its own fitting
and calibration under the original quality requirements. No model change,
confidence admission or outside human validation follows from these checks.

# Reject both separated-voice audio variants

7 November 2026. Neither predeclared arm qualifies for fresh confirmation or
replacement. Separation alone loses almost all useful conclusions. Adding its
response to the original mixture improves probability loss and lowers false
alarms, but substantially reduces correct AI and genuine conclusions, including
clean recordings. Keep the service unchanged; do not tune the frozen points,
requirements or condition routing to rescue these results.

## Complete comparison

The [full extraction repeat](audio-v14-full-replication-result.md) completed all
4,248 inputs. Both classifiers use only the original1,926 fitting inputs and
their own separate873 calibration inputs. All1,449 development inputs completed:
1,420 assessed,29 unassessed, zero failures, representing161 originals in106
connected groups. Nine acoustic variants per source are correlated. This is
reused development, not a new-source validation or whole-service accuracy.

The [original protocol](audio-separation-v14-protocol.md) learns a shared
source response from two unordered estimates using a multiple-instance
any-source objective. It never labels both estimates AI merely because their
mixture contains AI, and never uses source identities or clean reference
waveforms as inference predictors. The second arm also uses the original
mixture response. Both keep the original applicability and stability checks.

| Measurement | Current service | Separated sources | Combined response |
| --- | ---: | ---: | ---: |
| Correct AI conclusions /756 AI inputs | 648 | 10 | 500 |
| Correct original conclusions /693 genuine inputs | 356 | 1 | 45 |
| Genuine incorrectly called AI | 58 | 0 | 7 |
| AI incorrectly called original | 0 | 0 | 0 |
| AI without an AI conclusion, including abstentions | 108 | 746 | 256 |
| Inconclusive /1,449 inputs | 387 | 1,438 | 897 |
| Total correct directional conclusions | 1,004 | 11 | 545 |
| Inconclusive rate | 26.71% | 99.24% | 61.90% |
| Balanced Brier error, lower is better | 0.094859 | 0.209252 | 0.081010 |
| Balanced log loss, lower is better | 0.317112 | 0.610109 | 0.268913 |

The standalone arm passes7/16 requirements; it fails probability-loss, interval,
strong-condition loss and five usefulness/retention requirements. Combined
response passes11/16 but fails clean retention, per-generator AI retention,
strong-condition usefulness, background-speech usefulness and background-speech
class retention. Its fewer false flags largely accompany fewer useful calls;
this is not an improved detector merely because it abstains more.

## Background interference

Each condition contains the same161 source variants. Entries are correct
directional conclusions, with inconclusive counts in parentheses; false calls
account for the remainder.

| Condition | Current service | Separated sources | Combined response |
| --- | ---: | ---: | ---: |
| Clean | 154 (7) | 3 (158) | 96 (65) |
| Noise,0 dB | 97 (62) | 0 (161) | 61 (98) |
| Noise,5 dB | 112 (49) | 0 (161) | 62 (98) |
| Noise,15 dB | 139 (22) | 1 (160) | 75 (86) |
| Music,5 dB | 95 (66) | 1 (160) | 55 (106) |
| Music,15 dB | 141 (20) | 0 (161) | 75 (86) |
| Background speech,0 dB | 75 (75) | 1 (160) | 23 (137) |
| Background speech,5 dB | 93 (50) | 1 (160) | 38 (121) |
| Background speech,15 dB | 98 (36) | 4 (157) | 60 (100) |

For all background-speech conditions together, combined response lowers false
AI56 to4 but correct AI234 to106 and correct original32 to15. Clean useful
calls also fall154 to96. Do not infer foreground authorship from a separated
estimate, waveform shape, sound category or reduced false-alarm count.

Combined Brier difference versus service is-0.013849, connected-group bootstrap
95% interval[-0.018683,-0.009385]; log-loss difference is-0.048199,
interval[-0.065207,-0.032274]. These condition on fixed fitting/calibration and
selected consumed sources/backgrounds; they exclude fitting and population
uncertainty and are not customer confidence intervals. The public
[complete aggregate](audio-separation-v14-development-evaluation.json)
corrects an inherited helper's stale phrase "Fresh source comparison" to
consumed-development scope. Numeric results, rules, original internal receipt
and every gate remain unchanged.

## Verification and remaining work

[Separate arithmetic reconstruction](evaluation-panels/audio-v14-separation-numerical-review.json)
passes for all1,449 outcomes, including
normalization, source/bag responses, calibration ranks, stability decisions,
counts and both exact fitting-only refits. Declared statistical/gate functions
and the refit learner remain shared; this is not outside human review.
The [completed native replay](evaluation-panels/audio-v14-separation-native-review.json)
passes all170 unique cases: the union of all99 predeclared cells and every one
of the74 computed filtered probes. Original activity/raw/vector responses,
complete estimated-source features, filtered responses and upstream-graph
waveform hashes reproduce exactly. Every public case's source/PCM commitment
and predetermined/probe membership were also checked against the complete
journals. This verifies the measured computation, not detector superiority.

Do not score the1,080 reserved confirmation variants for these rejected arms.
Future audio work must improve discrimination under interference rather than
trade most useful calls for abstention. Separator license/sample-rate
ambiguities and unknown training/speaker overlap remain explicit in the
original protocol. No new recordings, voice-feature values or head coefficients
are published here. Model settings, customer evidence, overall confidence,
prepaid billing and GCP remain unchanged. Outside human reviewer count is zero.

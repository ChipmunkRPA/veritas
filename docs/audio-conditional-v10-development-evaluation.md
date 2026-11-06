# Conditional audio comparison: lower false alarms, poorer useful coverage

6 November 2026. **Both eligible candidates failed development and remain disabled.**
The nonlinear arm and its separate added-quality arm each failed the same five
of sixteen unchanged requirements. Neither advances to the reserved confirmation
panel. The current audio service and overall confidence are unchanged.

This experiment separates a response-mapping change from two added acoustic
quality descriptors. Better aggregate probability scores did not produce a
passing replacement: both candidates gave substantially fewer correct conclusions.

## Data and frozen method

The [prospective protocol](audio-conditional-v10-protocol.md) and
[execution freeze](audio-conditional-v10-execution.md) were published as `aea0558`
before project fitting or new neural forwards. No model, point or acceptance
rule was changed after seeing these outcomes.

All 4,248 consumed four-second inputs use the corrected v9 source roles:
214 fitting, 97 calibration and 161 development originals, in 169/82/106
canonical connected groups. Nine variants of each original are correlated,
not nine independent people. Genuine and cloned versions of a known parent
speaker stay in the same role. Pretrained-data and unknown biometric overlap
remain unverified. These are reused development sources, not a fresh accuracy test.

All 4,248 original quality measurements and 402 cached filtered-probe measurements
passed. Speech assessment remained unchanged: 4,186 original inputs assessed,
62 unassessed. Development contains 1,449 inputs: 1,420 assessed, 29 unassessed,
zero failed. The 1,080 reserved confirmation inputs remain unchanged and unscored.

Three fixed depth-two, 200-tree response mappings used original binary labels,
source/class/family-balanced weights and separate group/class calibration:

- **Nonlinear:** existing raw response and sixteen frozen, fitting-only residual
  projection coordinates.
- **Quality:** the same inputs plus original-waveform spectral-flatness and
  frame-energy-floor descriptors. Filtered probes have their own measured values.
- **Quality only:** the two descriptors alone, a nuisance diagnostic that is
  never eligible for selection.

All original neural parameters/checkpoints remained frozen. Their existing
commercial-permission checks were retained. No new pretrained weights were
downloaded. The installed scikit-learn 1.9.0 learner has a verified BSD-3-Clause
license. Original class guards, raw/mapped stability checks and internal research
operating points were retained; they are not a public SaaS threshold promise.

## Complete development results

| Measurement | Current service | Nonlinear | Added quality | Quality only |
|---|---:|---:|---:|---:|
| Correct AI conclusions / 756 AI inputs | 648 | 544 | 538 | 1 |
| Correct original conclusions / 693 genuine inputs | 356 | 260 | 71 | 0 |
| Genuine inputs incorrectly called AI | 58 | 0 | 0 | 1 |
| AI inputs incorrectly called original | 0 | 1 | 0 | 0 |
| AI without an AI conclusion, including abstentions | 108 | 212 | 218 | 755 |
| Inconclusive / 1,449 planned inputs | 387 | 644 | 840 | 1,447 |
| Total correct directional conclusions | 1,004 | 804 | 609 | 1 |
| Inconclusive rate | 26.71% | 44.44% | 57.97% | 99.86% |
| Balanced Brier error, lower is better | 0.094859 | 0.059494 | 0.060222 | 0.251809 |
| Balanced log loss, lower is better | 0.317112 | 0.193695 | 0.194178 | 0.700183 |

Nonlinear Brier difference is -0.035365, with a source-group bootstrap 95%
interval [-0.049948, -0.021596]. Added-quality difference is -0.034636, interval
[-0.048855, -0.021361]. Intervals are conditional on these reused development
sources, fixed fitting/calibration and selected backgrounds. They exclude fitting
and background-population uncertainty and are not customer confidence intervals.

The nonlinear arm made 200 fewer correct conclusions; added quality made 395
fewer. The latter produced no false directional calls by withholding 840 results,
including 622 of 693 genuine inputs. That is not evidence of stronger detection.
Compared directly with nonlinear, added quality lost 195 correct conclusions
and slightly worsened both proper probability scores. The nuisance arm provides
no useful stand-alone detector in this comparison; this does not prove acoustic
quality can never be useful in another representation or population.

## Background interference

Each individual condition contains the same 161 originals. Values are total
correct conclusions, followed by inconclusive results in parentheses.

| Condition | Current service | Nonlinear | Added quality |
|---|---:|---:|---:|
| Clean | 154 (7) | 139 (22) | 107 (54) |
| Noise, 0 dB | 97 (62) | 85 (76) | 60 (101) |
| Noise, 5 dB | 112 (49) | 88 (72) | 58 (103) |
| Noise, 15 dB | 139 (22) | 126 (35) | 79 (82) |
| Music, 5 dB | 95 (66) | 59 (102) | 49 (112) |
| Music, 15 dB | 141 (20) | 111 (50) | 81 (80) |
| Background speech, 0 dB | 75 (75) | 40 (121) | 33 (128) |
| Background speech, 5 dB | 93 (50) | 64 (97) | 56 (105) |
| Background speech, 15 dB | 98 (36) | 92 (69) | 86 (75) |

Over all background-speech variants, correct conclusions fell 266 → 196/175.
False AI calls fell 56 → 0/0, but correct AI conclusions fell 234 → 166/165.
Both eligible arms failed clean retention, per-generator AI retention,
strong-condition usefulness, background-speech usefulness and background-speech
class retention. Each passed eleven of sixteen gates. No candidate is selected.
The [aggregate JSON](audio-conditional-v10-development-evaluation.json) and full
capsule retain every generator, condition, background stratum, uncertainty and gate.

## Numerical and native verification

Fitting-only CPU refit reproduced the original fitted objects exactly.
Exported numerical-tree prediction matched native learner prediction with zero
absolute error on all 4,186 assessed originals plus 330 computed probes, for
each of three arms. Fitting used no calibration/development labels; held input
vectors were used only to check prediction parity.

All 1,449 outcomes, independent calibration group ranks, original cached-service
verdicts, both stability checks, operational counts and statistical gates were
reconstructed. There were 330 required probes: 308 exact v9 reuses and 22 newly
computed probes. Every new probe, plus all 99 predetermined native development
cells, was re-run from source audio: 121 unique cases, zero mismatches. The extra
22 cases audit exhaustive probe provenance, not a new accuracy panel.

The [public numerical capsule](evaluation-panels/audio-v10-development-review.json.gz)
contains original small fitted JSON objects, every sanitized outcome/probe and
the complete summaries/audit receipts. It references the existing public v9
vectors and v10 quality packet rather than duplicating them. Source audio,
transcripts, customer uploads, raw provider identifiers, private paths and full
pretrained checkpoints are excluded. Dataset/derived-feature terms remain separate
from the repository's source-code license.

From the repository root, with the project's numerical dependencies installed:

```sh
PYTHONPATH=backend OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  backend/.venv/bin/python backend/scripts/publish_audio_conditional_results_v10.py review \
  docs/evaluation-panels/audio-v10-development-review.json.gz \
  /tmp/new-audio-v10-result-review.json --project . --refit
```

Use a new output path. The numerical review loads no source audio, private
manifest or pretrained model. It verifies original fit/head bytes exactly;
sanitized outcome/summary bytes differ from their original lineage artifacts.
It reconstructs arithmetic and checks recorded native receipts without claiming
to re-execute the native audio model. An outside tester must separately identify
what they reran; our local review is not outside validation.

Public-only fitting reproduction passed with a deliberately empty model-cache
location. All eight deliberate changes to the actual public capsule were refused,
including changed source grouping, fitted parameters, probabilities, cached probe
quality, complete denominators and native coverage, an unassessed input called
original, and an embedded private path. All 105 selected regression tests passed;
these verify implementation and audit contracts, not detection accuracy.

## Implication for the next experiment

The [fixed-probability ceiling](evaluation-panels/audio-v10-guard-only-bound.json)
shows that removing every guard/stability veto still cannot restore the required
AI retention for Chatterbox, ElevenLabs, NaturalSpeech2, PflowTTS and StyleTTS2
in either eligible arm. This is a post-result diagnostic, not a new selection rule
or authorization to weaken uncertainty checks.

Another guard-only adjustment is therefore unsupported. A next prospective
comparison should test a supervised, strongly regularized mapping of the full
frozen representation against the current sixteen-coordinate projection. Useful
low-variance information may have been discarded, but that remains a hypothesis,
not a demonstrated cause. Freeze the method before fitting; retain independent
calibration, all error/usefulness gates and unchanged reserved confirmation.
If that approach also fails, assess a commercially permitted alternative
representation or explicitly modeled source separation instead of repeatedly
relaxing cutoffs. Evidence from this short-window development cannot authorize
arbitrary-duration audio or whole-video confidence.

Image localization, video mouth manipulation, mixed writing and outside human
testing remain unfinished. GCP deployment stays deferred.

# Audio v7 fresh confirmation: more conclusions, more errors; rejected

**Do not enable v7.** All1080 prospectively frozen inputs completed. The new
decision layer made823 correct conclusions versus761 for the current guarded
service, but false AI calls rose35→38 and AI speech incorrectly called original
rose39→52. It failed four of sixteen executable gates. The current local
calibration, overall confidence and rollback remain unchanged; GCP is deferred.

This is a new local source comparison following the passing
[development result](audio-guard-v7-evaluation.md). The
[prospective rules](audio-guard-v7-confirmation.md), executable code and
[exact input capsule](audio-guard-v7-confirmation-execution.md) were frozen and
published before the first fresh classifier forward. No point, head, guard,
source, offset or failure was changed after seeing these outcomes.

## What was tested

The120 foreground recordings comprise60 known-unused genuine LibriSpeech
speakers,30 ElevenLabs voices/complete scripts and30 Chatterbox prompt-speaker
IDs/dialogues. Shared known script openings leave110 connected foreground
groups:60 human,20 ElevenLabs and30 Chatterbox. These are known metadata groups,
not proof of biometric identity disjointness or absence from pretrained data.

Each source supplies one exact four-second production-decoded window and nine
conditions: clean; noise0/5/15; music5/15; and overlapping genuine background
speech0/5/15 dB SNR. Backgrounds are eight unused noise recordings, four unused
music tracks and six other genuine speakers. Every condition has60 genuine and
60 synthetic foregrounds. All1080 inputs were assessed, with zero extraction
or probe failures. There are120 source files, not1080 independent recordings.

Both arms use the same unchanged commercially permitted speech backbone and
original-view speech gate. V7 keeps its own previously fitted probability map,
independent class-set guards and original-only stability rule. The comparator
calls the actual current `local-speech-v3.1` calibration. No training or
calibration uses these confirmation labels. Internal tendency points are fixed
research diagnostics, not a public confidence or accuracy claim.

## Complete operational results

“False AI” means genuine speech called AI; “false original” means synthetic
speech called original. Inconclusive cases remain in every denominator.
Each non-total row contains120 inputs,60 in each truth class.

| Condition | Correct: service → v7 | False AI | False original | Inconclusive |
|---|---:|---:|---:|---:|
| Clean | 115 → 115 | 0 → 0 | 0 → 1 | 5 → 4 |
| Noise0 dB | 78 → 84 | 3 → 3 | 14 → 18 | 25 → 15 |
| Noise5 dB | 86 → 86 | 3 → 3 | 13 → 16 | 18 → 15 |
| Noise15 dB | 100 → 102 | 0 → 2 | 7 → 8 | 13 → 8 |
| Music5 dB | 80 → 89 | 1 → 1 | 2 → 5 | 37 → 25 |
| Music15 dB | 99 → 106 | 1 → 1 | 3 → 3 | 17 → 10 |
| Background speech0 dB | 66 → 79 | 6 → 7 | 0 → 1 | 48 → 33 |
| Background speech5 dB | 66 → 79 | 11 → 9 | 0 → 0 | 43 → 32 |
| Background speech15 dB | 71 → 83 | 10 → 12 | 0 → 0 | 39 → 25 |
| **All1080** | **761 → 823** | **35 → 38** | **39 → 52** | **245 → 167** |

Across540 genuine inputs, false AI rates are6.48%→7.04%, with correct original
calls338→392. Across540 synthetic inputs, false-original rates are7.22%→9.63%,
and correct AI calls423→431. AI inputs **without an AI conclusion**, including
both false-original and inconclusive outcomes, are117→109 (21.67%→20.19%).
Overall inconclusive rates are22.69%→15.46%. These are correlated-panel rates,
not measured error probabilities for a customer's upload.

The62 extra correct conclusions come mostly from genuine speech (+54), with
eight additional correct AI calls. The78 fewer inconclusive outcomes also
include16 additional wrong conclusions. More definitive outputs alone do not
establish the comparable error rate required for replacement.

## Generator and background weaknesses

Each generator contributes270 correlated variants from30 foreground files.

| Generator | Correct AI: service → v7 | AI called original | Inconclusive AI | AI without AI conclusion |
|---|---:|---:|---:|---:|
| ElevenLabs | 173 → 181 | 36 → 45 | 61 → 44 | 97 → 89 |
| Chatterbox | 250 → 250 | 3 → 7 | 17 → 13 | 20 → 20 |

The largest false-original burden is ElevenLabs. Neither family satisfies the
fixed allowance of at most one additional false-original call. Balanced
two-class Brier/AUC values are deliberately null in the single-class generator
tables; their operational detection/error/inconclusive counts remain complete.

Strong noise still hides synthetic speech: at0 dB, current and candidate call
14/60 and18/60 synthetic inputs original. V7 gains six correct conclusions
there but adds four incorrect original conclusions. This is an important
failure of the desired noisy-speech behavior.

Overlapping genuine voices also remain difficult. Across the three speech
conditions, genuine+genuine mixtures are falsely called AI27/180→28/180 times.
The truth labels contain **no AI speech** in these controls. V7 makes more
correct genuine conclusions31→66, while AI detection in synthetic+genuine
mixtures improves172→175 of180. This does not remove the multi-speaker false
alarm problem or identify which talker caused a response.

## Probability scores and frozen decision

All1080 common assessed inputs contribute to proper scores even if a guard
withholds a conclusion. Sources receive equal weight, then the two truth
classes are balanced. Lower Brier and log loss are better.

| Measure | Current guarded service | v7 |
|---|---:|---:|
| Brier | 0.121088 | 0.113554 |
| Log loss | 0.397481 | 0.370780 |
| AUC | 0.929372 | 0.922689 |

The paired Brier difference is−0.007534, conditional95% interval
[−0.013460, −0.001392]; log-loss difference is−0.026701, interval
[−0.045983, −0.007070]. All2000 requested sorted, stratified connected-group
bootstrap draws are defined. Intervals condition on fixed fitting/calibration
and the selected background pool; they exclude fitting/calibration and
background-population uncertainty. Better proper scores coexist with worse
ranking and more wrong directional calls.

Twelve gates passed; four failed:

- Strong-condition Brier: equal-level background speech worsened
  0.101705→0.110477. Its difference+0.008772 has conditional95% interval
  [−0.004346,+0.021013]; do not claim statistically established worsening from
  this interval, but the required observed improvement was not present.
- Total false calls: +3 false AI and+13 false original exceed the≤1 allowances.
- Condition false calls: noise0/5 and music5 add4/3/3 false-original calls;
  noise15 and background speech15 each add two false AI calls.
- Per-generator false-original calls: ElevenLabs adds nine and Chatterbox four.

Overall proper-score gains, clean correct-call retention, generator AI-call
retention, strong-noise/music usefulness, background-speech usefulness/class
retention, identical assessment, complete inputs and zero execution/probe
failures passed. Passing those requirements does not override the four failures.

## Reproducibility and next work

Input replay checked138 source recordings, all1080 exact mixtures and
69,120,000 sample values, with18 sampled production decodes. The complete
numerical review reconstructs probabilities, guards, stability, operational
counts and frozen gates; maximum pretrained raw-head arithmetic error is
8.75e-8. All18 preselected source/class/condition native replays matched raw
responses and representations exactly. This samples neural forwards; it does
not rerun every forward. The selected regression suite passed137 tests with
one private-data test skipped in the isolated checkout; that unchanged saved
holdout fixture had passed separately in the main checkout.

The [complete result](audio-guard-v7-confirmation-evaluation.json),
[full numerical capsule](evaluation-panels/audio-v7-confirmation-review.json.gz)
and [portable replay](evaluation-panels/audio-v7-confirmation-result-replay.json)
retain every outcome. The capsule publishes representations, recipes, original
small head/projection parameters and all source/audit commitments, while
excluding source audio, full transcripts, private paths and pretrained
checkpoints. Local replay is not outside validation; no outside human reviewer
has participated. Publisher source/weight permissions and attribution records
remain required.

The capsule SHA-256 is
`adb266b57f2164d61619741a496c16b05a6574e805938e7384d7cd5b9c57af95`.
The final portable replay reproduces all1080 outcomes, twelve passing/four
failing gates and the eighteen-cell native-audit receipt. Five
[targeted tamper checks](evaluation-panels/audio-v7-confirmation-tamper-checks.json)
reject changed candidate probability, probe clock, truth label, missing outcome
and altered head. The final targeted reviewer/calibration suite passed23 tests,
including strict discrete-count/gate comparisons. Source and expanded-capsule
secret scans were clean. These are local integrity checks, not outside review
or an exhaustive security assessment.

Reject v7 under the unchanged prospective rules. These sources are now consumed
development material. Next work should address diverse synthetic voice fitting
and genuine multi-speaker controls, followed by a separate untouched
confirmation panel. Do not rescue this failed result with a point change or
present a post-hoc subgroup as a passing replacement. This comparison does not
validate whole recordings, locate an edited interval in long audio, establish
per-upload confidence or complete the image/video/writing improvement program.

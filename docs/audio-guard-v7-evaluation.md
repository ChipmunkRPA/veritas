# Source-separated audio decision layer, v7: development result

**Passed all twelve frozen development gates; not enabled.** On the complete
720-input consumed panel, the candidate made 563 correct conclusions versus
500 for the actual guarded service, with the same one false AI call and zero
false original calls. Inconclusive outcomes fell from 219 to 156. These results
permit a separately frozen fresh comparison; they do not establish deployment
readiness, general accuracy, or outside validation.

## Method and comparator

The [protocol](audio-guard-v7-development.md) and implementation were frozen
before new representation extraction. The unchanged commercially permitted
speech backbone supplies its 1024-dimensional head input. A small decision
layer fits only the original V3 fitting sources: 40 recordings, 240 variants,
28 connected groups, including eight known synthetic voice groups from one
generator family. Separate calibration uses 80 recordings, 480 variants and
75 groups (40 genuine-class and 35 synthetic-class groups). Development uses
all 120 previously consumed recordings, 720 variants and 60 connected groups.

Known voice, source-group, normalized known transcript and conservative DFADD
prompt connections remain together. Background hashes are disjoint between
roles. No development or calibration labels fit the projection or coefficients.
The candidate has its own balanced-reference probability mapping, independent
class-set guards and same-recording bandpass stability probe. A missing or
unstable probe can withhold an original tendency; it cannot create an AI call.

This compares against the **exact current guarded service**, including its
voice guard and original-only stability veto. Before inference, all 720 saved
baseline probabilities and outcomes replayed without a difference. The earlier
V6 refitted, unguarded comparator is not substituted. V4–V6 remain rejected
under their original rules; this is a new fit/calibration experiment.

No neural checkpoint, serving configuration, billing behavior or overall
confidence calculation changed. The .8/.2 directional points and alpha .1
are frozen internal research diagnostics, not public confidence claims.

## Complete operational results

Each condition contains 60 genuine and 60 synthetic inputs, with 115 assessed
and five original unassessed inputs. All 30 unassessed variants remain in the
720-input denominator and remain inconclusive. There were no new extraction
or probe failures. Conditions are correlated variants, not independent people.

| Condition | Correct: service → v7 | False AI: service → v7 | False original: service → v7 | Inconclusive: service → v7 |
|---|---:|---:|---:|---:|
| Clean | 114 → 114 | 0 → 0 | 0 → 0 | 6 → 6 |
| Noise, 0 dB SNR | 52 → 65 | 0 → 0 | 0 → 0 | 68 → 55 |
| Noise, 5 dB SNR | 69 → 83 | 0 → 0 | 0 → 0 | 51 → 37 |
| Noise, 15 dB SNR | 101 → 105 | 0 → 0 | 0 → 0 | 19 → 15 |
| Music, 5 dB SNR | 64 → 88 | 1 → 1 | 0 → 0 | 55 → 31 |
| Music, 15 dB SNR | 100 → 108 | 0 → 0 | 0 → 0 | 20 → 12 |
| **All 720 inputs** | **500 → 563** | **1 → 1** | **0 → 0** | **219 → 156** |

Across 360 synthetic inputs, the service recognized 266 and v7 recognized 277.
The remaining 94 and **83 synthetic inputs respectively were inconclusive**;
zero false-original calls must not be described as zero missed AI content.
Across 360 genuine inputs, correct original calls rose from 234 to 286, one
was falsely called AI in each arm, and inconclusive outcomes fell from 125 to
73. Overall inconclusive rates were 30.42% and 21.67% on this consumed panel.

## Probability scores and uncertainty

Proper scores use all 690 common assessed inputs, including cases where a
guard withholds a conclusion. Sources receive equal weight, with balanced
class weight. Lower Brier and log loss are better; they assess these reference
probabilities, not a customer's probability that a particular upload is AI.

| Measure | Current guarded service | v7 |
|---|---:|---:|
| Brier | 0.062751 | 0.048507 |
| Log loss | 0.223498 | 0.183699 |
| AUC | 0.987834 | 0.990665 |

The paired Brier difference is −0.014244, with conditional 95% interval
[−0.023332, −0.004635]; log-loss difference is −0.039798, with interval
[−0.066251, −0.012613]. All 2,000 requested connected-group bootstrap draws
were defined. These intervals condition on fixed fitting/calibration and the
reused background pool, excluding fitting/calibration uncertainty.

Not every metric improved in every condition: music at 5 dB had slightly
better Brier (0.056508 → 0.054430), but **worse log loss**
(0.213856 → 0.218601) and AUC (0.986388 → 0.983364). The frozen gates require
condition-specific Brier and operational usefulness, not improvement of every
secondary metric. No gate or operating point was relaxed after inspection.

The inherited bootstrap helper uses set iteration for strata. Before the
first v7 evaluation, the statistical runtime fixed `PYTHONHASHSEED=0` in
addition to seed 461109; its observed strata order is `joint`, then `libri`.
The public capsule records the numerical runtime. The helper's historical CV
wording is corrected only in public nonnumeric scope text; the original local
summary hash is preserved, with all numbers unchanged. V7 is a fixed
fit/calibration/development comparison, not cross-validation.

## Source and numerical verification

- New extraction returned 1,410 assessed representations across 1,440 total
  role inputs, preserving 30 development unassessed inputs. Every assessed
  representation reconstructed the original linear-head response.
- The development audit checked all 720 source/PCM bindings and 43,656,000
  original sample values, and regenerated all 297 exact filtered probe views.
  Six deterministic occupied-cell probe neural forwards replayed with zero
  raw-response or embedding difference. This is a sample of neural forwards,
  not a rerun of every neural calculation.
- The [public CPU reviewer](../backend/scripts/review_audio_guard_v7.py)
  reconstructed all 1,440 source roles, 720 outcomes, independent guard pools,
  297 stability decisions, complete counts and frozen gates. The declared
  fitter/projection refit had zero maximum coefficient difference. Fitting and
  bootstrap reuse the declared implementations; guard/decision arithmetic is
  reconstructed separately. Waveforms and neural forwards are not rerun by
  this CPU reviewer.
- Four targeted tamper checks rejected an omitted input, changed role/group,
  altered guard maximum and changed probe sample clock. They are local
  integrity checks, not an exhaustive security assessment or outside review.

The selected audio/rollback/public-explanation/panel suite completed 109 tests
with one private-data test skipped in the isolated checkout. That exact saved
holdout test passed separately in the main checkout with its existing private
fixture. The initial isolated rollback failure was caused by absent ignored
model/HF caches and local sequence settings; matching those explicit settings
resolved it without changing any assertion or serving code. All ten new Python
modules/tests passed lint and format checks.

The [complete numerical result](audio-guard-v7-evaluation.json),
[fitting capsule](evaluation-panels/audio-v7-fitting-review.json.gz),
[source audit](evaluation-panels/audio-v7-source-audit.json),
[CPU receipt](evaluation-panels/audio-v7-fitting-replay.json) and
[tamper receipt](evaluation-panels/audio-v7-tamper-checks.json) are published.
The capsule includes small fitted/fixed head and projection parameters, but
no source audio, full transcripts, private paths or pretrained checkpoints.
Existing source attribution and commercial-permission records remain required.

## Limits and next decision

These short clips are about 2–4 seconds long and contribute one complete
analysis window. This is not a whole-recording or mixed-speaker validation.
There are 120 original development files and 60 connected groups, not 720
independent sources. Fitting covers only eight known synthetic voice groups
from one generator; source, channel, duration and generator confounding remain
possible. Known metadata connections do not establish biometric identity
across corpora, complete transcript independence or unknown pretrained overlap.
No outside human reviewer has validated this result.

Next, freeze new source identities, generators, background recordings,
transformations and gates before fresh detector inference. Include strong
noise, music and overlapping genuine background speech. A failed fresh
comparison preserves the current service. Any later local integration requires
the candidate's own calibration artifacts, matching source-linked UI/PDF/JSON,
paid-flow and rollback checks. GCP remains deferred.

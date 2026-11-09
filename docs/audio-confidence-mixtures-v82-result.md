# Clean and background panel prepared for combined audio confidence

All **4,176 planned four-second cases** are prepared and verified through normal
app audio decoding. They come from 464 foreground recordings and 48 backgrounds,
with 1,044 cases in each of the four separate roles. **No detector or confidence
fitting has used this panel yet.** These results establish the inputs for a
comparison, not improved detection, false-alarm rates or confidence reliability.

| Condition | Cases |
| --- | ---: |
| Clean | 464 |
| Noise at 0, 5 and 15 dB | 1,392 |
| Music at 5 and 15 dB | 928 |
| Overlapping genuine speech at 0, 5 and 15 dB | 1,392 |
| Total | 4,176 |

There are 2,016 genuine-foreground variants and 2,160 AI-foreground variants.
The AI foregrounds cover three generator families; the
[complete source report](audio-complete-sources-v81-result.md) describes the
224 genuine, 80 cloned, 96 expressive and 64 matched stock recordings. Each
foreground retains its fixed first four seconds. No VAD, detector score,
transcript alignment or favorable-case selection chose an interval.

Each role retains its own four speech, four noise and four music backgrounds.
Every background of each kind is used by 14 genuine and 15 AI foregrounds within
that role. This balances exposure between truth classes. The source/interval
allocation, complete source manifest, unchanged mixture helper and code were
published before any mixture was generated. All original source-role and
permission limits remain in force.

The final review preserves every planned source, condition, truth label and
role in its exact order; rehashes every variant; verifies all 64,000 encoded
samples against production PCM; and finds **zero exact cross-role byte or PCM
aliases**. It does not prove the absence of unknown semantic near-duplicates or
model pretraining overlap. Source-role assignments remain those of the 157
source-ancestry groups. Shared mixture backgrounds add dependence within each
role; these are not 157 independent groups of mixture outcomes.

Requested background SNR is measured after normal app encoding/decoding. Across
3,712 background cases, the maximum absolute deviation is 0.000323 dB. The
mixture helper applies foreground RMS normalization and one shared peak gain
across each foreground's nine variants. Accordingly, clean variants are
controlled normalized clips, and their common gain also depends on their chosen
background family. These are not 4,176 independent natural recordings.

All four role stages passed in 408.02 seconds total; the complete final review
passed in 3.96 seconds. Maximum sampled child-tree RSS was 582,270,976 bytes,
and minimum observed available RAM was 6,660,587,520 bytes. No resource guard,
sampling error or limit observation occurred. The observer records 38 verified
short-lived decoder exits. Sampled RSS is a lower bound, not a guaranteed peak.
The protected serving calibration is unchanged.

The [v83 training library](audio-score-fusion-v83-method.md) now implements two
candidate combinations: a small monotone global score fusion and two measured-
context specialists, each with its own separately sourced calibration. Ten
authored-data checks cover complementary specialists, monotonicity, role/group
separation, re-calibration refusal, skipped/failed checks, unsupported contexts
and prediction without truth/provider metadata. These checks test the training
code; they do not show performance on the prepared panel. All 31 focused
source, container, mixture and candidate-method tests and lint passed locally.

Next bind the exact neural extraction, models/runtime, candidate methods and
evaluation procedure before the panel's first detector call. Compare existing
checks, retained combinations and the two new score methods on common inputs.
Keep fitting, calibration, selection and final confirmation separate. Report
recovered/lost detections, false AI, false original, inconclusive/unassessed
cases, calibration quality and runtime together. A modest extra error remains
a tradeoff to assess, not automatic rejection. No final confirmation result
may tune a method or select a favorable subset.

Receipts: [fixed protocol](audio-confidence-mixtures-v82-protocol.md),
[published freeze](evaluation-panels/audio-confidence-mixtures-v82-freeze.json),
[complete counts and hashes](evaluation-panels/audio-confidence-mixtures-v82-result.json)
and [resource/SNR summary](evaluation-panels/audio-confidence-mixtures-v82-resources.json).
New audio, transcripts, source identities/pairings, individual fingerprints,
voice features and learned artifacts remain private on the development machine
and Seagate drive. No serving default, payment, workflow, outreach or GCP change.

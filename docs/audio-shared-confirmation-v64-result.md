# Shared audio comparison: useful coverage gain, original-speech risk

The two-head combination adds **91 correct conclusions** over the current check
on the complete fixed panel: five additional AI detections and 86 additional
genuine-speech conclusions, with no lost correct case or additional false AI
alarm. It also adds **28 AI recordings/conditions incorrectly leaning original**.
This is a meaningful coverage tradeoff to calibrate, not a reason for an
automatic zero-error rejection or a blanket claim of superior AI detection.
No serving model, protected calibration or customer confidence changed.

The original scientific settings and all cases were published before their
first reserved-input calls. The first run stopped on a monitor failure after
33 complete records, with one additional case possibly in flight. This report
uses the separately frozen complete technical repetition with the identical
scoring function, weights, calibrations and combinations. All 33 completed old
records reproduced exactly, excluding elapsed runtime. The original failed
attempt remains failed; this repetition is not a second fresh cohort.

## Complete outcomes

All 1,080 four-second cases remain: 540 AI and 540 genuine variants from 120
foreground recordings in 118 connected groups, with nine acoustic conditions.
There are 1,044 assessed and 36 unassessed cases, zero failed cases. The variants
share foregrounds and backgrounds; they are not 1,080 independent people.
“Without direction” includes assessed inconclusive and unassessed cases. “AI
not found” includes every AI case without an AI conclusion, including a wrong
original tendency and unknown results. These are screening tendencies, not
certified originality or fraud/identity findings.

| Method | AI found /540 | Genuine correct /540 | False AI /540 | False original /540 | Without direction /1080 | AI not found /540 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 385 | 319 | 31 | 12 | 333 | 155 |
| New features only | 2 | 33 | 0 | 14 | 1031 | 538 |
| Features + current score | 273 | 378 | 3 | 40 | 386 | 267 |
| Pair: abstain on conflict | 390 | 405 | 31 | 40 | 214 | 150 |
| Pair: current first | 390 | 405 | 31 | 40 | 214 | 150 |
| Pair: complement first | 390 | 405 | 31 | 40 | 214 | 150 |
| Pair: any AI | 390 | 405 | 31 | 40 | 214 | 150 |
| Pair: agreement only | 268 | 292 | 3 | 12 | 505 | 272 |
| Three heads: abstain on conflict | 383 | 406 | 31 | 41 | 219 | 157 |
| Three heads: any AI | 390 | 406 | 31 | 41 | 212 | 150 |
| Older v7 calibrated head | 394 | 377 | 34 | 41 | 234 | 146 |

The four pair availability/priority methods coincide here because the current
and complement heads have no directly conflicting directional calls. This does
not guarantee agreement on another source panel. Agreement-only discards many
correct cases. The feature-only head is weak alone; its shared backbone remains
useful through the complement. These are **four heads using two backbones**,
not four independent neural votes.

## What the gains and errors mean

For the pair, correct conclusions rise 704 to 795 (65.2% to 73.6% of all cases).
AI found rises 385 to 390 (71.3% to 72.2% of AI cases); genuine correct rises
319 to 405 (59.1% to 75.0%). False AI stays 31 (5.7% of genuine cases) across
the same 19 genuine groups. False original rises 12 to 40 (2.2% to 7.4% of AI
cases). Without direction falls 333 to 214 (30.8% to 19.8% of all cases).
Most coverage recovery is recognizing genuine speech, not recovering missed AI.
The 40 wrong original variants span 18 AI foreground sources, versus six sources
for the current check's 12. The 28 newly wrong variants involve 17 sources;
several already had a wrong call in another condition. These are correlated
variant counts, not 28 independent recordings.

The added-direction decomposition is five correct new AI calls with zero new
false AI, and 86 correct new original calls with 28 wrong new original calls.
This supports testing direction-specific calibration/routing on separately
assigned sources. Such a new rule would be a new method, not already validated
by selecting it after seeing this panel.

Of the 28 extra wrong original calls, 25 occur in the ElevenLabs source stratum,
two in Kokoro and one in Chatterbox. ElevenLabs AI found changes only 100 to
101 of 216, while wrong original tendencies rise 11 to 36. Chatterbox AI found
changes 198 to 201 of 216; Kokoro 87 to 88 of 108. These are source-provenance
strata used only for evaluation, never provider labels supplied to inference.

Three-head sensitive union adds one more correct genuine case and one more
wrong original case than the pair, with no extra AI detection. Three-head
conflict-abstention loses seven previously correct AI cases and removes no
false AI on this panel. The older v7 head finds nine more AI cases in aggregate
than current, but introduces five false AI calls, removes two and adds 29 wrong
original calls; it adds 80 correct cases and loses 13. Preserve these distinct
tradeoffs instead of declaring every extra model beneficial.

## Background conditions

Each row below contains 60 AI and 60 genuine variants. Entries show current
then pair. Full per-method condition and generator counts remain in the JSON.

| Condition | AI found | Genuine correct | False AI | False original | Without direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 53 → 53 | 57 → 57 | 0 → 0 | 0 → 1 | 10 → 9 |
| music_15 | 38 → 38 | 56 → 59 | 1 → 1 | 2 → 11 | 23 → 11 |
| music_5 | 32 → 33 | 34 → 49 | 2 → 2 | 3 → 10 | 49 → 26 |
| noise_0 | 31 → 33 | 28 → 40 | 1 → 1 | 2 → 5 | 58 → 41 |
| noise_15 | 45 → 45 | 57 → 59 | 0 → 0 | 2 → 5 | 16 → 11 |
| noise_5 | 36 → 38 | 43 → 49 | 0 → 0 | 2 → 5 | 39 → 28 |
| speech_0 | 44 → 44 | 15 → 33 | 8 → 8 | 1 → 3 | 52 → 32 |
| speech_15 | 58 → 58 | 15 → 28 | 11 → 11 | 0 → 0 | 36 → 23 |
| speech_5 | 48 → 48 | 14 → 31 | 8 → 8 | 0 → 0 | 50 → 33 |

At equal-level noise the pair recovers two AI and 12 genuine cases while
adding three wrong original tendencies. Music conditions account for 16 of the
28 extra wrong original calls. Overlapping speech retains 27 false AI calls
across its three conditions; combining these heads does not solve that issue
or attribute synthesis to one speaker in a long recording.

## Decision and calibration work

Advance the pair as a useful coverage candidate for separately sourced combined-
confidence calibration, explicitly testing AI and original directions and
expressive speech with music/noise. Retain current and v7 as comparators and
rollback; keep the stricter agreement method as a lower-coverage comparator.
The feature-only/third-head additions are lower priority as standalone or extra
voting checks on this evidence. Do not delete their shared assets or rewrite
the earlier experiments as passes.

There is no new combined probability in this study. Each head retains its own
frozen calibration; agreeing or averaging heads cannot establish a combined
confidence. The pair’s exploratory foreground-group bootstrap interval for net
correct-count change is +67 to +116 under 2,000 resamples. Shared background
dependence, historic voice reuse, three AI generators, short English clips and
unknown detector pretraining overlap limit transfer. The older v7 calibration
has two known voice overlaps with this panel. These are not customer accuracy
or confidence-interval guarantees, and fresh image/video/article testing remains
separate work.

## Verification and reproducibility

All 14 bounded repetition stages passed with no guard or sampling failure.
The corrected monitor recorded nine verified short-lived descendant exits.
Maximum sampled child-tree RSS was 4,907,958,272 bytes; this is a lower bound,
not a true peak. Total measured bounded-stage time was 1,482.16 seconds, not a
per-analysis speed comparison. All 86 frozen code bindings and source/fitted
artifact hashes reverify; the protected serving calibration is unchanged.

A separate set-algebra review reproduces every method, per-case assignment,
condition/generator total and newly gained/lost case. A second review reproduces
all bootstrap intervals using histogram weights and verifies the retained
33-case execution prefix, profiles and original failure. Two actual altered-
result probes reject a changed detection total and changed lost-case count.
All 33 focused audit/combination/monitor/bootstrap tests and lint pass locally.
These are local independent calculation reviews, not outside human review.

See the [complete aggregate result](evaluation-panels/audio-shared-confirmation-v64-result.json),
[set review](evaluation-panels/audio-shared-confirmation-v64-set-review.json),
[execution review](evaluation-panels/audio-shared-confirmation-v64-execution-review.json),
[adverse checks](evaluation-panels/audio-shared-confirmation-v64-adverse-review.json),
[original stop review](evaluation-panels/audio-shared-v62-stop-review-v64.json)
and [prospective repetition protocol](audio-shared-confirmation-v64-protocol.md).
New raw audio, record pairings/IDs, voice features and fitted weights remain
private. No customer file, billing, workflow, outreach or GCP change occurred.
GCP remains last.

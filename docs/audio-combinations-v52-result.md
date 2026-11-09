# Audio candidates recover complementary cases

The complete ten-method comparison supports retaining several candidates. On
the consumed development panel, the two-head combination adds **110 correct
conclusions with one additional false AI call** and loses no previously correct
conclusion. The sensitive three-head combination adds 112 with the same one
additional false AI call. A three-head conflict-abstention alternative makes
fewer false AI calls overall while losing four previously correct AI cases.
These are useful tradeoffs to confirm, without an automatic zero-loss or
one-error rejection. No serving calibration or default changed.

This compares **three heads using two backbones**, not three independent neural
votes. The feature-only and complement heads share the frozen WavLM features;
the complement also uses the current speech score. Existing calibrations and
scope/stability checks are unchanged. A new combined confidence was not fitted.

## All methods and complete denominators

There are 1,449 correlated windows from 161 foreground sources in 106 connected
groups: 756 AI and 693 genuine inputs, with nine acoustic conditions. All 29
unassessed windows remain. “Without direction” includes those 29 plus assessed
inconclusive outcomes. “AI not found” includes inconclusive and unassessed AI,
not just AI incorrectly called original.

| Method | AI found /756 | Genuine correct /693 | False AI /693 | False original /756 | Without direction /1449 | AI not found /756 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current | 648 | 356 | 58 | 0 | 387 | 108 |
| New features only | 30 | 28 | 0 | 4 | 1,387 | 726 |
| Features plus current score | 522 | 437 | 4 | 0 | 486 | 234 |
| Pair: abstain on conflict | **664** | **450** | **59** | **0** | **276** | **92** |
| Pair: current first | 664 | 450 | 59 | 0 | 276 | 92 |
| Pair: complement first | 664 | 450 | 59 | 0 | 276 | 92 |
| Pair: any AI tendency | 664 | 450 | 59 | 0 | 276 | 92 |
| Pair: agreement only | 506 | 343 | 3 | 0 | 597 | 250 |
| Three heads: abstain on conflict | 661 | 451 | 56 | 0 | 281 | 95 |
| Three heads: any AI tendency | **665** | **451** | **59** | **0** | **274** | **91** |

The four pair methods give identical outcomes here because the two component
heads have no conflicting directional calls on this panel. They can differ on
new data and remain separate methods. The pair recovers 16 AI and 94 genuine
cases; three-head sensitive union recovers 17 AI and 95 genuine cases. The
number of genuine foreground groups with any false flag stays 33 of 77 for
the pair and sensitive three-head method. These sources and backgrounds are
correlated; 1,449 is not a count of independent people.

The three-head conflict-abstention method adds 112 correct cases and loses
four, for a net gain of 108. It removes three old false AI calls and introduces
one, reducing false AI 58 to 56 and affected genuine groups 33 to 32. Four
previously detected AI cases become inconclusive because the feature-only head
gives a conflicting original tendency. This is a candidate tradeoff, not a
reason for automatic rejection or a claim of statistical superiority.

## Strong background conditions

At equal-level noise, the pair finds 65 AI cases versus 59 current and correctly
recognizes 48 genuine cases versus 38. False AI rises two to three, and cases
without direction fall 62 to 45. With music 5 dB below foreground speech,
correct AI rises 60 to 63, correct genuine 35 to 54, with zero false AI in both.

Overlapping voices remain a weakness. At equal-level background speech, the
pair preserves 67 AI detections and 11 false AI calls; genuine conclusions rise
eight to 23. Across the three overlapping-speech conditions, it still has 56
false AI calls. Combining the heads does not solve speaker attribution or
locate an edited interval in long recordings. All condition and generator
strata remain in the [complete result](evaluation-panels/audio-combinations-v52-result.json).

## Verification and next comparison

All three historical standalone operational totals reproduce exactly. A
[separate set-algebra reviewer](../backend/scripts/review_audio_combinations_v52.py)
reconstructs every method, condition/generator total and actual newly gained or
lost case without importing the first combination implementation. Its
[receipt](evaluation-panels/audio-combinations-v52-set-review.json) matches
exactly. Two actual altered-result checks reject a changed total and a changed
lost-case count. Twelve focused tests and lint passed; both bounded preparation
and evaluation profiles passed. The
[execution receipt](evaluation-panels/audio-combinations-v52-execution.json)
binds the remotely published freeze, result, reviewer and profiles. No neural
inference, refit or source download was performed.

Advance pair conflict-abstention, sensitive three-head union and three-head
conflict-abstention as named arms of a separately frozen fresh-source comparison,
alongside the current detector, v7 decision layer and standalone complement.
Include new genuine speakers, multiple generators and noise/music/overlapping
voice controls. Select on development; hold a final source cohort separate.
Inference rules must use measured inputs, never known generator/background
truth. A selected combination needs its own probability calibration before
entering overall confidence. The tradeoff evidence is sufficient to continue
these candidates; perfection is not an admission requirement.

These are post-result development measurements on already consumed sources,
not fresh accuracy or outside review. Historical failed protocols and all their
outcomes stay unchanged. No new source identities, voice representations or raw
audio are published. No model default, protected calibration, customer confidence,
workflow, outreach, payment or deployment changed. GCP remains last.

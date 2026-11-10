# Combined audio confidence: completed held-role comparison

Both published selection procedures chose the global combination before final
forwards. That choice stayed fixed. The [primary result](evaluation-panels/audio-combined-confidence-v84-result.json)
and [parallel fourteen-method result](evaluation-panels/audio-common-reference-v88-result.json)
now report the same completed final cohort. They are not two independent tests.
No serving default or protected production calibration has changed.

## Sources and execution

Final confirmation contains **1,044 variants of 116 foreground files**, with
540 AI and 504 genuine variants across 28 known connected ancestry groups.
Nine clean/mixture conditions share each foreground recording and background
sources. They are correlated observations, not 1,044 independent voices. Source
roles are separated under the published graph; unknown pretraining overlap and
identity links remain possible. This is controlled four-second English speech
from the declared generator families, not unrestricted real-world recordings.

After the original dispatcher stopped before launching the next chunk because
RAM admission expired, the owner freed memory and authorized continuation.
The [continuation receipt](evaluation-panels/audio-final-continuation-v90-result.json)
retains the completed prefix, original stop and exact published choices. Only
never-attempted chunks resumed. Neural limits, workers and learned maps did not
change; there was no failed-inference retry or overwritten result.

All twelve extraction stages passed, totaling 1,529.92 seconds of successful
worker time. Sampled RSS reached 4,781,047,808 bytes; minimum observed available
RAM was 2,453,618,688 bytes, above the unchanged running reserve. These are sampled
observations, not guaranteed peaks. Both numerical confirmation commands also
exited successfully. See the [resource record](evaluation-panels/audio-combined-confidence-v84-confirmation-resources.json).

The separate [numerical replay](evaluation-panels/audio-head-replay-v85-confirmation.json)
verified all 1,013 assessed feature packets, hashes and four-second sample
clocks, preserving 31 unassessed cases as missing. Raw-head probabilities
reconstructed within 8.55e-8; serving calibration and retained head outputs
reproduced exactly. This is an arithmetic/integrity check, not new inference,
truth adjudication, stability-decision reanalysis or outside human review.

## Decisions and coverage

| Method and fixed rule | AI found / 540 | Genuine correct / 504 | False AI | False original | Without direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current complete decision, including abstention | 366 | 263 | 19 | 17 | 379 |
| Retained pair with conflict abstention | 369 | 336 | 20 | 52 | 267 |
| Retained v7 complete decision | 373 | 339 | 33 | 42 | 257 |
| Selected global combination, tendency at 0.5 | 404 | 450 | 52 | 107 | 31 |
| Context-balanced combination, tendency at 0.5 | 403 | 450 | 52 | 108 | 31 |
| Shared-reference context combination, tendency at 0.5 | 403 | 450 | 52 | 108 | 31 |

“False original” means AI given an original tendency. “Without direction”
includes both assessed inconclusive and unassessed cases; it is not a correct
original verdict. All methods preserve the same 31 unassessed inputs. A tendency
at 0.5 is a research decision rule, not a public screening-confidence promise.

The pair adds **76 correct directions and loses none**, while adding one false
AI and 35 false-original directions. Thirty-four of those added false-original
calls are expressive-generator variants. Its three additional AI hits are a
small gain; most increased correct coverage is on genuine recordings.

Global adds **225 correct directions and loses none**, while adding 33 false AI
and 90 false-original directions. It turns all 348 assessed current inconclusive
cases into 225 correct and 123 wrong directions. This is a substantial coverage
change with a substantial error cost, not an apples-to-apples accuracy gain from
replacing one classifier while preserving its abstention policy.

## Strong noise, overlapping speech and generator differences

Each individual condition contains 60 AI and 56 genuine variants. Under strong
noise at 0 dB, current finds 21 AI, correctly clears five genuine variants,
makes three false AI and two false-original calls, and leaves 85 without a
direction. The pair produces 23, 18, four, five and 66 respectively. Global
produces 32, 47, nine, 19 and nine. Better coverage in noise does not establish
reliable recovery of every underlying voice.

Across the three overlapping-speech conditions, current falsely flags 14/168
genuine variants, the pair also 14/168, and global 37/168. These mixtures do not
identify which overlapping speaker is synthetic. All backgrounds in this study
are genuine; detection of AI-generated environmental audio remains untested.

For expressive AI speech, current finds 94/216 and makes 15 false-original
calls; the pair finds 94 and makes 49; global finds 110 and makes 77. Global's
remaining 29 AI cases without direction are also expressive. For cloned speech,
AI hits rise 149/180 to 163/180 with global, with false-original calls rising
zero to 17. For stock synthetic speech they rise 123/144 to 131/144, with
false-original calls rising two to 13. These are correlated controlled variants,
not independent generator-level sample sizes.

## Probability quality and matched score-only comparisons

The fixed balanced-reference squared probability loss, assigning neutral loss
only for missing scores during evaluation, is **0.119273 current versus 0.112917
global**; lower is better. On scored cases, weighted Brier is 0.116009 versus
0.109494, weighted log loss 0.373736 versus 0.360360, and weighted ten-bin
calibration error 0.072372 versus 0.051023. However, AUROC falls from 0.929565
to 0.906994. Better average probability error does not establish uniformly
better ranking, background behavior or correctness for an individual upload.

Retrospectively, the two context maps have lower balanced-reference losses:
0.111709 and 0.110792. Neither replaces the global arm selected before this
test. Selecting them now would consume the final cohort for selection and
require a new confirmation. The ineligible context-prior-only diagnostic has
AUROC 0.507360 and is not a customer detector.

For comparison at the **same fixed internal symmetric score point of 0.8**,
current's score-only rule gives 396 AI hits, 430 genuine correct, 42 false AI,
96 false original and 80 without direction. Global gives 377, 364, 20, 61 and
222. It reduces wrong directions while also leaving more correct cases without
a direction. The score-only current rule omits its complete validity/stability
abstention and must not be confused with the first table's current method.
The JSON retains every predeclared research point; none is a public threshold
claim or confidence guarantee.

## What to retain and test next

Keep the fixed global candidate, pair, context variants and current rollback
for explicit local serving comparisons. The new global map merits that work
because average probability error improves; it does not qualify a blanket
whole-file confidence or an automatic original verdict for skipped checks.
The pair's expressive-voice mistakes need direct attention. A modest added
error is not an automatic rejection, and increased coverage is not an automatic
promotion.

Serving extraction parity, four-second support enforcement, evidence/report
consistency and the paid-before-inference flow remain separate qualification
steps. Longer recordings, new languages/generators, unknown source relationships,
AI backgrounds and natural noisy recordings need new data. The new
[research review](recent-detection-research-2026-11-09.md) proposes complementary
follow-ups without retuning this frozen experiment.

Only aggregate counts and whole-artifact hashes are public. Raw media, source
identities and pairings, per-record fingerprints, features and learned maps
remain private. No outreach, Actions run, real payment or GCP deployment was
performed for this confirmation.

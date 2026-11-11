# Recorded background noise reduced synthetic speech detection

The frozen Turkish speech comparison completed all six conditions on the
Mac. The launch that followed the memory retry had 8.92 GiB available.
All six workers exited successfully under the 6 GiB admission rule.
They took 287.3 seconds in total and recorded 365 classifier forwards,
including stability checks. The largest sampled worker-tree RSS was
2.98 GiB. The source-clean replay matched all 48 previous dispositions,
including five source-preparation failures. [Frozen method](audio-demand-transfer-v208-method.md) · [Verified counts](evaluation-panels/audio-demand-transfer-complete-v208.json)

## Synthetic speech

All 24 synthetic clips passed the speech gate in every condition.

| Background | Leaning AI | Leaning original | Inconclusive |
| --- | ---: | ---: | ---: |
| Source clean | 21 | 2 | 1 |
| Gain-matched clean | 21 | 2 | 1 |
| Washroom, 0 dB | 15 | 0 | 9 |
| Washroom, +5 dB | 19 | 0 | 5 |
| River, 0 dB | 0 | 2 | 22 |
| River, +5 dB | 4 | 2 | 18 |

At river 0 dB, 20 of the 21 clips flagged in gain-matched clean audio
became inconclusive and one moved to leaning original. At +5 dB, 16
became inconclusive, one moved to leaning original and four kept their AI
direction. At each river level one of the two clean wrong-original calls
persisted and the other became inconclusive.

A post-run arithmetic check reapplied the fixed mapping and initial
decision to the saved scores. At river 0 dB, 21 of the 22 inconclusive
synthetic cases were already inconclusive before the stability check.
One initially leaned original and was withheld by that check. At +5 dB,
the split was 17 before the stability check and one withheld after it.
These clips passed the gate. The larger loss occurred in the classifier
response and its calibrated decision.

## Genuine speech

Five of the 24 planned genuine sources were unavailable during preparation
and recur in every condition. Among the 19 prepared sources, zero leaned
AI in any condition.

| Background | Leaning original | Inconclusive | Speech gate rejected |
| --- | ---: | ---: | ---: |
| Source clean | 19 | 0 | 0 |
| Gain-matched clean | 19 | 0 | 0 |
| Washroom, 0 dB | 18 | 0 | 1 |
| Washroom, +5 dB | 19 | 0 | 0 |
| River, 0 dB | 10 | 8 | 1 |
| River, +5 dB | 15 | 3 | 1 |

## Scope and next comparison

The 288 planned cases reuse 43 prepared foregrounds and two recorded
backgrounds. The synthetic speech came from one generator collection.
The genuine and synthetic clips also use different offsets in each
background tape. The strongest comparison is each clip against its own
gain-matched clean audio. The SNR measures the full clip, including pauses.
These conditions cover two specific recordings of environmental noise.
[Preparation and audio attribution](audio-demand-controls-v204-method.md)

A follow-up should measure whether candidate signal conditioning or an
alternate voice check recovers the missed AI flags. Reuse each exact
background excerpt across both classes, add independent tapes and multiple
voice generators, and keep contributors and voices separate across fitting
and evaluation. Measure gained AI flags, genuine false alarms, wrong-original
directions and availability on the same clips. The v208 study used the
existing calibration throughout.

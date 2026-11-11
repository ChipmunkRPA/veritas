# Short-speech audio transfer under recorded background noise

The local replay assessed 195 of 240 planned three-second Turkish speech variants. Each of the five conditions assessed the same 11 genuine windows and 28 synthetic windows from two source generators. The 45 unavailable variants came from preparation failures. Five serial runs verified one native feature extraction per assessed variant. The candidate used the head fitted on four-second English speech. [Method](audio-paired-transfer-v230-method.md) · [Verified counts and resource commitments](evaluation-panels/audio-paired-transfer-complete-v230.json)

Rank AUC is the fraction of synthetic–genuine comparisons ordered correctly, with half credit for ties. Within each condition the two methods were scored on the same windows.

| Condition | Current rank AUC | Candidate rank AUC | Current synthetic AI flags | Candidate synthetic AI flags | Current synthetic wrong-original | Candidate synthetic wrong-original |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched clean | 1.000 | 1.000 | 27 | 28 | 0 | 0 |
| Kitchen, 0 dB | 0.984 | 0.909 | 15 | 16 | 4 | 7 |
| Kitchen, +5 dB | 0.990 | 0.922 | 17 | 18 | 4 | 6 |
| Traffic, 0 dB | 0.831 | 0.877 | 7 | 9 | 4 | 5 |
| Traffic, +5 dB | 0.938 | 0.929 | 13 | 16 | 5 | 5 |

The current directions are the recorded v219 results. Candidate directions use its frozen English diagnostic score points. At matched English diagnostic points, the current score flagged 53 of 112 noisy synthetic variants and gave 18 wrong-original calls. The candidate flagged 59 and gave 23 wrong-original calls. Both methods gave zero false AI calls across the 44 assessed noisy genuine variants. The aggregate also retains clean-to-noise score drift and matched-utterance ranking.

The candidate improved ranking for both source generators at 0 dB traffic. The kitchen rank loss came from one generator, while the other maintained perfect separation there. These results support testing noise-specific routing before using the candidate in short-window scoring. The English score points are experimental and have no Turkish calibration. The panel was already used for baseline evaluation, and its variants share voices and two noise recordings. Production defaults and customer confidence were left unchanged. [English comparison](audio-paired-consistency-v223-result.md)

## Replay with the candidate as a second opinion

An offline replay reproduced the full aggregate, then applied the candidate only to assessed variants that the recorded route called inconclusive. A candidate AI direction changed those cases to AI at the same frozen English diagnostic point. That recovered seven synthetic variants from seven distinct foregrounds. AI flags rose from 52 to 59 out of 112 assessed noisy synthetic variants. The 17 wrong-original calls remained, and there were zero additional false AI calls across 44 assessed noisy genuine variants.

| Condition | Additional synthetic AI flags | Additional genuine false AI |
| --- | ---: | ---: |
| Kitchen, 0 dB | 1 | 0 |
| Kitchen, +5 dB | 1 | 0 |
| Traffic, 0 dB | 2 | 0 |
| Traffic, +5 dB | 3 | 0 |

The follow-up used the same saved rows and made zero model calls. Recoveries appeared in both backgrounds, so a fresh comparison can start with the second-opinion rule and measure whether noise-specific routing adds value. Separate speakers, generators and background recordings are needed to estimate its error rates.

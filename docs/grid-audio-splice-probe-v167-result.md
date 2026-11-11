# Ordinary speech-copy diagnostic result

All 64 planned GRID cases completed locally. The 32 original and hard-copy
cases reproduced the stored app availability, window and direction. All 26
scored raw responses matched exactly. Each condition contains 16 genuine
clips drawn from eight connected recording groups.
[Method](grid-audio-splice-probe-v167-method.md) · [Actual result](evaluation-panels/grid-audio-splice-probe-complete-v167.json)

| Condition | False AI tendencies | False AI tendencies with symmetric stability | Inconclusive with current rule | Unavailable | Exact copy witness |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untouched | 0/16 | 0/16 | 3 | 4 | 0/16 |
| Hard copy | 12/16 | 12/16 | 2 | 2 | 16/16 |
| 20 ms crossfades | 13/16 | 11/16 | 1 | 2 | 16/16 |
| 50 ms crossfades | 13/16 | 12/16 | 1 | 2 | 16/16 |

Both crossfades restored the original immediate sample jumps at both copy
boundaries in every case. False AI tendencies persisted. The median raw
score remained close to 0.999 in all edited arms. The filter-based
stability check left every hard-copy false AI tendency in place. These
observations rule out an immediate sample jump as a necessary condition
for this failure on these clips. The shifted speech and surrounding
context remain possible explanations.

The separate exact-sample cue located the surviving duplicated portion of
the audio in all 48 copy cases and found no repeat in the 16 originals.
It records paired source sample ranges. Its response supports inspection
of repetition and has no AI-or-original direction. Its coverage after
lossy recompression and on fresh recordings is still unmeasured.

The worker made 108 voice forwards in 67.72 seconds. Maximum sampled
descendant RSS was 2,921,234,432 bytes. Genuine-to-genuine and matched
genuine-to-AI splices from fresh recording groups are needed to measure
false alarms and missed AI together. The study does not change serving.

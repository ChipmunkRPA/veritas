# Overlap abstention suppresses both true and false AI flags

The guarded CPU comparison completed all 96 matched genuine and
synthetic inputs. The voice route ran 94 raw and 14 stability forwards.
The competing-speech check ran 96 forwards. The worker completed in
72.01 seconds with 2.84 GiB maximum sampled process-tree RSS and no
guard or sampling errors.
[Frozen method](audio-babble-overlap-comparison-v183-method.md) · [Verified aggregate](evaluation-panels/audio-babble-overlap-comparison-result-v183.json)

| Background | Genuine false AI flags, baseline → overlap abstention | Synthetic AI flags, baseline → overlap abstention |
| --- | ---: | ---: |
| Clean | 0/16 → 0/16 | 16/16 → 16/16 |
| Six competing speakers, 0 dB | 13/16 → 0/16 | 16/16 → 0/16 |
| Six competing speakers, +5 dB | 10/16 → 0/16 | 16/16 → 1/16 |

The rule withheld all 23 noisy false AI flags and 31 of 32 correct
noisy AI flags. At 0 dB it left all 32 results inconclusive. At +5 dB
it left 31 of 32 inconclusive. The clean results stayed unchanged,
with two genuine inputs rejected by the speech gate and all other
inputs receiving the correct direction. Neither arm called a known
synthetic input original in this panel.

The median paired raw-score gap between synthetic and genuine
foregrounds was 0.999 clean, 0.0024 at 0 dB and 0.0035 at +5 dB.
This collapse under identical added speech points to a discrimination
problem. Keep the overlap intervals as acoustic context, and reject
the tested abstention rule for origin decisions. Evaluate any next
origin check on matched genuine and synthetic foregrounds across
different overlapping voices and sources. The 16 pairs in this
study share four dependent components and one synthetic family.
Holdout scoring was not performed.
[Matched construction](audio-matched-babble-v181-result.md) · [Publisher-sample overlap context](audio-overlap-context-v180-result.md)

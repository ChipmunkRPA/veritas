# Training for paired noise stability improved the audio ranking

The fixed study fit all three heads on 324 assessed windows and 214
matched clean/noisy pairs. It calibrated on separate source groups and
noise recordings, then evaluated 326 assessed windows from the consumed
selection role. It reused recorded features throughout.
[Method](audio-paired-consistency-v223-method.md) · [Verified counts](evaluation-panels/audio-paired-consistency-complete-v223.json)

| Method | AI flags in both noisy conditions | Genuine false AI | Synthetic wrong-original | Noisy rank AUC |
| --- | ---: | ---: | ---: | ---: |
| Recorded current directions | 74 | 12 | 2 | 0.898 for its scores |
| Current score at matched calibration points | 76 | 13 | 2 | 0.898 |
| Raw-score fit | 76 | 13 | 2 | 0.898 |
| Feature combination without drift penalty | 80 | 16 | 9 | 0.930 |
| Feature combination with drift penalty | 84 | 14 | 5 | 0.946 |

At 0 dB the paired head's AUC was 0.926, compared with 0.857 for the
current score. At +5 dB the values were 0.971 and 0.935. The drift
penalty also improved both ranks and reduced both wrong directions
relative to the otherwise identical feature fit. On clean inputs it
flagged 56 synthetic clips, compared with 55 for the recorded current
route, with zero wrong-direction calls for either.

The gain over recorded directions came with two more false AI calls
and three more wrong-original calls. The false AI calls covered nine
connected groups with the paired head and seven with the current route.
The matched calibration points were set on separate backgrounds.
The realized error counts therefore differ in selection. The paired
head's directions are score-only diagnostics, with the current route's
additional stability check represented by the recorded comparison.

Advance the paired head to a native transfer test. These features came
from four-second English windows and the earlier selection role was
already used for model comparison. Three-second Turkish features need
exact native extraction and replay before evaluating the score there.

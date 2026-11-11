# Timed face-swap review in the full local app

All six analyses completed. Each scored the same 40 native face positions,
including 12 within the known 3–6 second intervention. Each appearance
check measured 12 windows. The final read-only replay verified 191 saved
evidence assets and all reported source timestamps. [Method](video-faceswapai-full-app-v190-method.md) · [Aggregate JSON](evaluation-panels/video-faceswapai-full-app-complete-v190.json)

| Input | Appearance review interval | Mean raw face score inside 3–6 s | Overall tendency |
| --- | --- | ---: | --- |
| Unswapped control | 6.125–7.375 s | 0.068815 | Inconclusive |
| Recompression only | 6.125–7.375 s | 0.059211 | Inconclusive |
| Swap, limited range and Rec.601 | 3.042–5.583 s | 0.008843 | Inconclusive |
| Swap, full range and Rec.601 | 3.042–5.583 s | 0.008140 | Inconclusive |
| Swap, limited range and Rec.709 | 3.042–5.583 s | 0.008720 | Inconclusive |
| Swap, full range and Rec.709 | 3.042–5.583 s | 0.007876 | Inconclusive |

The appearance interval overlapped 62 of the 72 substituted native frames
in each swap rendering, for temporal IoU 0.861. The representative images
came from 2.792, 4.333 and 5.833 seconds. The third image follows the
detected interval and still falls inside the true edit. The evidence
therefore shows a review range and its context, with incomplete coverage
at the actual edit boundary. Both controls highlighted the same later
appearance change, outside the test window. If interpreted as swap
detector output, those control highlights would be false alarms.

The raw frame classifier responded less strongly inside the edited range
than it did on either control. These scores are uncalibrated. That route
completed on every sample and produced no bound score-based review
interval. The appearance result supplies a useful location to inspect on
this source. The frame scores and overall verdict do not confirm AI origin.

This experiment used one synthetic source, one publisher swap and four
interpretations of unspecified color metadata. It supplies no population
false-alarm rate or lip-sync result. The v187 startup failure occurred
before inference and is preserved by the successor freeze. The six v190
workers took 37.9–40.8 seconds each with peak sampled process-tree RSS
at or below 3.38 GiB. No serving default changed.
[Source and rights](video-faceswapai-source-v185.md) · [Frozen startup correction](evaluation-panels/video-faceswapai-full-app-freeze-v190.json)

The aggregate's freeze digest covers the private worker's original JSON
serialization. The published freeze sorts the keys and matches every value.

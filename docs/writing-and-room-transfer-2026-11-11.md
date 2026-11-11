# Writing and measured-room transfer checks

## Writing review

The classifier's raw review point flagged many genuine articles in the archived replay. A stricter candidate fixed its point at the second-highest document maximum among 13 genuine development articles. The comparison retained six candidates in source order and measured overlap over the actual scored text. A useful location required character intersection-over-union of at least 0.30. The candidate was applied to archived responses with zero new model calls. [Science explainer methods](text-elife-relative-paragraph-v292-method.md)

| Archived comparison | Current review rule | Stricter candidate |
| --- | ---: | ---: |
| Science explainer confirmation, genuine files with a flag | 12/13 | 2/13 |
| Science explainer confirmation, mixed files with a useful location | 7/10 | 4/10 |
| Federal article transfer, genuine files with a flag | 43/54 | 14/54 |
| Federal article transfer, full rewrites with any review flag | 53/54 | 39/54 |
| Federal article transfer, longer inserts with a useful location | 22/54 | 7/54 |
| Federal article transfer, short inserts with a useful location | 1/23 | 0/23 |

The federal archive spans 30 different source groups and two different generators. The replay checked that the current plain-text extraction and source geometry matched all 262 available input records and used the current evidence scope for highlights. It excluded 62 unavailable planned records from the eligible denominators shown above and retains them in the exact aggregate. The two archives were already used to evaluate other rules. The stricter point reduced false alerts and missed too many real edits to replace the current review rule. [Explainer aggregate](evaluation-panels/text-review-point-explainer-replay.json) · [Federal transfer aggregate](evaluation-panels/text-review-point-federal-transfer.json) · [Original federal audit](text-gohumanize-transfer-v198-review.md)

## Measured stereo rooms

The model-free preflight prepared all 36 planned inputs from two rooms. Each room used six source groups, with two groups each of genuine speech and the two retained synthetic voice families. Each group had a clean control and the same noise excerpt rendered from two lateral positions at a joint signal-to-noise ratio of approximately −10 dB. A pair qualified when the cleaner channel reversed by at least 3 dB between the two noise positions.

The BBC room qualified for three of six source groups. The Surrey room qualified for zero of six. The predeclared two-room gate required at least three groups in each room. This preflight ran zero speech-model forwards and provides no detection or recovery result. The room geometry outcome remains part of the record. [Exact preflight aggregate](evaluation-panels/audio-measured-room-asymmetry-preflight.json) · [Surrey source qualification](audio-room-assets-v299-result.md) · [BBC source qualification](audio-bbc-room-assets-v303-result.md)

A separately frozen BBC-only test verified the native stereo and production mono samples for all 18 cases before running 5,076 speech-activity forwards. Mono and the stereo policy each admitted 12 cases. The same four groups were admitted in all three conditions. Both native channels rejected every case that mono rejected, including two noisy cases with at least 3 dB between the channel SNRs. The fixed stereo policy recovered zero cases. The separate voice-authenticity stage had no rescued intervals to compare. [Exact speech-activity aggregate](evaluation-panels/audio-measured-bbc-activity-transfer.json)

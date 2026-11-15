# Testing lip timing between native frames

The local video review compares a short change in the timing of speech
and mouth movements with the best constant offset for the clip. We
tested whether estimating mouth opening between frames or requiring
more than one speech event to support the same interval improves that
comparison. Neither variant was enabled in the application.
[Existing check and test sources](video-lip-viseme-transfer-2026-11-15.md)

## Comparison

- The current check uses the nearest native frame for each speech event
  and timing offset.
- The interpolation variant uses a linear estimate when both adjacent
  native frames have measured landmarks and are at most 50 ms apart.
  The estimate keeps the nearer original frame as its evidence anchor.
- The single-event variant removes each speech event inside the chosen
  interval in turn. It keeps the interval only if the same two-offset
  fit still beats every constant offset each time. It does not search for
  a replacement interval. We tested this rule with both frame measurements.

The four original fit speakers supply the reference mouth-opening
measurements. Each arm uses its own two operating points from the
maximum and second-highest control score across the same four
calibration speakers. The study replays archived measurements on 16
held speakers, then applies the same settings to eight separate
speakers in PCM and 48 kb/s AAC. The two carrier versions share the
same underlying recordings and should be read together. This study
uses no new recordings or model assets.

## Held results

Each row reports useful locations on staged local shifts and flags on
ordinary audio or an offset applied to the whole clip. A useful
location meets the previous study's interval overlap and shift
direction requirements. The original thresholds and baseline counts
reproduced.

| Panel and tighter operating point | Useful local shift locations | Ordinary flags | Constant-offset flags |
| --- | ---: | ---: | ---: |
| Primary, current | 9/128 | 2/192 | 5/128 |
| Primary, single-event check | 9/128 | 0/192 | 5/128 |
| Primary, interpolation | 8/128 | 16/192 | 7/128 |
| Primary, both | 8/128 | 20/192 | 8/128 |
| PCM transfer, current | 4/32 | 0/48 | 3/32 |
| PCM transfer, single-event check | 3/32 | 0/48 | 3/32 |
| PCM transfer, interpolation | 3/32 | 11/48 | 6/32 |
| PCM transfer, both | 2/32 | 10/48 | 6/32 |
| AAC transfer, current | 3/32 | 0/48 | 1/32 |
| AAC transfer, single-event check | 3/32 | 0/48 | 1/32 |
| AAC transfer, interpolation | 3/32 | 11/48 | 6/32 |
| AAC transfer, both | 1/32 | 10/48 | 5/32 |

On the primary set, interpolation found five new useful locations and
lost six of the current ones at the tighter point. The single-event
check removed two ordinary flags there and lost a valid location in the
PCM transfer clips. At the wider point, the same check lost useful
locations in all three panels. The full receipt retains paired gains,
losses, and counts of clips with too few eligible events. These results
support retaining the current video timing check.

## Replay

The replay uses existing source measurements and Python's standard
library. It verifies the source digests, reconstructs the two fit
profiles and both points for each arm, and checks the resulting
per-file intervals and paired counts against the saved receipt.
[Evaluation receipt](evaluation-panels/video-lip-subframe-transfer-2026-11-15.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_lip_subframe_transfer review \
  docs/evaluation-panels/video-lip-subframe-transfer-2026-11-15.json.gz
```

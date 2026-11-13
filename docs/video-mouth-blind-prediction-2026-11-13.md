# Predicting a mouth picture from neighboring frames

This offline experiment tests whether neighboring pictures predict an
unedited mouth more closely than a redrawn mouth. It borrows the idea
of a counterfactual picture from [LoCC](https://arxiv.org/abs/2606.22772).
The implementation uses OpenCV operations and the project's pinned,
MIT-licensed face geometry weights.
[Face geometry license](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Both arms align frames on the eyes and estimate the middle picture from
pairs one and two native frames away. The first averages the surrounding
pair. The second computes bidirectional flow between those two pictures
and averages their half-warped predictions. The middle frame is absent
from the motion estimate. It supplies the observed pixels and positions
the mouth and surrounding comparison regions. The response compares
upper-quartile prediction error in those regions and uses the weaker
of the two native hops. The same fixed 8-, 12-, 16- and 20-frame selector
requires five frames on both flanks in a single observed shot.

The third-highest response on 56 ordinary GRID development deliveries
set each operating point, with the preset floor. Both arms then ran
on the same 92 H.264 carrier files used in the earlier compressed-domain
test. The comparison reuses previously evaluated footage and spans
17 authored neural redraws from two workflows and 75 ordinary deliveries.
One correct location requires temporal intersection over union of at
least one half at the frozen operating point. Ten LivePortrait cases
and one MuseTalk case qualified for the separately verified local speech
timing comparison. [Carrier protocol and source terms](video-mouth-codec-prediction-2026-11-13.md)

| Held outcome | Neighbor average | Motion-adjusted prediction |
| --- | ---: | ---: |
| Measured neural redraws | 17/17 | 17/17 |
| Measured ordinary clips | 75/75 | 75/75 |
| Correct edit locations | 1/17 | 0/17 |
| Correct locations in independently timing-qualified subset | 0/11 | 0/11 |
| Neural redraws flagged at the wrong location | 5/17 | 3/17 |
| Ordinary clips flagged | 17/75 | 8/75 |
| Matched unaltered originals flagged | 4/17 | 3/17 |

The one correct location came from a MuseTalk redraw outside the
timing-qualified subset. Warping reduced ordinary flags but lost that
location and left three wrong locations. Both arms remain offline. This
panel tests short, prepared edits at one encoding profile. It does not
estimate performance on other encoders, ordinary recordings in general,
or synchronized neural edits from other sources.

Independent FFmpeg replay reopened 4,200 development and 5,834 held
timestamps and 106 exact before, during or after frame and crop pairs.
The public ledger contains scalar results, anonymous groups, timestamps,
and pixel hashes. The source videos and crops stay local.
[Decision ledger](evaluation-panels/video-mouth-blind-prediction-2026-11-13.json)

```sh
python3 -B -IS backend/scripts/review_video_mouth_blind_prediction.py \
  docs/evaluation-panels/video-mouth-blind-prediction-2026-11-13.json
```

# Review overlapping face changes on opposite sides of a split screen

The brief face-change review can show a range on each side of a
split-screen clip even when the changes overlap. Each face must pass
its own appearance and surrounding-motion checks at both boundaries.
The second range may overlap the first by at least three native frame
intervals. The existing first, same-side follow-up and separated
opposite-side selections retain priority. This extension reuses the
existing pictures, track measurements and review points.
[Separated opposite-side review](video-face-opposite-pane-2026-11-14.md)
· [Face localization weights and MIT license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license)
· [Face appearance weights and Apache 2.0 license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

The video evidence view, Standard PDF and JSON identify each side,
its exact interval, and original frames and face crops before, during
and after the change. When both ranges use the same source frame, their
crops still point to the two independently tracked faces. A face
appearance change carries zero weight in the AI estimate.

The local scope is four seconds of upright 720p H.264 at exactly 30 fps,
with a stable face track on each side. The existing
`VERITAS_VIDEO_FACE_OPPOSITE_PANE_AVC_REVIEW=true` setting enables the
extension when the short AVC, pane track and same-side review are also
enabled. Historical separated intervals keep their original verification
rules.

## Local comparison

The panel constructs 56 new video deliveries from eight earlier source
recordings with known face alterations. Eight clips align the onset on
both sides. Eight offset one side by three pictures, alternating which
side changes first. The remaining clips have one altered pane, no edit,
ordinary speaker cuts or a short whole-picture brightness change. The
source-clock adjustment on the right shifts clean and altered source
pictures together. The first or last available picture fills any
uncovered frames at the clip edges. Each authored interval is verified
against the original encoded side's labeled interval. The clips reuse
earlier people and edited source frames.
[Earlier clips, rights, and archive preparation](video-face-split-track-2026-11-13.md)

An exact result requires the correct side and both known
native frame boundaries. The evaluation compares the prior output
with the new same-time follow-up while counting additional ranges
on ordinary controls and single-edit clips.

| Clip type | Prior exact ranges | New exact ranges | Extra incorrect ranges |
| --- | ---: | ---: | ---: |
| Simultaneous starts, eight clips | 8/16 | 16/16 | 0 |
| Staggered starts, eight clips | 8/16 | 16/16 | 0 |
| One altered pane, 16 clips | 16/16 | 16/16 | 0 |
| Unchanged, speaker cuts and brightness changes, 24 clips | 0 | 0 | 0 |

All eight groups gained a correct location. These results measure
source-bound locations on staged edits in this one video profile.
Natural calls, additional codecs, and larger multi-person layouts
require separate evaluation.
[Paired measurements](evaluation-panels/video-face-overlapping-pane-2026-11-14.json)

## Local app verification

Eight local analyses compared the opposite-side option off and on.
No job started before simulated USD 9.99 payment. Two edited examples
from one group and two controls retained their previous overall
assessments. In the simultaneous example the right-side range is
**2.033–2.300 s** and the left-side range is **2.033–2.233 s**. In the
staggered example the right-side range is **1.933–2.200 s** and the
left-side range is **2.033–2.233 s**. The one-edit and light-pulse
controls gained no range.

The worker, evidence API, JSON export, Standard PDF and signed
original-file verification agreed on the range and its side. Each
positive report reproduced twelve exact original frames and crops
across the two faces. The separate panel replay checked all 6,720
presentation ticks and 135 selected source-frame commitments.
[Local app receipt](evaluation-panels/video-face-overlapping-pane-app-2026-11-14.json)

## Independent replay

The checked-in evaluation contains source timing, hashes and scalar
measurements. It includes neither footage nor face vectors.
Recount the result with standard-library Python.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-overlapping-pane-2026-11-14.json
python3 -B -S -m scripts.video_face_overlapping_pane_transfer review "$receipt"
```

Build the parent video bank from the
[side-by-side replay](video-face-split-track-2026-11-13.md#replay).
The new replay verifies the same native frame clocks and original
picture commitments. To run the local app comparison, use the same
working directory and the installed face weights.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_overlapping_pane_transfer prepare "$bank" --bank "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_overlapping_pane_transfer run "$new_receipt" --bank "$bank"
python3 -B -S -m scripts.video_face_overlapping_pane_transfer pixels "$new_receipt" --bank "$bank"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app \
  "$bank" "$app_run" --overlapping-receipt "$new_receipt"
```

# Brief face changes on opposite sides of a split screen

When the app finds a brief face change on one side, it can now report
a distinct change on the other side. It retains the existing first
range and gives the existing same-side follow-up priority. If that
follow-up found no range, the app considers the strongest result from
the independently measured face on the opposite side. At least three
native frames must separate the ranges. The app uses the existing
five-to-nine-frame scope, face appearance measurements, and motion
checks outside the face. The new selection uses measurements already
taken by the side-by-side check.

Each finding shows the side, exact time range, source frames and crops
from before, during and after the change, and support at both endpoints.
The browser, PDF, JSON, and signed evidence use the same source clock.
The observations have zero AI confidence weight. A change can also
come from a normal cut or a change in lighting or expression.
[Existing side-by-side method and replay](video-face-split-track-2026-11-13.md)
· [Face detection weights and MIT license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license)
· [Face appearance weights and Apache 2.0 license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Local comparison

Forty-eight newly composed 720p30 H.264 clips reused eight recordings
from the earlier side-by-side study. Each pairing had two staged
opposite-side swap deliveries, two single-swap deliveries, an unchanged
delivery, and a delivery with ordinary speaker or camera cuts. The
original and deliberately changed parent pictures were verified before
composition. An exact location requires the correct side and both
known frame boundaries.

| Measure | Existing review | With opposite-side follow-up |
| --- | ---: | ---: |
| Exact ranges on 16 two-swap clips | 16/32 | 30/32 |
| Two-swap clips with both ranges located | 0/16 | 14/16 |
| Misplaced ranges on two-swap clips | 0 | 0 |
| Exact ranges on 16 single-swap clips | 15/16 | 15/16 |
| Extra ranges on single-swap clips | 0 | 0 |
| Ordinary clips flagged | 0/16 | 0/16 |

The 14 additional correct locations covered six people. The staged
layouts and reused recordings limit how far these results transfer.
The trial applies to upright four-second 720p H.264 files at exactly
30 fps. Natural split-screen calls, simultaneous changes and other
codecs still need their own evaluation.
[Paired measurements and decoded frame commitments](evaluation-panels/video-face-opposite-pane-2026-11-14.json)

## Local app verification

Eight simulated prepaid runs compared the feature off and on. No job
started before payment, and all four paired overall results matched.
The two positive cases added intervals at **2.967–3.233 s** on the
right and **3.167–3.367 s** on the left. Each retained the first
range on the other side. The worker output, evidence API, JSON, PDF and
signed source replay agreed on the ranges and all 12 source frames and
crops. The single-swap and ordinary-cut controls gained no findings.
The separate replay of the comparison checked 5,760 native time ticks
and 135 distinct displayed source frame commitments.
[Local app receipt](evaluation-panels/video-face-opposite-pane-app-2026-11-14.json)

## Local use and independent replay

For the local trial, set
`VERITAS_VIDEO_FACE_OPPOSITE_PANE_AVC_REVIEW=true` alongside the
short AVC, pane track and same-side follow-up flags. The source
setting defaults to off.

Use the [parent bank instructions](video-face-split-track-2026-11-13.md#replay)
to build the input clips. The receipt stores measurements, frame hashes
and side/time labels. The source video and face vectors stay local.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-opposite-pane-2026-11-14.json
python3 -B -S -m scripts.video_face_opposite_pane_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_face_opposite_pane_transfer prepare "$bank" --bank "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_opposite_pane_transfer run "$new_receipt" --bank "$bank"
python3 -B -S -m scripts.video_face_opposite_pane_transfer pixels "$new_receipt" --bank "$bank"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app "$bank" "$app_run" --opposite-receipt "$new_receipt"
```

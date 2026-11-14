# Face-change localization in side-by-side video

The short-video check can follow the face on each side of a 720p
side-by-side view. It uses rows of face boxes from verified source
frames. At least eight of the twelve sampled pictures must contain one
candidate on each side, with neither box crossing the center margin.
Each side then has its own geometry and boundary shortlist. A region
with two competing boxes is ineligible at that picture. The approach
adapts within-shot association by geometric overlap, using the existing
commercially admitted face assets. [Tracking method](https://arxiv.org/abs/2001.05080)
· [Face assets and terms](video-appearance-plan-v107-method.md#weights-resources-and-publication)

For each possible boundary, the check compares motion around the face
with motion in the rest of that side. It counts 24-pixel tiles on the
384 × 216 measurement grid with an abrupt mean luma change of at least
12 levels above nearby motion. It excludes tiles within a padded face
box. Eight or more changed tiles at either boundary withhold that range.
The check retains the same existing appearance point and interval
lengths. It marks a short appearance change for inspection and carries
zero weight in the overall AI result.

## Evaluation

The panel contains 64 newly composed clips with exact 720p30 source
clocks. The eight source recordings differ from the eight in the
earlier single-large-face study. The recordings had prior project
exposure. Three deliveries per recording reuse the labeled swaps in
the left view, right view, or a view with an ordinary resize around
the first swap. Five matched controls include unchanged views,
speaker resize, ordinary camera cuts, a guest-only replacement and
a brief exchange of the two views. These are staged layouts from
existing recorded sources. The panel does not establish field accuracy
for real video calls, crowded frames, moving cameras, or other codecs.
[Source bank and attribution](video-face-moving-shortlist-2026-11-13.md#source-separated-confirmation)
· [Earlier layout study](video-face-position-track-2026-11-13.md)

| Selection | Exact interval on edited clips | Ranges on ordinary controls | Clips with a measurable boundary pair |
| --- | ---: | ---: | ---: |
| Current face-position fallback | 0/24 | 0/40 | 0/64 |
| Largest detected face | 15/24 | 0/40 | 42/64 |
| Separate face on each side | 24/24 | 0/40 | 52/64 |
| Separate faces with off-face motion check | 24/24 | 0/40 | 52/64 |

All 24 new locations were on the labeled side and match both
boundaries exactly. The eight source recordings account for the
24 repeated-layout deliveries. The improvement comes from separating
the tracks. No candidate boundary in this panel reached the off-face
motion limit, so this experiment establishes no incremental reduction
in false alarms from that rule. A controlled pixel test exercises
the rule separately by moving the full image and the face alone.

The published reader keeps proposed ranges from each side so an
independent reviewer can check which face supplied the range. The
ordinary guest replacement and camera cut cases remain separate in
the receipt. Researchers can reuse the same reporting format for
their own rights-cleared recordings.

## App verification

The app showed the exact **2.033–2.233 s** interval in both the left
and right examples. Source frames **60, 64 and 67** showed the face
before, during and after the edit. API evidence, JSON, PDF and signed
original replay agreed on the tracked side, source frames, and
counts of abrupt changes away from the face. The ordinary speaker
cut gained no range. In the isolated six-run checkout comparison,
zero jobs ran before simulated payment. The existing overall
assessments and previously recorded ranges were preserved.
[App verification receipt](evaluation-panels/video-face-split-track-app-2026-11-13.json)

The repository default is off. To try the feature locally with
appearance mode and the existing short AVC check, set
`VERITAS_VIDEO_FACE_PANE_TRACK_AVC_REVIEW=true`. It runs after
existing first-range checks find no range, within the same shared
12-second time budget. The source must be an upright four-second
H.264 `avc1` MP4 with 120 verified pictures at 30 fps and 8-bit
4:2:0.

## Replay

The public receipt keeps timing, measurements, and frame
commitments. Source video and face vectors remain local.
The standard-library reader recomputes paired results and the
selection between sides. The separate FFmpeg replay matched 7,680
native clock ticks and the hashes of 78 selected original frames.
Altering an in-range tile count in the app evidence caused source
replay to reject the record.
Use the source-separated bank from the earlier
[moving-face replay](video-face-moving-shortlist-2026-11-13.md#replay).

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-split-track-2026-11-13.json
python3 -B -S -m scripts.video_face_split_track_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_face_split_track_transfer prepare "$parent_bank" "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_split_track_transfer run "$pane_bank" "$new_receipt"
python3 -B -S -m scripts.video_face_split_track_transfer pixels "$receipt" "$pane_bank"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app "$pane_bank" "$app_run" --pane-track
```

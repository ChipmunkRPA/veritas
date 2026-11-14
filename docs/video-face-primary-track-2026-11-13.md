# Keep the larger face in view during a brief video change

The local Full and Deep video worker can follow the main face when a
smaller detection appears nearby. The review uses position and size.
It requires a face at least four times the area of competing detections,
with little overlap, plus at least four consistent anchor positions.
It interpolates expected position between those anchors. Ambiguous
faces and large jumps in position are unassessed. The approach adapts
the idea of geometric association from multi-object tracking and loads
no additional model. [Geometric association research](https://arxiv.org/abs/2110.06864)
· [Existing face assets and permissions](video-appearance-plan-v107-method.md)

The optional check runs only if existing brief face review found no
range and sampled pictures included a second detection. It uses
the current 12-second proposal budget and searches for a face change
lasting five to nine frames. The bounding boxes and both motion
measurements accompany the exact source clock and before, during,
and after pictures. The JSON carries the saved candidate boxes used
to select the measured face. Signed replay recalculates the association
from those boxes, checks the linked images against the submitted video,
and reproduces the pixel shortlist. The evidence is an appearance
observation with zero weight in the AI assessment.

To try it locally, set `VERITAS_VIDEO_FACE_PRIMARY_TRACK_AVC_REVIEW=true`
and `VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true` in appearance mode.
The route requires an upright 1280 × 720 H.264 `avc1` MP4 with
120 verified native pictures at 30 fps and 8-bit 4:2:0. It defaults
off. Simulated prepaid analysis recovered an exact **3.400–3.600 s**
range with source pictures **101, 105 and 108**. A genuine cut added
no range. The API, JSON, PDF and signed package agreed on the original
images and crops. [App verification](evaluation-panels/video-face-primary-track-app-2026-11-13.json)

## Evaluation

The comparison uses the same five-to-nine-frame motion shortlist and
appearance response for three methods. The first takes sole detections,
the second takes the largest detection per frame, and the third
requires a geometrically consistent dominant face. The new arms count
additional matches only when the sole-face method supplied no range.
The principal comparison reuses videos from two existing local
studies. It includes every ordinary clip in those studies once and
the edited clips without an incumbent first range.
[Previous moving-face study](video-face-moving-shortlist-2026-11-13.md)

| Replayed uploads | Sole-face check | Added by largest face | Added by consistent dominant face |
| --- | ---: | ---: | ---: |
| Source-separated bank, 4 edited and 24 ordinary | 0 exact locations | 4 | 4 |
| Development bank, 19 edited and 97 ordinary | 4 exact locations | 0 | 0 |
| Additional ordinary flags across both banks | 0 | 0 | 0 |

All four additions came from the one recording that motivated this
test. They include two encodes of the same dual-swap clip. Only one
source improved in that replay, leaving the prespecified minimum of
two sources unmet. The application option remains off by default.
[Replayed case ledger](evaluation-panels/video-face-primary-track-transfer-2026-11-13.json)

A separate set of 35 authored video-call deliveries adds a static
portrait inset from a different archived source. It tests the two
rendered swap variants and matching untouched, same-recording jump,
and different-speaker cut controls. All seven host captures exclude
the recording that motivated the change. Their original clips had
prior project exposure. The inset simulates a second face detected in
the frame without adding a second moving person or occlusion.

| Picture-in-picture test | Sole-face check | Largest face | Consistent dominant face |
| --- | ---: | ---: | ---: |
| Edited deliveries with an exact interval | 0/14 | 13/14 | 11/14 |
| Ordinary deliveries with a range | 0/21 | 0/21 | 0/21 |

The consistent route recovered exact ranges across six hosts. The
largest-face route reached one additional host and two additional
deliveries. The stricter rule serves as the local trial because it
requires a continuous position and refuses similarly sized faces.
The set offers limited evidence for ordinary cuts and cannot measure
performance with crossing faces or multiple people of comparable size.
[Picture-in-picture ledger](evaluation-panels/video-face-primary-track-inset-2026-11-13.json)

## Replay

Use standard-library Python to recount findings and source timing.
The receipts include scalar responses, source commitments and
presentation ticks. The source media and face vectors remain local.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.video_face_primary_track_transfer review \
  docs/evaluation-panels/video-face-primary-track-transfer-2026-11-13.json
python3 -B -S -m scripts.video_face_primary_track_picture_in_picture review \
  docs/evaluation-panels/video-face-primary-track-inset-2026-11-13.json
```

For a media replay, supply the local bank from the moving-face study
and the earlier two-range bank. The `pixels` phase uses a separate
FFmpeg stream to verify all native timestamps and all selected
source-image hashes. The replay checked 17,280 timestamps and 24 source
frames across the 144 retained clips, plus 4,200 timestamps and 42 source
frames in the 35 picture-in-picture clips.

```sh
python3 -B -S -m scripts.video_face_primary_track_transfer pixels "$receipt" \
  --confirmation "$moving_face_bank" --previous "$two_range_bank"
python3 -B -S -m scripts.video_face_primary_track_picture_in_picture pixels "$inset_receipt" "$inset_bank"
```

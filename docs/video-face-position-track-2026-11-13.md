# Face-change localization when another face changes size

The face-position fallback uses matched observations before and after a
temporary change in the size of a nearby face. It interpolates the expected
main-face box between dominant anchors, up to 50 pictures apart. An
intermediate detection needs at least 0.35 intersection over union with the
expected box and between half and twice its area. The check requires one
plausible detection and rejects overlap with a competing face above 0.10.
Results outside the anchors still require the dominant face. The existing
short motion and appearance measurements then look for a five- to nine-picture
change with supporting evidence at both boundaries. The geometric association
uses the existing commercially admitted face assets. The work draws on
observation-based tracking and published evidence of association errors during
occlusion. It implements its own box rules. [Observation-Centric SORT](https://arxiv.org/abs/2203.14360)
· [Occlusion-Aware SORT](https://arxiv.org/abs/2603.06034)
· [Existing asset terms](video-appearance-plan-v107-method.md#weights-resources-and-publication)

## Local comparison

The bank has 48 newly composed deliveries from eight retained publisher
recordings. These recordings differ from the eight used to prepare the
earlier static-inset bank. Both banks have prior project exposure. The
moving guest comes from another recording in this bank and temporarily
enlarges around the existing frame-labeled events. Each host contributes
two versions of a dual-swap clip and four matched controls. The controls
include an otherwise unchanged host, a same-source jump, a cut to another
person, and a guest-only cut. Each pair of encodes shares one underlying
edit. The replays keep all 48 deliveries in the denominator. The
largest-face comparison also uses the moving shortlist and follows the
largest detected face at each measured frame. [Source bank and labels](video-face-distinct-avc-2026-11-13.md)
· [Publisher and source terms](video-common-source-preparation-v89.md)

| Selection rule | Edited deliveries with an exact interval | Ordinary deliveries with a range | Deliveries with a measurable boundary pair |
| --- | ---: | ---: | ---: |
| Sole face | 0/16 | 0/32 | 0/48 |
| Largest face | 14/16 | 0/32 | 28/48 |
| Consistently dominant face | 0/16 | 0/32 | 0/48 |
| Position between dominant anchors | 6/16 | 0/32 | 13/48 |

The six recoveries represent three source events. Both boundaries match the
authored labels exactly. The fallback met the study requirement of three
additional exact deliveries across at least two hosts, with zero new cut
flags and no increase in misplaced edit ranges. The largest-face arm covered
more edits in this layout. It remains an evaluation arm pending a comparison
against controls where the largest detection switches between people.
Overlapping or crossing faces and untested cameras remain open questions.
None of the reported observations changes the overall AI assessment.

## Replay

The published receipt contains scalar responses, timings and source
commitments. The writer keeps video, face vectors and unpublished source
metadata local. The public reader runs without inference dependencies.
The FFmpeg readback verifies all 5,760 clock ticks and 42 distinct selected
original frames. Regenerate the parent source-separated AVC bank with the
[native commands](video-face-distinct-avc-2026-11-13.md#independent-replay)
before preparing or scoring the new deliveries.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/video-face-position-track-2026-11-13.json
python3 -B -S -m scripts.video_face_position_track_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_face_position_track_transfer prepare "$parent_bank" "$new_bank"
backend/.venv/bin/python -B -m scripts.video_face_position_track_transfer run "$new_bank" "$new_receipt"
python3 -B -S -m scripts.video_face_position_track_transfer pixels "$receipt" "$new_bank"
backend/.venv/bin/python -B -m scripts.video_face_moving_shortlist_app "$new_bank" "$app_run" --position-track
```

For a local trial in appearance mode, set
`VERITAS_VIDEO_FACE_POSITION_TRACK_AVC_REVIEW=true` together with
`VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true`. The fallback runs only when
the existing checks have found no brief range and a second face was
detected. It uses the existing 12-second budget and needs an upright
720p H.264 MP4 with 120 verified native pictures at 30 fps. The repository
default is off.

## Local app verification

Six simulated prepaid analyses compared the option on and off. The
tandem-compressed swap produced the exact **3.200–3.400 s** range
with source frames **95, 99 and 102** before, during and after the
change. The two ordinary controls produced no additional range.
The overall assessment and earlier ranges matched in each pair of runs.
The API, JSON, and Standard PDF showed the same evidence. The signed
package reproduced all six verified original images and crops. No job
ran before the simulated payment. [Application receipt](evaluation-panels/video-face-position-track-app-2026-11-13.json)

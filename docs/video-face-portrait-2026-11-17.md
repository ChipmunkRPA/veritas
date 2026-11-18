# Brief face changes in 30 fps portrait video

The local video worker can bracket a brief change in face appearance on an
upright four-second 360 × 640 or 720 × 1280 H.264 MP4. The supported source has
120 presented pictures at exactly 30 fps, with no rotation flag or interpolated
clock. The check compares every other picture, including five sampled faces on
either side of the proposed interval. It checks every adjacent picture around
the proposal for an abrupt cut. Each size has its own frozen operating point.

The measured appearance change is shown with its source time range, three
original frames from before, during, and after, and their face crops. PDF,
JSON, and the signed evidence package use the same source timestamps. The
record reports an observed appearance change at zero AI-confidence weight.
Local development users can enable it with both
`VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW=true` and
`VERITAS_VIDEO_FACE_PORTRAIT_RETURN_REVIEW=true`. The feature remains opt-in.

## Held results

The evaluation used 84 exact publisher clips from 28 source groups. Ten
groups fixed each operating point and 18 groups were held for testing. Each
group contributes two staged swaps and five ordinary controls, including
exposure changes, face blur, a jump, and a cut to a different recording.
Each treatment has a direct and a strongly recompressed delivery at both
sizes. The 504 deliveries repeat source groups across sizes and encodes.
An interval counts as located when intersection-over-union with the labeled
edit is at least 0.5.

| Portrait delivery | Correct edit ranges | Edits with a comparable proposal | Ordinary alerts | Controls with a comparable proposal |
| --- | ---: | ---: | ---: | ---: |
| 360 × 640 direct | 20/36 | 24/36 | 0/90 | 63/90 |
| 360 × 640 recompressed | 19/36 | 24/36 | 0/90 | 63/90 |
| 720 × 1280 direct | 17/36 | 25/36 | 2/90 | 65/90 |
| 720 × 1280 recompressed | 16/36 | 23/36 | 1/90 | 58/90 |

No selected range landed on the wrong part of a staged edit. The adjacent
shot check removed one ordinary alert for each size and encoding, preserving
all correctly located ranges. The broader video checks run alongside this
comparison. Their sensitivity and false-alarm rates are outside this panel.

Publisher clips were reserved by asset hash and known source URLs and donor
keys. The roster excludes the earlier comparison groups. Different donor keys
do not establish different people or exclude model pretraining overlap.
The panel uses controlled swaps on portrait crops, so it does not measure
performance for all phone uploads, variable frame rates, or lip sync.
The locator weights use MIT terms and the appearance weights Apache-2.0.
[Publisher dataset and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) ·
[Locator license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance weight license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Local product replay

The local worker ran the face locator, tracker and appearance checks for
one held positive, at both sizes and encodings. The previous face checks
recorded no range on those four deliveries. Enabling the portrait check
returned frames 16–27 at 0.533–0.933 s, matching the staged interval in all
four cases. In each replay the ordinary browser view showed three frames
and three crops, and the middle control sought to 0.733333 s. Offline
verification confirmed the signed packages against the inputs. The four
paired rollback runs returned the same assessment. All eight jobs followed
simulated $9.99 payment. This targeted worker replay exercised one held
source group with the face checks active. The held panel above supplies the
ordinary controls and per-size coverage counts.
[Worker and browser receipt](evaluation-panels/video-face-portrait-app-2026-11-17.json)

## Reproduce

The [source roster](evaluation-panels/video-face-portrait-sources-2026-11-17.json)
pins each publisher asset. The [development measurements](evaluation-panels/video-face-portrait-calibration-2026-11-17.json.gz)
and [held measurements](evaluation-panels/video-face-portrait-held-2026-11-17.json.gz)
bind the operating points, exact frame comparisons, and placement decisions.
The [independent input export](evaluation-panels/video-face-portrait-independent-2026-11-17.json.gz)
contains the verified roster of all 504 delivered clips and matching source
bindings, interval labels, and paired predictions with and without the shot check.

```sh
panels=docs/evaluation-panels
roster="$panels/video-face-portrait-sources-2026-11-17.json"
frozen="$panels/video-face-portrait-calibration-2026-11-17.json.gz"
held="$panels/video-face-portrait-held-2026-11-17.json.gz"
runner=backend/.venv/bin/python
export PYTHONPATH=backend
"$runner" -B -m scripts.review_video_face_portrait verify "$held" --freeze "$frozen"

run=$(mktemp -d)
"$runner" -B -m scripts.video_face_temporal_panel acquire "$roster" "$run/sources"
for height in 640 1280; do
  "$runner" -B -m scripts.video_face_portrait_panel prepare "$roster" "$run/sources" \
    "$run/held-$height" --role held --height "$height"
  "$runner" -B -m scripts.video_face_portrait_panel measure "$run/held-$height" "$run/$height.jsonl"
done
"$runner" -B -m scripts.review_video_face_portrait evaluate "$run/held.json.gz" \
  --panel "$run/held-640" "$run/640.jsonl" --panel "$run/held-1280" "$run/1280.jsonl" --freeze "$frozen"
"$runner" -B -m scripts.export_video_face_portrait "$run/held.json.gz" "$frozen" \
  "$run" "$run/independent.json.gz" --method strict
```

Build the frontend before rerunning the local product replay with
`scripts.video_face_portrait_app` and `--heights 640 1280`. The saved receipts
let another detector test the same case labels and file commitments. Exact
media reproduction also depends on the local encoder and image libraries.

# Short face changes in square video

This local test asks how well the four-second portrait face check transfers to
upright 480 × 480 H.264 MP4. It uses 120 presented pictures at exactly
30 fps, 8-bit 4:2:0, the `avc1` tag and zero starting time. It tests a new
crop of the original recording. The square is cut from the native picture,
centered using the same horizontal face anchor as the portrait panel. Direct
H.264 deliveries use CRF 23.
The test keeps the previous 360 × 640 operating point, every-other-frame
face comparison and adjacent-picture shot guard.

The [before-score plan](evaluation-panels/video-face-square-plan-2026-11-18.json)
reserves two groups of cases. The paired comparison uses all 18 held
portrait captures with the same staged intervals and ordinary treatments.
Eight further captures contribute fresh clips. Their known capture links,
donor keys and content hashes were checked against the source preparation
records before measuring. Each capture has two known replacement ranges
and five ordinary controls, covering the unchanged file, an exposure
change, local face blur, a same-recording jump and a cut to another capture.
A range counts as located when its temporal intersection over union with
the labeled edit reaches 0.50.
[Prior portrait results](video-face-portrait-2026-11-17.md) ·
[Additional publisher asset roster](evaluation-panels/video-face-square-sources-2026-11-18.json)

All original publisher media are length- and hash-checked. For every delivered
file, an independent FFprobe invocation verifies the picture clock. For every
comparable proposal, a separate FFmpeg pass reopens the upload and hashes
its three decoded RGB pictures. The files were rendered from controlled short
replacements in published paired recordings. In-the-wild swaps, lip sync,
other shapes and variable picture clocks remain outside this evaluation.

The RTFS-10K dataset declares CC BY-SA 4.0 at the pinned revision. The weights
used in this comparison carry MIT and Apache 2.0 terms. The receipt contains
measurements, positions and hashes. Different known donor keys do not establish
that the portraits depict different people.
[Publisher dataset card](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) ·
[Face locator license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance weights license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Held results

| Delivery | Matched edit locations | Edits with a comparable proposal | Ordinary alerts | Ordinary clips with a comparable proposal |
| --- | ---: | ---: | ---: | ---: |
| Earlier portrait, paired captures | 20/36 | 24/36 | 0/90 | 63/90 |
| Square, same captures | 28/36 | 32/36 | 0/90 | 81/90 |
| Square, fresh captures | 16/16 | 16/16 | 0/40 | 40/40 |

Square retained 18 portrait locations, recovered 10 and lost two. Seven
paired square locations and four fresh locations matched both boundaries
exactly. No selected range was misplaced. The adjacent-picture guard withheld
four paired and one fresh proposals. The fresh evaluation uses eight captures
and the same two swap families as the earlier panel. All 182 square files and
their 21,840 exact delivery timestamps passed the independent clock check.
[Case and frame commitments](evaluation-panels/video-face-square-transfer-2026-11-18.json.gz)

## Local app verification

Six simulated $9.99 checkouts exercised the fresh edit, an unchanged file
and an ordinary cut from the same source group with the option enabled and
disabled. The enabled worker found frames 16 through 27, from 0.533 to
0.933 s. The PDF and JSON agreed on source frames 14, 22 and 28 and their
crops. Original-file replay independently verified the signed evidence.
Both ordinary controls had no square face range, and the paired runs returned
the same overall AI assessment. Chromium showed all three original frames
and the three crops, and the middle control sought to 0.733333 s.
[Local checkout and browser receipt](evaluation-panels/video-face-square-app-2026-11-18.json)

## Local use and replay

The existing development options `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW`
and `VERITAS_VIDEO_FACE_PORTRAIT_RETURN_REVIEW` also enable the measured
square profile. A qualifying upload uses an upright 480 × 480 `avc1`
video stream with 120 presented frames at exactly 30 fps, starting at zero.
The finding shows the range and original frames and face crops
before, during and after it. The finding carries zero AI-confidence weight.

The standard-library reviewer recounts detections and false flags from the
released measurements. To regenerate deliveries, use the existing project
environment, acquire both pinned rosters and configure the pinned local
assets from the [portrait replay](video-face-portrait-2026-11-17.md#reproduce).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
plan="$panels/video-face-square-plan-2026-11-18.json"
new="$panels/video-face-square-sources-2026-11-18.json"
old="$panels/video-face-portrait-sources-2026-11-17.json"
calibration="$panels/video-face-portrait-calibration-2026-11-17.json.gz"
held="$panels/video-face-portrait-held-2026-11-17.json.gz"
results="$panels/video-face-square-transfer-2026-11-18.json.gz"
runner=backend/.venv/bin/python

python3 -B -S -m scripts.review_video_face_square_transfer "$plan" "$held" "$results"
"$runner" -B -m scripts.video_face_square_transfer review "$plan" "$new" "$old" "$calibration" "$held" "$results"

run=$(mktemp -d)
"$runner" -B -m scripts.video_face_temporal_panel acquire "$new" "$run/new"
"$runner" -B -m scripts.video_face_temporal_panel acquire "$old" "$run/old"
"$runner" -B -m scripts.video_face_square_transfer run \
  "$plan" "$new" "$old" "$calibration" "$held" "$run/new" "$run/old" "$run/deliveries"
npm --prefix frontend run build
"$runner" -B -m scripts.video_face_square_app "$run/deliveries/measurements.json.gz" "$run/deliveries" "$run/app"
```

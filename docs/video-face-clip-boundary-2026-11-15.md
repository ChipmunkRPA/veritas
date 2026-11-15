# Review brief face changes near the beginning or end of a video

The bounded face review checks for 5–9 picture changes near either end
of a supported clip. The earlier stationary and moving proposals scan
native transition indices 6–114. The additional pass considers pairs
with an onset at frame 3, 4, or 5 or an exclusive end at frame 115,
116, or 117. Each transition still uses three native face views on
each side. The rule pairs the entry and exit in the motion shortlist
before using the fixed appearance and nearby-motion measurements. It
keeps a previously selected stationary or moving range.
[Prior bounded review](video-face-moving-shortlist-2026-11-13.md)
· [Consecutive-frame localization research](https://arxiv.org/abs/2606.07090)
· [Appearance and temporal residual research](https://arxiv.org/abs/2611.12508)

The supported scope is an upright, roughly four-second 720p H.264 MP4
with 120 exact native pictures at 30 fps or 30000/1001 fps and enough
single-face context.
The first and last three pictures supply context for the closest
supported transitions. A change with fewer surrounding pictures, an
irregular clock, heavy motion or no comparable face can be missed. The
measured range carries zero weight in AI confidence. Ordinary recording
changes can produce a similar appearance difference. The existing
pinned face-localization and appearance weights are reused.
[Face localization assets and MIT license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license)
· [Face appearance assets and Apache 2.0 license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Evaluation

The initial source-position check found that the retained tracks only
contained qualifying swap pictures in native frames 60 through 119.
The frozen experiment therefore offsets the intact source window to
bring edited footage near the upload's edges. These offsets were locked
before detection. All face replacements come from the same timestamp
in the corresponding publisher track. The processor verifies exact
source clocks, checks original pixel hashes and encodes the straight
120-picture excerpt without interpolation.

Four development groups and eight held groups use distinct recorded
source URLs, known donor keys and decoded source components. They are
separate from the immediately preceding fresh AVC trial. The archived
sources had earlier project exposure. Each group supplies an early
swap, a late swap, a tandem recompression of the late edit, and four
controls covering an original, an exposure change, a same-source
jump and a cut to a different recording. The two swap tracks use
the earlier publisher's different manipulation methods. A useful
location overlaps at least half the temporal union with its exact
authored range. Exact locations require both native boundaries.
[Retained source qualifications and permissions](video-common-source-preparation-v89.md)

| Held case | Prior exact times | Expanded exact times |
| --- | ---: | ---: |
| Early edit | 0/8 | 4/8 |
| Late edit | 0/8 | 3/8 |
| Late edit after tandem compression | 0/8 | 3/8 |
| Ordinary control flags | 0/32 | 0/32 |

Every detected edit matched both authored boundaries. Five of the eight
held source groups supplied a detection. Of 24 edited deliveries, 13
were assessed without a usable range and one was unassessed. Of 32
ordinary deliveries, 30 were assessed without a flag and two were
unassessed. Development found eight of 12 edits with zero flags on 16
controls. These are results for staged edits in one delivery profile.
[Development measurements](evaluation-panels/video-face-clip-boundary-development-2026-11-15.json)
· [Held measurements](evaluation-panels/video-face-clip-boundary-held-2026-11-15.json)

## Local app

Enable `VERITAS_VIDEO_FACE_BOUNDARY_AVC_REVIEW=true` with
`VERITAS_VIDEO_FACE_SHORT_AVC_REVIEW=true` for eligible Full and Deep
analyses. The added pass uses single-face measurements. The browser and
exports display timestamps, the before, during and after source frames
and crops, and the measurements at each boundary. The option defaults
to off. The evidence version records which proposal produced the range
so signed verification can replay its matching schedule.

The simulated USD 9.99 checkout exercised each of two edited clips and
one ordinary cut with the option off and on. No analysis job started
before payment. The enabled clips showed **0.133–0.367 s** and
**3.633–3.833 s**. Each PDF and JSON agreed with the evidence API on
the range and six exact source images. Signed replay recomputed the
shortlist and both boundaries against the original upload. The cut
gained no finding. The automated assessments agreed across each pair.
[Local app verification](evaluation-panels/video-face-clip-boundary-app-2026-11-15.json)

## Reproduce

The numeric reviewer recounts exact frame bounds, useful locations and
ordinary flags. Restore the publisher source archive and the earlier
AVC source-plan freeze before recreating the native inputs.
An independent FFmpeg decode checked 6,720 native timestamps and 30
representative frames on held clips, in addition to 3,360 timestamps
and 24 representative frames on development clips.
[Earlier freeze procedure](video-face-distinct-avc-2026-11-13.md)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.video_face_clip_boundary_transfer
"$runner" -B -m "$module" freeze "$run" --previous "$previous/freeze.json"
for split in development held; do
  "$runner" -B -m "$module" prepare "$run" --split "$split"
  "$runner" -B -m "$module" score "$run" --split "$split"
  python3 -B -S -m "$module" pixels "$run" --split "$split"
done
python3 -B -S -m "$module" review docs/evaluation-panels/video-face-clip-boundary-held-2026-11-15.json
"$runner" -B -m scripts.video_face_clip_boundary_app "$run" "$run/app"
```

## 29.97 fps source clocks

The same bounded search now admits `30000/1001` fps H.264 MP4 video.
The worker requires a verified, uniform presentation timestamp for every
one of the 120 pictures. It keeps each original tick and the source time
base in the evidence. The offline viewer and original-file verifier both
validate that clock. A timestamp that falls off the exact cadence leaves
this bounded check unassessed. The face measurements, schedule and decision
point carry over from the 30 fps check.

One comparison remuxed the 56 earlier held clips at the fractional clock.
The compressed picture packets were copied. All 6,720 decoded pictures
matched the original delivery byte for byte. The other comparison used
four separate retained publisher captures with native fractional clocks.
It selected and encoded consecutive source pictures without interpolation.
The two source groups are disjoint across banks. Each native capture
contributed an early swap, late swap, tandem-compressed late swap, original,
exposure change, same-recording jump and cross-recording cut. The staged
swap pictures came from corresponding publisher tracks.
[FFmpeg timestamp filter](https://ffmpeg.org/ffmpeg-bitstream-filters.html#setts)

| Fractional-clock panel | Exact edited intervals | Ordinary controls flagged |
| --- | ---: | ---: |
| Earlier pictures with only the clock changed | 10/24 | 0/32 |
| Separate captures recorded at the fractional rate | 6/12 | 0/16 |

The remuxed clips reproduced every 30 fps selection and every evidence
picture. The native captures supplied two exact early ranges, two exact
late ranges and two exact tandem-compressed late ranges. The measured
examples are short staged edits. Ordinary movement and lighting can
produce a facial appearance change in other footage. The result has
zero weight in the overall AI assessment.
An independent FFmpeg decode verified all 10,080 presentation timestamps
and 48 evidence frame hashes. Python 3.12 and 3.14 produced identical
numeric review output.
[Fractional-clock evaluation](evaluation-panels/video-face-fractional-avc-2026-11-15.json)

Six simulated USD 9.99 checkouts ran the isolated Full worker with the
fractional-clock review off and on. The enabled checks recorded
**0.133–0.367 s** and **3.637–3.837 s**, with exact source ticks
**[4004, 11,011)** and **[109,109, 115,115)** at a time base of
`1/30000`. The genuine cut produced no additional finding. The API, PDF,
JSON and signed source replay agreed on the ranges and native before,
during and after frames and crops. Both variants retained their original
automated assessments, and no unpaid job entered the queue.
[Local checkout and report replay](evaluation-panels/video-face-fractional-avc-app-2026-11-15.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.video_face_fractional_avc
"$runner" -B -m "$module" retime "$prior_boundary_run" "$run/retimed"
"$runner" -B -m "$module" native "$prior_boundary_run" "$run/native"
"$runner" -B -m "$module" publish "$run" "$run/fractional.json"
python3 -B -S -m "$module" review docs/evaluation-panels/video-face-fractional-avc-2026-11-15.json
"$runner" -B -m scripts.video_face_clip_boundary_app "$run/retimed" "$run/app" --rate 30000/1001
```

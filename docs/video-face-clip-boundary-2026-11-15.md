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

The supported scope is an upright, four-second 720p H.264 MP4 with
120 exact native pictures at 30 fps and enough single-face context.
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

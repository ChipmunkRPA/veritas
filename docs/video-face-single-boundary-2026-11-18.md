# Face appearance changes with one visible boundary

A face replacement can continue to the end of a clip or be present at its
first picture. The existing brief and local return checks compare temporary
changes with matching appearance before and after. This experiment compares
two sustained face-appearance states within one shot and reports the source
picture where the states separate. A candidate uses only the submitted
clip. It can support inspection at the transition. Either side can contain
the cause of the change.

The candidate reads every other original frame in upright 640 × 360 H.264
with 100 verified pictures at 25 fps. It compares variation within the two
sides of each proposed boundary with the separation between them. Both
the local flanks and the full observed sides must support the step. One
arm requires an abrupt jump relative to adjacent changes within the same
face track. Both arms require compatible face boxes and consistent
background detail at the proposed transition. A clip with a detected
abrupt shot change is outside the search. After coarse selection, up to
five adjacent native boundaries are retested using the closest supported
facial states. Three original pictures and aligned face crop hashes bind
the display to the submitted clip. [Recent partial face-forgery research](https://arxiv.org/abs/2608.13064)
· [Temporal consecutive-picture research](https://arxiv.org/abs/2606.07090)
· [Existing face localization assets](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license)
· [Existing appearance assets](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

The source freeze separates six development and 12 held RTFS targets by
known normalized video URL, donor key and exact source asset digests.
The underlying publisher footage had been used in earlier studies.
The staged prefix and suffix replacements are new. They use two published
renderers at matched native times. The ordinary comparison includes originals,
an exposure step, a focus step, a jump inside the same recording, and a
cut to another recording. The two encodes preserve identical authored
frame indices. Related encodes and alternative edits of the same source
share the source group. The profile tests authored transitions in the
same recording setup. The publisher license and limitations were reviewed
under the pinned source plan. [Publisher acquisition and permissions](video-common-source-preparation-v89.md)

Development selects the best arm by new supported locations, misplaced
ranges, and then ordinary source groups flagged. Each arm's point uses
the second highest ordinary source maximum across the direct and tandem
encodes, with a floor of 0.06. A boundary counts if it is within one native
picture of the authored transition. Exact boundaries are counted separately.

## Held results

The two arms tied on development with 43 exact locations from 48 edits
and two alerts from the same ordinary source group across 60 controls.
The fixed tie-break selected the persistent appearance arm at 0.165750624.
The held gate requires 12 additional locations across six or more source
groups and at least four per renderer. It allows at most two ordinary
groups or two misplaced edit locations and requires zero alerts on cuts to
other recordings.

| Delivery | Existing return checks within one frame | New boundary within one frame | Exact boundary | Ordinary flags |
| --- | ---: | ---: | ---: | ---: |
| Direct H.264 | 0/48 | 41/48 | 41/48 | 0/60 |
| Strong tandem re-encode | 0/48 | 40/48 | 39/48 | 0/60 |
| Total | 0/96 | 81/96 | 80/96 | 0/120 |

The additions span 11 of 12 held groups and both renderers. The
comparison found 39 of 48 transitions where the change began before
the clip and 42 of 48 where the change continued to its end. It
misplaced one recompressed edit, assessed six more without a qualifying
reading, and had eight unassessed edits. Eighty-two of 120 ordinary
controls had a measurable native proposal. Twenty-six ordinary
deliveries had detected cuts and stayed outside the search. The
other 12 lacked a supported native proposal. The held gate passed
with zero ordinary flags. The alternate arm gave the same held
counts at its frozen point. [Development and held decisions](evaluation-panels/video-face-single-boundary-2026-11-18.json.gz)

These results compare against the existing brief- and local-return
checks on the same 216 held files. A whole-clip reading or a different
localization method can produce other findings. All controls, including
clips without sufficient evidence, remain in the denominators. Both
encodes share the same source groups. Further testing on unstaged
footage and additional capture setups would establish broader coverage.

## Transfer to other recordings

The frozen app check also received 24 ordinary deliveries from eight
additional webcam recordings and four controlled head-movement sources.
These sources had prior project exposure for other checks. A separate
publisher provided one synthetic, aligned unswapped/swap pair. At frames
36 and 62, the test combined the publisher's paired pictures to form
four new prefix or suffix edits. All source excerpts start at 0.40 s
using the nearest native timestamps, with direct H.264 and a second
strongly compressed delivery of each case. The decoder verified all
3,400 test picture timestamps. [Webcam data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)
· [Head movement recordings](https://www.mdpi.com/2306-5729/4/3/130)
· [Publisher pair and rights](https://faceswapai.com/resources/video-face-swap-continuity-study)

The check assessed 18 of 24 ordinary camera deliveries and flagged none.
It abstained on all eight staged outside-publisher deliveries and both
encodes of their matched unchanged synthetic source. In a follow-up
check of the direct unchanged and frame-36 suffix deliveries, all 29
candidate positions had matching adjacent face boxes. At most five and
four shared textured background tiles were available, respectively.
The frozen comparison needs eight. This identifies a coverage gap for
portrait footage against a plain background. The eight staged deliveries
share one publisher target. [Case results and source commitments](evaluation-panels/video-face-single-boundary-outside-2026-11-18.json)

## Native evidence

Two independent readbacks rebuilt the exact submitted MP4 files from
the publisher sources and matched their file hashes. The direct example
located a transition at **2.280 s**, with source frames **56, 57 and
69**. The strongly recompressed example located **1.280 s**, with
frames **31, 32 and 43**. The first example used an edited prefix
and the second an edited suffix, from different source groups and
renderers. The two readbacks verified all 200 native timestamps,
six exact RGB frame hashes and six aligned crop hashes. The receipt
retains the time base, native picture positions and crop geometry.
[Native evidence readback](evaluation-panels/video-face-single-boundary-native-2026-11-18.json)

## App verification

The local Full and Deep profiles can enable this check in development
with `VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_REVIEW=true`. The supported
delivery is upright, four-second 640 × 360 H.264 MP4 with an exact
25 fps source clock. The check needs a single continuous shot and
enough comparable views of the face. The evidence view reports the
coverage outcome when a required condition is missing.

The app displays the source timestamp, last frame before the change,
first frame at the transition, and a later supporting frame. Each
view includes an unmarked original and an exact face crop. The PDF and
JSON share those frame positions and the surrounding-picture reading.
The signed original-file replay verifies exact frame clocks, original
and cropped pixels, aligned face inputs, and nearby picture similarity.
The app records a single appearance boundary and leaves the cause
and affected side undetermined. The check contributes zero overall
AI-confidence weight.

The local test completed five simulated USD 9.99 purchases before
starting analysis. It localized both held edited files at their exact
source frames, produced matching PDF and JSON reports, and independently
replayed both signed packages against the originals. The browser checked
all six playback seeks and the visible original images and crops. The
ordinary control had no match and the hard cut was shown as unavailable.
The same input returned an identical overall assessment with the check
disabled and enabled. [Local checkout and browser receipt](evaluation-panels/video-face-single-boundary-app-2026-11-18.json)

The serving implementation also rebuilt and matched the 216 held
deliveries, their recorded measurements, and 21,600 native timestamps.
It matched 510 original-frame and 510 aligned-crop hashes. A separate
35-delivery comparison matched the unrounded measures. The serving
replay retained 82 observations, including the one misplaced transition
from the held comparison. [Serving parity receipt](evaluation-panels/video-face-single-boundary-serving-2026-11-18.json)

## Reproduce

Run the numeric recount with standard Python. Its input packet retains
the measured candidates for development and held cases along with the
frozen operating point. The publisher video files, extracted pictures
and face vectors stay in the local test directory.

```sh
export PYTHONPATH=backend
python3 -B -S -m scripts.video_face_single_boundary review \
  docs/evaluation-panels/video-face-single-boundary-2026-11-18.json.gz
```

For native replay, restore the archived raw source rosters and verified
files from the [portrait trial](video-face-portrait-2026-11-17.md) and
the [local face comparison](video-local-face-intervals-2026-11-16.md).
Use FFmpeg 7.0.2 and the recorded weights. The `reserve` stage uses
only the source rosters and asset commitments. The `calibrate` stage
must finish before measuring the held data.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export VERITAS_OFFLINE=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_face_single_boundary reserve \
  "$portrait_roster" "$local_held_roster" "$run/plan.json.gz"
"$runner" -B -m scripts.video_face_single_boundary measure \
  "$run/plan.json.gz" "$portrait_sources" "$run/dev.jsonl" --split development
python3 -B -S -m scripts.video_face_single_boundary calibrate \
  "$run/plan.json.gz" "$run/dev.jsonl" "$run/lock.json.gz"
"$runner" -B -m scripts.video_face_single_boundary measure \
  "$run/plan.json.gz" "$local_held_sources" "$run/held.jsonl" --split held --lock "$run/lock.json.gz"
python3 -B -S -m scripts.video_face_single_boundary report \
  "$run/plan.json.gz" "$run/lock.json.gz" "$run/held.jsonl" "$run/result.json.gz"
```

## Compare another detector

The offline comparison accepts one result for each exact submitted file.
It measures exact and one-frame locations, misses, abstentions, source-group
gains and losses, and ordinary alerts split by control type. The prediction
template contains file hashes and leaves the expected frames and labels in
the panel. Use the native commands above to regenerate the submitted files.

```sh
export PYTHONPATH=backend
panel=docs/evaluation-panels/video-face-single-boundary-2026-11-18.json.gz
python3 -B -S -m scripts.compare_video_face_boundaries prepare "$panel" > predictions.json
python3 -B -S -m scripts.compare_video_face_boundaries score "$panel" predictions.json
```

Set each result to `localized`, `no_match`, or `unassessed`. A localized
result includes the zero-based index of the first frame of the second appearance, from
1 to 99. Use a null frame for the other two outcomes. Both commands accept
`--split development` to score the earlier development set separately.
Fix the detector and operating point before measuring held inputs. The
comparison includes all ordinary and unassessed inputs in its denominators.

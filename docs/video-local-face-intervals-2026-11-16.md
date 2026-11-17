# Locating temporary face swaps in short videos

This comparison tests whether a short appearance change can be
located on its original video clock. It uses moving footage with
labeled starts and ends in both a direct encode and a strongly
recompressed copy. It separately compares the shipped brief-change
scan and the broader appearance review already used by the worker.

## Methods and sources

Take an appearance reading every second native picture. Compare the
middle of a proposed range against four adjacent views on each side.
The proposal needs consistent adjacent face positions and similar
off-face picture detail at both ends. The fit compares the separation
of the two appearance states, the transition at both boundaries
and the variation within each state. A second comparison measures
changes in multiscale detail around the aligned face. The study
fitted an appearance-only arm, a relative-detail arm and a combined
arm. It used the same measured sample positions for all three.
[Consecutive-picture research](https://arxiv.org/abs/2606.07090) ·
[Separate visual and audio timelines](https://arxiv.org/abs/2610.33413)

The experiment reuses the pinned MIT face locator and Apache-2.0
appearance weights. It fits only the small linear comparison from
the measured feature values. Publisher swap outputs are test inputs
and their generator weights are outside the application.
[Face locator license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance weights license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

Eight source groups fit the comparison and eight separate groups
select its operating point. Both sets come from the archived
whole-clip study. The confirmation set uses 20 additional targets
with distinct publisher asset digests and recorded donor keys.
The retained metadata does not establish distinct people or unknown
pretraining membership. Publisher rights and limitations are
described in the source study.
[Publisher dataset and terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) ·
[Pinned source acquisition](video-common-source-preparation-v89.md)

Each original supplies two face-swap tracks at matching source times,
with a fixed change lasting 10–40 delivered pictures. The relative
clocks are compared from each publisher file's native origin.
Only pictures inside the labeled range are replaced before encoding.
Controls are an original, an exposure change, a jump within the same
recording, a cut to another camera recording and a local face blur.
One source lacked face detections during its scheduled blur and
contributed the other control variants. The 278 held files are
640 × 360 H.264 at 25 fps. Related encodes share source groups.

## Results against the brief-change scan

A useful range overlaps at least half the temporal union of the
prediction and the labeled range. Exact timing requires matching
both native frame boundaries. The appearance-only point selected on
the eight calibration groups increased useful ranges there from
6 to 23 of 32, with no flags on 80 controls. All choices were frozen
before measuring the held sources.

| Held delivery | Useful ranges, brief scan → with appearance fallback | Ordinary flags, brief scan → with appearance fallback |
| --- | ---: | ---: |
| Direct | 6 → 34 of 40 | 2 → 4 of 99 |
| Strong recompression | 6 → 26 of 40 | 2 → 4 of 99 |
| Both | 12 → 60 of 80 | 4 → 8 of 198 |

The selected fallback found three exact intervals and 49 within two
frames at both ends, including the intervals that the brief scan
already found. Five edited and ten ordinary deliveries remained
unassessed with the fallback. Most reported boundaries are estimates
from sampled frames. The staged edits used boundaries on the two-frame
sample grid. The wider appearance point located 67 of 80 edits with
16 ordinary flags. The selected texture-only and combined alternatives
located 14 and 27 of 80 at their tighter points with 17 and 20
ordinary flags. All three alternatives can be recounted independently.
[Frozen measurements, source positions and grouped results](evaluation-panels/video-local-face-intervals-2026-11-16.json.gz)

## Comparison with the app worker

The broader appearance and boundary checks already located most of
the edits that the fallback would add to the brief scan. An isolated
worker replay ran every one of the 63 files above the frozen point
and one matched blur control below it. Both broader checks used the
same uploads and source clocks. This replay measures additions
among all possible fallback proposals. The 64 cases do not estimate
the broader worker's overall recall or ordinary flag rate.

| Held delivery | Additional useful locations | New ordinary flags |
| --- | ---: | ---: |
| Direct | 2 | 2 |
| Strong recompression | 1 | 2 |
| Both | 3 | 4 |

The three additional locations come from two source groups. All four
ordinary flags come from the two encodes of two genuine shot changes.
The existing worker remains in use. The isolated replay exercised
65 simulated purchases, including the same recompressed file with
the candidate off and on. The app queued zero jobs before simulated
payment. The toggle left the overall assessment unchanged.

For the additional recompressed match, the browser, PDF and JSON
reported **0.480–1.680 seconds**. Signed source replay verified the
original frames and face crops before, during and after the range.
The browser sought the measured frame at **1.040 seconds**.
[Actual worker replay and exact positions](evaluation-panels/video-local-face-intervals-app-2026-11-16.json)

Separately, the video evidence viewer now keeps its voice and timing
tabs on saved reports that include the clip-boundary face check. The
viewer regression verifies both recorded audio ranges and their
playback links in that report version.

## Independent replay

The file audit verified all 27,800 native picture timestamps and 156
original anchor frames and aligned crops associated with the added
proposals. The common test manifest records 278 submitted files with
556 file bindings. A second reader uses labeled frame coordinates
to reproduce the event counts. The repository contains source
commitments and measured positions. Original assets, rendered crops
and intermediate descriptors stay in the local test directory.
[Native audit](evaluation-panels/video-local-face-intervals-native-2026-11-16.json) ·
[Shared benchmark and event counts](evaluation-panels/video-local-face-intervals-review-2026-11-16.json.gz)

Use FFmpeg 7.0.2 and the pinned backend weights to recreate the media
on your computer. The prepared output and downloaded files should
be immediate children of the same review directory.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
receipt="$panels/video-local-face-intervals-2026-11-16.json.gz"
roster="$panels/video-local-face-intervals-sources-2026-11-16.json"
runner=backend/.venv/bin/python
"$runner" -B -m scripts.video_face_temporal_panel acquire "$roster" "$run/held-source"
"$runner" -B -m scripts.video_face_local_panel "$roster" "$run/held-source" "$run/held-qualified" \
  --purpose held --picks $(seq 0 19)
python3 -B -S -m scripts.video_face_local_fit verify "$receipt"
python3 -B -S -m scripts.video_face_local_replay verify-worker "$receipt" \
  "$panels/video-local-face-intervals-app-2026-11-16.json"
python3 -B -S -m scripts.video_face_local_replay export "$receipt" "$run/held-qualified" \
  "$run/held-source" "$run/review"
python3 -B -S -m scripts.independent_review count-events "$run/review/manifest.json" \
  "$run/review/truth.json" "$run/review/candidate.json" "$run/recount.json" --match .50 .50
```

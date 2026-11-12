# Mouth appearance at repeated poses in short retalking edits

A comparison of similarly posed mouths localized one of 32 GRID redraws and none of 20 CREMA-D redraws. It missed all seven redraws with source-compatible mouth motion and an unchanged soundtrack. The same edge comparison using nearby pictures without pose matching localized two GRID redraws. The pose-matched check failed the frozen criteria and stays outside the app. [Aggregate and source replay](evaluation-panels/video-mouth-pose-recurrence-2026-11-12.json)

## Frozen comparison

The trial reuses the exact native renderer and delivery bank from the earlier mouth-motion comparison. Eight GRID development recordings from four known source components set the two review points. The eight held GRID recordings from four separate components supply 32 neural edits and 56 matched ordinary views. The CREMA-D transfer supplies 20 neural edits and 40 ordinary views from ten separate source groups. Eight independent H.264 re-encodes and four genuine-picture hard cuts complete the 160 held views. These sources had earlier project exposure. The renderer bank preserves the original soundtrack and known edit bounds, with exact picture and pixel commitments. [Source roles and construction](video-mouth-jaw-transfer-2026-11-11.md) · [GRID source terms](https://zenodo.org/records/3625687) · [CREMA-D source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)

The seven source-compatible GRID redraws were previously qualified by visible mouth motion on the source clock. The separate replay here verified the unchanged PCM and actual picture edits. Achieved phoneme agreement and perceived synchrony remain unmeasured. [Visible movement verification](grid-mouth-motion-v158-result.md)

LIPINC-V2 compares adjacent mouth frames and more distant frames with similar mouth openings. SAVe separately tests lip-specific visual appearance. This trial implements a small single-file comparison inspired by those observations. It uses none of the research code, checkpoints or datasets from either study. The only weights are the existing face detector. [LIPINC-V2](https://arxiv.org/abs/2504.01470) · [SAVe](https://arxiv.org/abs/2603.25140) · [Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The comparison samples the same verified picture clocks and keeps the existing shot boundaries. It aligns each observed face by the eyes, centers on the mouth and measures the dark cavity in six cells. These values and the mouth position select the closest measured pose in each surrounding interval. The two reference pictures must be at least 160 ms from the target and close in pose to each other. The score compares the arrangement of edge directions inside the mouth, accounting for changes in the surrounding face. It measures how much more the middle differs from both flanks than the two flank pictures differ from each other. A second arm uses the nearest eligible references in time. Both arms keep one strongest 0.40–0.88-second range and require strict exceedance of their third-highest ordinary development response. The direct picture comparator keeps its original results and point. No original counterpart, soundtrack or edit mask enters inference.

A correct result needs at least 0.50 native time intersection over union with the authored interval. The frozen gate requires at least four new GRID locations across two groups, two new source-compatible locations, two new CREMA-D locations across two groups, and at least 75% positive and ordinary coverage in each bank. It permits up to four additional GRID control ranges, three CREMA-D control ranges, one re-encode range, and zero hard-cut ranges.

## Held outcomes

| Outcome | Direct picture check | Pose-matched references | Nearest-time references |
| --- | ---: | ---: | ---: |
| Assessed views | 160/160 | 160/160 | 160/160 |
| Correct GRID locations | 1/32 | 1/32 | 2/32 |
| Misplaced GRID ranges | 0/32 | 2/32 | 0/32 |
| Correct source-compatible GRID locations | 0/7 | 0/7 | 0/7 |
| GRID ordinary flags | 0/56 | 2/56 | 0/56 |
| Correct CREMA-D locations | 1/20 | 0/20 | 0/20 |
| CREMA-D ordinary flags | 8/40 | 6/40 | 1/40 |
| H.264 re-encode flags | 0/8 | 1/8 | 0/8 |
| Genuine hard-cut flags | 0/4 | 1/4 | 1/4 |

The pose-matched arm found a different GRID edit from the direct check. Its **1.120–1.840 s** range overlapped the authored **1.080–1.880 s** speech-conditioned redraw at 0.90 IoU. This redraw was driven by audio 320 ms earlier than the source window. The supporting native pictures were at **0.560**, **1.760**, and **2.160 s**. For one source-compatible redraw, the check selected **0.400–1.120 s**, overlapping only 0.040 s of the actual edit. Its ordinary flags included genuine content, blur, brightness and resampling. All six CREMA-D control ranges would add a new or differently located range relative to the direct check.

The 91.7-second CPU pass checked 5,498 sampled pictures. The audit verified 160 file digests and 480 decisions, re-decoded and matched 90 evidence crops, and reopened both actual picture streams for all seven source-compatible cases. It checked 1,050 native pictures, rebuilt 140 authored changed-pixel masks, and confirmed exact PCM equality for the seven pairs. All eight audio-only edits returned the same measurements as their unchanged picture counterpart. Per-source records and the replayed crops stay local.

## Reproduce

Use the retained mouth-jaw transfer bank and a fresh local `$run` folder.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_pose_recurrence_transfer "$phase" "$run" --prior "$prior" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_pose_recurrence_transfer "$run" --prior "$prior"
```

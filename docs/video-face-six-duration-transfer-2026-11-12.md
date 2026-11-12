# Six-picture face-change review on shorter and longer H.264 clips

The 12.5 Hz fallback localized all eight held six-picture swaps at each of the two durations. The current coarse and 8 Hz checks localized none. The fallback added zero ranges on the 40 matched ordinary deliveries and eight whole-picture cuts. [Held results and source replay](evaluation-panels/video-face-six-duration-transfer-2026-11-12.json)

## Frozen comparison

The previous six-picture trial qualified a 12.5 Hz fallback on 4.8-second H.264 clips. This study transfers the same calculation to exact **3.2-second** and **6.0-second** deliveries. Both use upright 640 × 360 H.264 4:2:0 at exact 25 fps and the same internal separation point. The 8 Hz method takes precedence when it produces a distinct range. If it measures a candidate and produces no distinct range, the fallback uses every second native picture, the unchanged scene and face guards, and the same duplicate check against the coarse range. [Previous held comparison and local app scope](video-face-six-picture-transfer-2026-11-12.md)

Two development and four held source groups have separate original links and known donor keys. The freeze excludes the 31 groups in the preceding source and delivery comparisons and any overlapping known links. Every chosen source has the publisher's head-rotation tag. Some footage had other project exposure. Separation covers the documented source links and donor keys. The archived publisher media supplied swap pictures from two generator families. The trial runs locally with the previously admitted checkpoints. [Original native preparation](video-native-inserts-v95-method.md) · [Source and rights review](video-common-source-preparation-v89.md)

Each held duration has eight swap deliveries across the four sources, 20 matched ordinary deliveries and four cuts. The short placements are **2.160–2.400 s** and **2.440–2.680 s**. The long placements are **2.480–2.720 s** and **3.560–3.800 s**. Each interval contains six exact native pictures. Original, tandem re-encode, local illumination, local blur and time-step controls share each target clock. The cut controls temporarily switch whole pictures between two genuine sources. Both placement and clip length vary, so the trial measures transfer under the stated conditions. The primary CRF 23 and tandem CRF 36 files are actual, silent H.264 outputs. The coarse comparison uses directly observed single-face features and the serving calculation, with the complete application tracking path outside this benchmark.

Each duration has a separate fixed gate requiring at least four new correct intervals across three source groups, at least 90% fallback coverage on swaps and ordinary controls, at most one additional ordinary range and zero additional cut ranges. A correct interval overlaps the authored half-open range by at least 0.50 IoU. The development pass verifies execution and supplies no new threshold.

## Held outcomes

| Held outcome | 3.2-second clips | 6.0-second clips |
| --- | ---: | ---: |
| Correct locations, current → combined | 0/8 → 8/8 | 0/8 → 8/8 |
| Source groups with additional correct locations | 4/4 | 4/4 |
| Additional ranges on matched ordinary views | 0/20 | 0/20 |
| Additional ranges on whole-picture cuts | 0/4 | 0/4 |
| Measured fallback views | 32/32 | 32/32 |

At each duration, four ranges match both endpoints exactly. The other four move both edges one native picture, or 40 ms, later. The median time IoU is 6/7 and the minimum is 5/7. The denser pass selected **2.160–2.400 s** on a shorter example and **2.480–2.720 s** on a longer one. The corresponding source pictures and crops are at **2.080**, **2.240**, and **2.400 s**, and **2.400**, **2.560**, and **2.720 s**, respectively.

Both durations cleared the frozen gate. The 157.3-second held CPU pass measured 5,920 sampled positions. These results cover exact 80- and 150-picture 25 fps H.264 deliveries from four held sources. Each observed range measures a brief appearance change for review.

The independent audit reopened 7,360 decoded H.264 pictures and matched 48 selected face crops. It reopened 840 native source pictures and rebuilt all 64 held MP4 files byte for byte, including the 40 swap or ordinary local edit intervals and eight whole-picture cuts. It also replayed the timing decisions and checked each advancement criterion.

A private runtime preview used the app's existing short-range measurement function with each proposed sample grid. It checked a known swap, a genuine original and a whole-picture cut at each length. Both passes completed together in under 3.3 seconds for all six clips, within the shared 12-second budget. The two swap intervals, raw scores and native frame bindings matched the held scorer. The four controls yielded no distinct range. This preview exercises measurement and timing before admitting the additional profiles to the evidence format.

## Reproduce

Use the retained native archive and the previous private 4.8-second transfer. Write all new outputs to a separate private folder.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze prepare development held report; do
  backend/.venv/bin/python -B -m scripts.video_face_six_duration_transfer "$stage" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_six_duration_transfer "$run"
```

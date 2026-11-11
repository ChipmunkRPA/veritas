# Denser timing for short face appearance changes

A 12 Hz refinement improved all 12 measured 0.75-second swap ranges on six held source recordings. It improved two and worsened ten of the 12 measured 1.50-second swap ranges. The appearance flags were identical, including zero false flags across 42 matched genuine and ordinary-edit controls. The timing change remains experimental. [Aggregate and exact commitments](evaluation-panels/video-face-native-boundary-2026-11-11.json)

## Frozen comparison

The six source groups were disjoint from the preceding texture evaluation. They were the six smallest eligible remaining native HD groups in the earlier confirmation role. All original decoded pictures differed across time. All six source recordings carried the publisher's “head rotation” tag. The initial report searched for a spelling without the space and missed those labels. The source set supplies two publisher-provenance swap families with two durations each and matching original, lighting, blur and time-step controls. The archived sources carry a publisher CC BY-SA 4.0 grant with documented attribution gaps. The same 24 positive views and 42 controls entered both arms. The source substitution used the verified native RGB and exact presentation times inside and outside each known half-open range. [Source permission review](video-common-source-preparation-v89.md) · [Native insert construction](video-native-inserts-v95-method.md)

The comparison uses the existing face appearance window calculation and decision point on a single observed face. It does not reproduce the full app tracker and fallback. For each flagged window, the candidate derives appearance prototypes from measured faces before, during and after the coarse interval. At 12 Hz, it searches within 0.375 seconds of each coarse edge and requires two consecutive samples on both sides of both transitions. The observed face boxes must overlap and the sampling must remain continuous at each transition. When these checks succeed, the first observed sample in the new state defines each refined boundary. The original coarse range remains the fallback. The only weights were the commercially licensed face detector and appearance representation used by the comparator. [Previous local comparison](video-face-texture-return-2026-11-11.md) · [June 2026 temporal localization study](https://arxiv.org/html/2606.07090v1)

| Known edit duration | Timing measure | Current 4 Hz window | With 12 Hz refinement |
| --- | --- | ---: | ---: |
| 0.75 s | Overlap IoU at least 0.75 | 2/12 | 12/12 |
| 0.75 s | Median IoU | 0.667 | 0.957 |
| 0.75 s | Median absolute onset and end errors | 0.133 s and 0.167 s | 0.033 s and 0 s |
| 1.50 s | Overlap IoU at least 0.75 | 10/12 | 12/12 |
| 1.50 s | Median IoU | 0.957 | 0.915 |
| 1.50 s | Median absolute onset and end errors | 0.033 s and 0.033 s | 0.067 s and 0.067 s |

All 24 swaps had a flag and at least 0.50 IoU with both methods. All 66 views had a measured current appearance window. The six source groups represent the independent content units. The repeated durations and swap variants share underlying footage.

## A criterion to test prospectively

All 12 true 0.75-second edits had coarse ranges from 0.720 to 0.760 seconds. Two true 1.50-second edits had coarse ranges of 1.000 seconds, and ten had coarse ranges of 1.500 seconds. The proposed cutoff uses an observable property of the coarse range. Applying refinement only when the coarse range is at most 1.000 second would have kept 14 gains and avoided ten losses in this completed test. This subgroup was examined after the results. Freeze the rule before a new source-disjoint evaluation that includes head rotation, ordinary delivery and longer edits with short apparent ranges.

For one larger timing gain, the known range was **2.280–3.760 seconds**. The current window was **2.480–3.480 seconds** and the refinement selected **2.320–3.760 seconds**. Independent command replay verified the clock and nine exact PNGs across the known, current and refined exhibits. The before, during and after pictures in this case were at **2.240, 3.000 and 3.760 seconds** for all three displays. Private media, source IDs and individual result rows stay local.

The CPU run completed 1,603 native RGB inputs in 213.0 seconds. Peak worker RSS was about 333 MB. Independent replay reproduced interval arithmetic, timing summaries, counts and source anchor roles. Four authored checks exercised an exact synthetic transition, an incomplete transition, a constant sequence and a range below the current decision point. This experiment changes no serving code or confidence contributions.

## Local replay

Run from the repository root with the exact archived inputs, the previous texture run directory and a new output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_face_native_boundary freeze "$run_dir" --media "$archive/video-native-inserts-v95" --previous "$texture_run_dir"
backend/.venv/bin/python -B -m scripts.video_face_native_boundary run "$run_dir" --media "$archive/video-native-inserts-v95"
backend/.venv/bin/python -B -m scripts.video_face_native_boundary readout "$run_dir"
```

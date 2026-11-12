# Replication of the shortest face appearance ranges

The first comparison found ten 0.375-second edits that the current appearance method missed. Its fitted cutoff also raised many false flags with negative margins. A post-hoc reading of that first panel found that requiring a positive margin kept the shortest hits and removed its rank false flags. This replication froze the positive-margin rule before scoring a separate source panel. [First comparison](video-face-return-transfer-2026-11-11.md) · [Replication freeze](evaluation-panels/video-short-return-replication-freeze-2026-11-11.json)

## Frozen rule

The current check keeps every existing flag and its exact endpoints. When it returns no flag, the additive cue can use the top guarded rank range only if its separation margin is strictly positive and its half-open native duration is below 0.50 seconds. The margin is positive when the core-to-flank separation exceeds within-period and between-flank variation. A scene check rejects candidates when textured regions outside the face show a large structural discontinuity at a boundary. The search uses the same fixed dense windows as the first trial. A top result outside the new range limit leaves the current outcome in place. Unavailable observations remain separate, and the rule supplies no identity or authenticity verdict.

The advancement criteria were six additional correct 0.375-second intervals across at least three sources, no more than two new ordinary false flags, no new cut false flags and every incumbent flag unchanged. The correct overlap criterion remains 0.50 intersection-over-union. The held comparison also reports the 0.50- and 0.75-second swaps to show how the additive scope affects the longer cases.

## Source panel

Six 30 Hz RTFS groups have distinct known source links and donor keys from the first study and five earlier localization rosters. They provide 36 short swaps across two publisher families and 60 matched genuine, light, blur and time-step controls. Two separate camera sources provide six unchanged, chroma-converted and recompressed deliveries. The 12 legitimate source-cut views switch between two genuine recordings in both directions for 0.50 and 0.75 seconds. They reuse the held RTFS originals with exact shared dimensions and native clocks. The clips had prior project-level whole-clip or app exposure. The new rule was frozen before these localization results. [Native data construction](video-native-inserts-v95-method.md) · [Source rights and known identity-linkage limits](video-common-source-preparation-v89.md) · [Camera terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA)

## Results

| Held cohort | Views | Current | With additive cue |
| --- | ---: | ---: | ---: |
| 0.375s swaps localized | 12 | 0 | 10 |
| 0.50s swaps localized | 12 | 10 | 10 |
| 0.75s swaps localized | 12 | 10 | 10 |
| Ordinary RTFS false flags | 60 | 0 | 1 |
| Genuine camera false flags | 6 | 0 | 3 |
| Legitimate source-cut false flags | 12 | 8 | 8 |

The ten added shortest hits came from five independent source groups. Each had 0.833 intersection-over-union and a 33-millisecond error at each endpoint. All 28 incumbent flags kept their original ranges. The four new ordinary false flags came from two sources, including three deliveries of one camera clip for which current appearance was unassessed. The cue missed its frozen two-view false-flag limit. [Exact aggregate](evaluation-panels/video-short-return-replication-2026-11-11.json)

All added true intervals had margins from 0.432 to 0.642. The four false intervals had margins from 0.006 to 0.027. These observed ranges support a separate prospective test with a positive margin above zero. The frozen rule remains out of the app. Six swap views were unassessed by both paths, and two legitimate-cut views also remained unassessed.

The CPU run verified 2,475 native pictures and clocks in 437 seconds with 326 MiB peak memory. A separate standard-library audit reproduced all 114 outcomes, the 28 preserved flags and the 42 newly selected frame anchors. Exact private before, during and after crops were retained for three added detections from different sources. Eight tests cover precedence, positive separation, the strict duration bound and unavailable results.

## Reproduce

Use the retained archive and the six private localization study directories. Only aggregate counts and commitments are published.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_replication freeze "$run_dir" \
  --first "$face_return_run" --prior "$texture_run" --prior "$boundary_run" \
  --prior "$short_boundary_run" --prior "$landmark_run" --prior "$chroma_run"
backend/.venv/bin/python -B -m scripts.video_short_return_replication run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_short_return_replication readout "$run_dir"
```

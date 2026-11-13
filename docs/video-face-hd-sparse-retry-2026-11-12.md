# Guarded face review after a sparse result on 720p30 video

The existing 720p30 check uses sparse samples for context and reports only a guarded dense range. On the previous panel, six edited files returned a sparse range outside the reporting scope and never received the denser check. This comparison tests a dense retry on that exact disposition. [Prior 720p30 result](video-face-two-returns-hd-2026-11-12.md)

## Frozen test

The trial uses eight already archived recordings with original exact 30 fps clocks. Their known original links and donor keys are separate from each other and from the three previous 360p and 720p short-face panels used to motivate the retry. They had other project exposure. The paired edit recipes were new. Every delivery contains 120 upright 1280 × 720 H.264 MP4 pictures with 8-bit 4:2:0 and exactly 30 fps. Every output picture takes the source picture at the same native time, with spatial downsampling where needed. [Source and rights](video-common-source-preparation-v89.md)

The two known swaps are eight and ten native pictures from different publisher-rendered families. They cover **2.033–2.300 s** and **2.767–3.100 s**. Each group contributes an encoded dual-swap file, its tandem recompression, an unchanged original, its tandem recompression, two ordinary same-source picture jumps, and two cross-recording whole-picture cuts. The genuine cut counterpart uses the highest border luminance correlation from a fixed preview picture among the other held hosts. This choice uses no face scores.

The comparator runs the actual Full probe, decode, shot, track, face and detect stages. The retry runs only when the existing sparse pass would have published a range under its generic rules and the 720p30 app would have withheld it. It uses the unchanged dense point, background guard, existing 12-second limit and exact native 15 Hz grid. If that dense pass reports a range, its measurements can supply the existing second disjoint check. Previously published ranges are preserved. The check measures local face changes and carries zero AI-confidence weight.

A useful range needs at least half temporal intersection over union with a known edit. The frozen gate requires three additional correct ranges from at least two groups, at most one added alert on original or recompressed ordinary footage, zero added cut or jump alerts and at most one misplaced added range on edited footage.

## Held results

| Condition | Files | Useful ranges before | Useful ranges with retry | Retry invoked |
| --- | ---: | ---: | ---: | ---: |
| Two swaps, first encode | 8 | 7/16 | 7/16 | 0 |
| Two swaps, tandem compression | 8 | 7/16 | 7/16 | 0 |
| Genuine original | 8 | 0 alerts | 0 alerts | 0 |
| Tandem original | 8 | 0 alerts | 0 alerts | 0 |
| Same-source jumps | 8 | 0 alerts | 0 alerts | 0 |
| Cross-recording cuts | 8 | 0 alerts | 0 alerts | 0 |

The sparse-only disposition did not recur in any of the 48 files. Fourteen of the 16 edited files already had a dense brief range, and two lacked suitable same-shot context. All 16 showed some face-review evidence from the incumbent path, with 14 of 32 authored edits usefully localized. No held clip tested the retry itself, so the absence of added ordinary alerts gives no false-alarm estimate for that branch. The retry stays outside the app.

The first preparation stopped before scoring when a tandem encode referenced the wrong scratch filename. Preparation restarted with a new freeze after correcting the path. The source roster, edit timing, candidate and gate were identical. The held Full comparison took 298.8 seconds and reported 5.46 GiB peak RSS. Independent replay verified all 5,760 native pictures and regenerated 24 swap, tandem and cross-recording cut deliveries byte for byte. [Held results and native readback](evaluation-panels/video-face-hd-sparse-retry-2026-11-12.json)

## Previously observed misses

A separate diagnostic reran the six clips that originally motivated the retry. It selected ten known swap intervals across two source groups, including both intervals on four dual-swap deliveries and one interval on each of two illumination-and-swap controls. Every selected interval matched both authored endpoints. A separate source readback matched 30 exact crops and recomputed 20 guarded background boundaries. One recovered range covers **3.000–3.267 s**, with before, during and after at frames **88, 94 and 98**. These were the already-known sparse-only cases. They contain no negative example that reaches the retry and supply no independent evidence for release.

## Replay

Use the retained source archive and a fresh private directory. The comparison binds the original serving adapter revision before the trial. Source identities, media and individual records remain local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_hd_sparse_retry "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_hd_sparse_retry "$run"
```

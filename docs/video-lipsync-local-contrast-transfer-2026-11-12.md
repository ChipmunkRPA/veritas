# Paired speech and mouth features for short timing slips

The experimental timing check measured all 160 held views and correctly localized two of 64 known signed audio slips. It also flagged 17 of 96 controls. The current timing check measured 150 views and returned no correct ranges or control flags. The experimental check missed the frozen advancement criteria and stays outside the app. [Aggregate and native replay](evaluation-panels/video-lipsync-local-contrast-transfer-2026-11-12.json)

## Source and fixed method

This test uses 32 genuine, licensed GRID recordings with exact native picture timestamps and verified 16 kHz audio. Twelve clips from six known source components fit the comparison, four clips from two different components set its review point, and 16 clips from eight other components were held. These recordings had earlier project exposure. The held recordings supplied ordinary picture edits for the previous paired H.264 test and were outside the fitting and point rosters for this timing study. Each has 75 native pictures at exactly 25 fps and 47,648 verified source samples. [GRID publisher and CC BY 4.0 terms](https://zenodo.org/records/3625687) · [Previously verified source groups](video-reference-h264-grid-reserve-2026-11-11.md)

Recent work tests paired audio and lip features for local audiovisual consistency. This trial independently learns a small alignment score from correctly timed pairs and from audio shifted within the same utterance. The fit uses 519 aligned and 2,700 shifted speaking ticks. The visual stream measures normalized mouth texture and adjacent changes. The acoustic stream measures the relative spectral shape across ten bands and adjacent changes. A regularized logistic fit uses only interactions between the streams. The existing admitted face detector places the crop. No other pretrained weights, transcripts or outside reference videos enter the comparison. [Lip-focused audiovisual consistency](https://arxiv.org/abs/2610.30872) · [CVPR 2026 cross-domain study](https://arxiv.org/abs/2511.17181) · [Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The candidate compares a range of 480–960 ms with its two 400 ms flanks. It requires a different lag to score better inside the proposed range, while the same reference lag scores better on both flanks. Lag comparisons share identical measured picture ticks. Each supported lag needs active audio in at least 35% of those ticks. The largest ordinary calibration response plus 0.025 set the internal review point at 0.2610859752 before held scoring. The current comparator runs the existing one-second timing calculation with the same directly observed mouth crops. This uses the existing timing and speech attribution arithmetic, with a single observed face in place of full application track selection. [Earlier short timing comparison](video-lipsync-local-delay-transfer-2026-11-11.md)

Each held recording supplies four signed slips, ±200 ms for 640 ms and ±280 ms for 880 ms. All preserve the pictures and source samples outside the authored interval. A 10 ms blend applies at each interval boundary. Matched controls use the original soundtrack, local gain, local low-pass filtering, mild local noise, and two constant whole-track offsets. A correct result needs at least 0.40 temporal intersection over union and an estimated displacement within 80 ms of the actual signed edit. The frozen gate requires four added correct ranges across three or more source groups, preservation of all current correct ranges, at least 75% assessed coverage, and at most one added control flag.

## Held results

| Outcome | Current timing check | Paired-feature candidate |
| --- | ---: | ---: |
| Assessed views | 150/160 | 160/160 |
| Correct signed local slips | 0/64 | 2/64 |
| Flagged edits with wrong location or displacement | 0/64 | 14/64 |
| Control flags | 0/96 | 17/96 |
| Recorded soundtrack flags | 0/16 | 3/16 |

The two successful review intervals came from separate known source groups. A **1.200–1.680 s** range overlaps the authored **1.200–1.840 s** edit at 0.75 IoU, with **+280 ms** estimated against **+200 ms** authored. A **1.280–1.760 s** range overlaps the authored **1.040–1.920 s** edit at 0.545 IoU, with **−200 ms** estimated against **−280 ms** authored. Both signed estimates are at the outer edge of the allowed error. Ordinary flags included three gain changes, five low-pass edits, one noise control and five constant-offset controls in addition to the three recorded views. Increased measurement coverage did not make the estimate reliable on these sources.

A separate reviewer decoded all 1,200 held pictures from the source files and matched their timestamps and pixel hashes. It reconstructed all 160 PCM deliveries, replayed every candidate score, and checked 480 source-bound mouth crops and 960 exact PCM slices. The held pass took 10.8 seconds on one local CPU thread. This study covers scripted short slips in one corpus. Modern synchronized redraws, varied delivery codecs and spontaneous multi-speaker speech remain outside its tested scope. Source media, per-actor records and the fitted matrix stay local.

## Reproduce

Use the verified private GRID plan from the earlier video evaluations and a new local `$run` folder.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze fit point calibrate held report; do
  backend/.venv/bin/python -B -m scripts.video_lipsync_local_contrast_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_lipsync_local_contrast_transfer "$run"
```

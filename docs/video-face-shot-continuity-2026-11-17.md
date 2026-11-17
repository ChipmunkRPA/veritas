# Temporary face changes within one shot

The brief face-appearance fallback measures every other native frame in
upright 640 × 360 H.264 MP4 clips with exactly 100 pictures at 25 fps.
It compares the proposed period with four sampled faces before and after,
and measures visual similarity in areas outside the face at both ends.
The measured change can span 8–44 frames. Its boundaries use the two-frame
sampling grid.

The strongest proposed range now also checks every adjacent original
picture from the first supporting frame through the last. The existing
shot-change comparison uses both color changes and histogram overlap.
For this local check it can detect shots shorter than half a second.
If the supporting pictures cross an abrupt shot change, the app records
that reason and withholds the proposal. A cut before the first supporting
picture is outside the comparison.

The face comparison uses the previously frozen operating point and the
pinned MIT face locator and Apache-2.0 appearance weights. Consecutive
picture comparisons are also used in research on locating short
manipulations inside longer recordings.
[Original comparison and source panel](video-local-face-intervals-2026-11-16.md) ·
[Consecutive-picture research](https://arxiv.org/abs/2606.07090) ·
[Face locator license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance weights license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Replay the known cut errors

The native shot check removes all four cut alerts from the previous
worker replay and preserves all three additional correctly located
swaps. Against the brief scan on all 278 retained deliveries, direct and
recompressed useful ranges rise from 12 to 60 of 80 staged edits. The
four original alerts among 198 ordinary deliveries still come from the
brief scan. The extra shot check is limited to the new fallback proposals.

The held transfer is formed from 16 source groups reserved in the
publisher acquisition. Normalized source video URLs, known donor keys,
and publisher asset hashes are disjoint from the earlier comparison
panels. The panel has two staged face-swap families plus originals,
exposure changes, face blurs, jumps within a recording, and cuts to a
different recording. Each input has a direct and a strongly recompressed
delivery. Counts are paired across encodes of related clips. A useful
location has at least 50% overlap of the temporal union with the edit.
[Publisher source preparation and permissions](video-common-source-preparation-v89.md)

All 224 deliveries completed the native comparison. The shot check
rejected six new alerts on ordinary cuts, including both deliveries of
each of three distinct cuts. It preserved every useful location. The
brief scan and fallback together had comparable face inputs in 212
deliveries.

| Delivery | Useful ranges with brief scan | Useful ranges with fallback and shot check | Ordinary alerts, brief → with shot check |
| --- | ---: | ---: | ---: |
| Direct | 7/32 | 30/32 | 3/80 → 3/80 |
| Strong recompression | 7/32 | 27/32 | 1/80 → 1/80 |
| Both | 14/64 | 57/64 | 4/160 → 4/160 |

These counts compare the new fallback with the brief scan alone. The
app also runs the broader face appearance and boundary checks.

## Results through the worker

The isolated worker replayed all 62 proposals above the frozen point
with the broader face checks active. It added two useful locations
from two distinct source groups. Both matched the labeled frame
boundaries exactly, spanning 1.120–1.520 s and 1.440–1.840 s. The
worker rejected all six cut proposals and added zero ordinary or
misplaced edit alerts. Both additions were in direct encodes.
[Fresh app replay](evaluation-panels/video-face-shot-continuity-app-2026-11-17.json)

The repeated worker test on the seven earlier gains and cut errors
recovered three additional locations and rejected four ordinary
cuts. The recompressed example spanned 0.480–1.680 s and its browser
playback sought the verified source frame at 1.040 s.
[Earlier worker inputs](evaluation-panels/video-face-shot-continuity-prior-input-2026-11-17.json) ·
[Earlier worker replay](evaluation-panels/video-face-shot-continuity-prior-app-2026-11-17.json)

The two worker series completed 71 simulated prepaid analyses,
including the same recorded input with the local option off and on.
The comparison recorded zero jobs before simulated payment.

The full local configuration also recovered the recompressed example
and withheld its paired cut proposal. Three simulated paid analyses
covered the edit, control, and rollback. Both on/off runs returned the
same customer assessment on that file.
[Full local inputs](evaluation-panels/video-face-shot-continuity-full-input-2026-11-17.json) ·
[Full local replay](evaluation-panels/video-face-shot-continuity-full-app-2026-11-17.json)

## Evidence shown after upload

The ordinary video evidence view shows the estimated time range, three
original frames, and their face crops. The PDF and JSON carry the same
range and positions. With the submitted video, the signed-package verifier
checks the frame clock, the crops, the background similarity readings,
and the absence of an abrupt shot change within the supporting pictures.
This evidence describes a change in appearance and carries zero weight
in overall AI confidence.

## Reproduce the comparison

The source roster binds the publisher revision, original assets and
hashes. The replay binds the delivered clips, their labeled frame
ranges, and each recorded shot decision. The publisher provides the
video assets under its dataset terms.
[Source roster](evaluation-panels/video-face-shot-continuity-sources-2026-11-17.json) ·
[Held comparison](evaluation-panels/video-face-shot-continuity-held-2026-11-17.json)

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run=$(mktemp -d)
panels=docs/evaluation-panels
roster="$panels/video-face-shot-continuity-sources-2026-11-17.json"
held="$panels/video-face-shot-continuity-held-2026-11-17.json"
frozen="$panels/video-local-face-intervals-2026-11-16.json.gz"
runner=backend/.venv/bin/python

"$runner" -B -m scripts.video_face_temporal_panel acquire "$roster" "$run/sources"
"$runner" -B -m scripts.video_face_local_panel "$roster" "$run/sources" "$run/prepared" \
  --purpose held --picks $(seq 0 15)
"$runner" -B -m scripts.video_face_local_measure "$run/prepared" "$run/measurements.jsonl"
"$runner" -B -m scripts.video_face_shot_guard scan "$run/prepared" "$frozen" "$run/fresh.json" \
  --measurements "$run/measurements.jsonl"
python3 -B -S -m scripts.video_face_shot_guard verify "$held"
```

The paid-flow replay uses simulated Stripe events and the serving
face-detection, tracking, and appearance checks. Build the frontend
before running the browser test.

```sh
"$runner" -B -m scripts.video_face_shot_guard_app "$held" "$run/prepared" "$run/app"
python3 -B -S -m scripts.video_face_shot_guard verify "$held" \
  --worker "$panels/video-face-shot-continuity-app-2026-11-17.json"
```

# Paired scene cuts and incumbent face-review false flags

The prior video comparison found 27 ordinary views with an incumbent face-review flag, including eight exact source-cut views. The added short-range cue raised no false flags in that panel. This study freezes a conservative cut-only rule and tests it on new source groups. [Prior study](video-short-return-secondary-2026-11-11.md) · [Frozen comparison](evaluation-panels/video-incumbent-cut-guard-transfer-freeze-2026-11-11.json)

## Frozen rule

The incumbent comparator uses the Full profile's uniform whole-clip frame indices with native face triplets and the existing appearance point. It uses only one observed face per sampled picture and one research track. The local application has its own track selection and fallback. Both experimental arms share the same incumbent windows, native inputs, and flag point.

For each incumbent flag, inspect adjacent native pictures within 0.25 seconds of each endpoint. Use the current shot detector's HSV content threshold of 27 and histogram dissimilarity threshold of 0.35. A candidate pair must meet or exceed both thresholds at entry and exit. The first and last middle pictures must be below both thresholds against each other, as must the two surrounding pictures. Require at least 0.50 intersection-over-union with the incumbent interval. The pair with the largest overlap supplies the cut observation. Missing scene or timestamp support leaves the incumbent decision in place. The thresholds were set before the held comparison and were inherited from the app's existing shot measurements.

The gate requires exact preservation of every correctly localized swap and suppression of at least four incumbent paired-cut flags across two or more source pairs. The suppressed flags must represent at least half of the incumbent paired-cut flags, with at least six incumbent flags in the denominator. Single-cut flags must be preserved. The study records genuine illumination, blur, time-step, original and camera controls separately.

## Source panel

The six new RTFS source and donor groups have complete 1080p, 25 Hz clocks. Selection excluded all known original links, donor keys and indexes from nine previous localization rosters. It prioritized head rotation and gesticulation tags and then smaller archived byte size. Some of the source clips had prior project-level whole-clip exposure. Each group provides six short face-swap views and ten ordinary views. The nominal 0.375-, 0.50- and 0.75-second interventions use exact half-open intervals from the native clock.

Three matched pairs of genuine originals provide 18 A/B/A edits and six single hard cuts. Each pair uses both directions at identical native picture times and dimensions. The remaining independent VCD2 camera group has three matched deliveries. This is one recording, so its results describe only those three deliveries. [Native construction](video-native-inserts-v95-method.md) · [Original and donor rights](video-common-source-preparation-v89.md) · [Camera terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Matched delivery](video-vcd2-delivery-v206-method.md)

## Reproduce

Supply the retained private archive and the eight earlier directories bound by the previous secondary-range freeze. The independent reviewer checks the complete decision arithmetic and reconstructs each selected cut measurement from the original source pictures. Only aggregate counts and commitments are published.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_incumbent_cut_guard_transfer freeze "$run_dir" \
  --secondary-run "$secondary_run" --prior "$margin_run" --prior "$replication_run" \
  --prior "$face_return_run" --prior "$texture_run" --prior "$boundary_run" \
  --prior "$short_boundary_run" --prior "$landmark_run" --prior "$chroma_run" \
  --texture-external "$texture_run/external-freeze.json"
backend/.venv/bin/python -B -m scripts.video_incumbent_cut_guard_transfer run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_incumbent_cut_guard_transfer readout "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_incumbent_cut_guard "$run_dir" \
  --archive "$archive" --receipts "$native_receipts"
```

## Results

| Views | Count | Incumbent | Paired-cut rule |
| --- | ---: | --- | --- |
| Short swaps with known boundaries | 36 | 20 correct ranges, 4 wrong ranges, 12 measured without a flag | Same 36 outcomes |
| Genuine paired cuts | 18 | 12 false face-change ranges | 6 false face-change ranges |
| Genuine single cuts | 6 | 0 flags | 0 flags |
| Ordinary originals, blur, illumination and time-step controls | 60 | 0 flags | 0 flags |
| Matched camera deliveries | 3 | 0 flags | 0 flags |

The six qualifying ranges matched both exact authored cut boundaries. They covered all three source pairs. Among the 12 incumbent paired-cut flags, all 12 authored edge pairs passed the scene cut and return conditions, nine passed endpoint proximity, and six passed the overlap check. The independent reviewer rederived every decision and replayed all six selected transitions using 48 distinct original pictures. This comparison supports the bounded paired-cut case. The other ordinary and camera controls contain no incumbent false flags, so they provide no estimate of how many such false flags the rule could remove. Several true swaps still received either a misplaced range or no range. [Aggregate results](evaluation-panels/video-incumbent-cut-guard-transfer-2026-11-11.json)

## Local application transfer

All six research-qualified paired-cut views were losslessly assembled into playable videos and checked against all 150 native RGB pictures per clip. Each ran through the local Full-profile upload, checkout and worker with the actual shot, face-tracking, and incumbent face-appearance checks enabled in an isolated database. A simulated Stripe test checkout verified that an unpaid event queued no analysis and a paid event queued exactly one.

All six incumbent checks ran and produced no face-change range. The application's 640 px shot stage found both exact cuts on five clips and the exact entry cut on the remaining clip. The tracker starts a new track at each detected shot boundary, and the appearance check compares samples only within a track. The frozen comparison's single research track could span cuts that split the application's tracks. The paired-cut rule improved the research comparator. With zero incumbent flags in these six application cases, it remains an evaluation-only result.

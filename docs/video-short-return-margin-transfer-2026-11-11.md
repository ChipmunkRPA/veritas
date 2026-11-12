# Source-separated test of a higher short face-change margin

The prior zero-margin study found ten additional face swaps in 12 nominal 0.375-second views. It also produced four additional ordinary flags across two recordings. On that panel, added correct ranges scored 0.432–0.642 and ordinary false flags scored 0.006–0.027. The follow-up fixed the raw separation margin at **0.10** before examining the new sources and kept the same guarded rank search. The margin is a detector operating point for this experiment. [Prior replication](video-short-return-replication-2026-11-11.md) · [Frozen follow-up](evaluation-panels/video-short-return-margin-transfer-freeze-2026-11-11.json)

## Frozen method

Retain every incumbent flag and both endpoints. When the incumbent does not flag, consider the single highest guarded rank interval. Require a margin strictly greater than 0.10 and a positive half-open duration strictly below 0.50 seconds. If the highest interval fails either condition, retain the no-flag or unassessed outcome. The search compares the middle face group with its two neighboring groups. The scene guard checks for structural discontinuity outside the face at both edges.

The RTFS panel uses six distinct source and donor groups from the archived localization corpus. The freeze excludes the prior seven localization rosters by source URL, known donor keys and group index. It selects the smallest remaining complete 1080p, 30 Hz groups. Each group supplies two swap methods and three matching ordinary interventions involving illumination, blur and time-step changes, with an unchanged original. Some of these sources had earlier whole-clip project exposure. The experiment measures fresh source groups for this localization method and does not establish disjoint model-training data. [Native construction](video-native-inserts-v95-method.md) · [Source and rights preparation](video-common-source-preparation-v89.md)

The shorter nominal durations are 0.375, 0.50 and 0.75 seconds centered at 3 seconds. The 0.375-second intervention is also placed at centers 35/12 and 73/24 seconds. Each edge advances to the next exact native frame at or after the nominal boundary. The three shortest placements use native half-open intervals `[85,96)`, `[82,94)` and `[86,97)` in picture indices. They cover two distinct 8 Hz rank sample alignments. The nominal and later placements share three rank sample indices, with different native ground truth and incumbent triplet inputs. This yields 60 known swap views and 96 ordinary RTFS views. The placements and transforms within each group are paired variants of one source.

Four previously prepared VCD2 camera recordings have source hashes disjoint from the earlier localization and coverage rosters. One uses the v290 raw matched delivery and three use the v206 common encode. Each supplies an unchanged, chroma-converted and recompressed view. Participant keys are unavailable. Twelve additional ordinary source-cut views pair the held RTFS originals in both directions for exact 0.50- and 0.75-second inserts. Matching geometry and presentation clocks constrain all of these comparisons. [Camera release rights](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Common delivery](video-vcd2-delivery-v206-method.md) · [Matched camera controls](video-vcd2-matched-controls-v290-method.md)

A useful interval has intersection-over-union of at least 0.50 with the known half-open native interval. Advancement requires six additional useful shortest intervals across at least three sources **at each of the three placements**, at most two additional ordinary flags, no additional source-cut flags, and exact preservation of every incumbent flag. Unassessed views stay in the denominators. An interval supports inspection of an appearance change and supplies no identity or authenticity determination or combined confidence weight.

## Results

| Held cohort | Views | Current | With frozen fallback |
| --- | ---: | ---: | ---: |
| 0.375 s swaps localized, nominal placement | 12 | 0 | 11 |
| 0.375 s swaps localized, earlier placement | 12 | 4 | 5 |
| 0.375 s swaps localized, later placement | 12 | 0 | 6 |
| 0.50 s swaps localized | 12 | 12 | 12 |
| 0.75 s swaps localized | 12 | 12 | 12 |
| Ordinary RTFS false flags | 96 | 0 | 0 |
| Genuine camera false flags | 12 | 3 | 3 |
| Ordinary source-cut false flags | 12 | 12 | 12 |

The fallback improved correct localization from **28/60 to 46/60**, with gains in six, one and four groups at the nominal, earlier and later placements. It added zero flags on 120 controls and preserved all 57 incumbent flags and endpoints. All 18 new correct ranges had 0.833 interval overlap and at most 33.3 milliseconds absolute error at either edge. The earlier-placement gain fell below the frozen advancement requirement. The tested fallback remains experimental. [Exact aggregate and model-free audit](evaluation-panels/video-short-return-margin-transfer-2026-11-11.json)

The audit separately inspected the same highest rank interval at 0.10 regardless of incumbent status. This post-hoc comparison found **36/36** shortest swaps. Eighteen had no incumbent flag, 14 had an incumbent flag at a wrong interval, and four already had a useful incumbent interval. Suppressing secondary ranges with at least 0.50 IoU against the incumbent removes those four overlaps and retains 32 useful ranges. The standalone rank cue flagged zero of all 120 controls, including the 15 that the incumbent flagged. Source-group counts and the suppressed duplicates are included in the public readout. A new prospective panel should test the separately labeled secondary range and its deduplication rule.

The local CPU run verified 2,895 decoded source pictures in 441 seconds and used 322 MiB peak memory. The standalone standard-library reviewer recomputed 180 outcomes, checked 720 native endpoint pairs and matched 54 selected picture anchors to exact clocks. Three added ranges from distinct source groups were independently decoded for before, during and after crops. The nine crops are private.

## Reproduce

With the retained private archive and prior run directories, freeze the inputs before calling the scorer. The prior list consists of the initial face-return study and the five earlier localization runs used by the zero-margin replication. The separate texture external freeze adds the v206 recordings previously used in that check to the camera exclusion roster. The model-free reviewer recomputes the decisions and verifies exact interval clocks against the supplied receipts. Only aggregate counts and commitments are published.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_margin_transfer freeze "$run_dir" \
  --replication "$replication_run" --prior "$face_return_run" --prior "$texture_run" \
  --prior "$boundary_run" --prior "$short_boundary_run" --prior "$landmark_run" \
  --prior "$chroma_run" --texture-external "$texture_run/external-freeze.json"
backend/.venv/bin/python -B -m scripts.video_short_return_margin_transfer run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_short_return_margin_transfer readout "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_short_return_margin "$run_dir" \
  --archive "$archive" --receipts "$native_receipts"
```

# Prospective distinct secondary face-change range

The preceding 0.10-margin trial added 18 correct shortest ranges without additional ordinary false flags. It missed the earlier-placement advancement requirement because its rule gave every incumbent flag precedence. A post-hoc comparison found that a distinct rank range could also point to the correct location when the incumbent flagged the wrong interval. This study freezes that routing on new sources. [Preceding trial](video-short-return-margin-transfer-2026-11-11.md) · [Prospective freeze](evaluation-panels/video-short-return-secondary-freeze-2026-11-11.json)

## Rule and source panel

Preserve every incumbent flag and its endpoints. Independently inspect the single highest same-shot rank interval. Keep it as a secondary range only if the raw separation margin is strictly above 0.10 and the exact half-open duration is below 0.50 seconds. If the interval overlaps an incumbent flag at 0.50 IoU or more, suppress it as a duplicate. An ordinary input counts as having additional false evidence if it receives a separate secondary range, including when the incumbent already flagged the input. Report ordinary edits, camera deliveries and genuine source cuts separately.

Six complete 1080p recordings at 25 Hz have tags for head rotation or gesticulation. Their known source links and donor keys are disjoint from the previous eight localization rosters. The six matched groups supply 60 labeled swap views and 96 controls, including illumination, blur and time-step changes at the same native intervals. The 0.375-second intervention appears at three prespecified centers. The nominal center also supplies 0.50- and 0.75-second interventions. Variants of one source stay together. Some sources were previously used for different whole-clip project tasks. [Matched native construction](video-native-inserts-v95-method.md)

Four v206 camera-original groups supply 12 ordinary camera deliveries. Their source byte hashes are disjoint from the previous localization and coverage panels. Participant identities are unavailable. Three exact-clock pairs of held RTFS originals supply 12 ordinary source-cut views. [Source and rights preparation](video-common-source-preparation-v89.md) · [Camera terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Camera delivery](video-vcd2-delivery-v206-method.md)

The gate requires at least 18 added correct shortest ranges in at least four source groups. Each of the two incumbent statuses, absent and wrong, must account for four or more additional correct intervals across at least two groups. The study allows two additional ordinary ranges and zero additional source-cut ranges. A correct interval requires at least 0.50 IoU against the exact labeled edit. A range is a face appearance observation for review and carries zero authorship-confidence weight.

## Results

| Held cohort | Views | Incumbent | With distinct secondary range |
| --- | ---: | ---: | ---: |
| 0.375 s swaps localized, nominal placement | 12 | 0 | 10 |
| 0.375 s swaps localized, earlier placement | 12 | 0 | 10 |
| 0.375 s swaps localized, later placement | 12 | 1 | 10 |
| 0.50 s swaps localized | 12 | 10 | 10 |
| 0.75 s swaps localized | 12 | 10 | 10 |
| Ordinary RTFS views with a false range | 96 | 16 | 16 |
| Ordinary camera views with a false range | 12 | 3 | 3 |
| Ordinary source-cut views with a false range | 12 | 8 | 8 |

The new routing met the frozen gate. It recovered **29 additional correct ranges**, raising the full result from **21/60 to 50/60** and shortest localizations from **1/36 to 30/36**. It added 13 useful ranges when the incumbent had none, across four groups, and 16 when the incumbent placed its range incorrectly, across five groups. It suppressed one duplicate and preserved all 64 incumbent flags and endpoints. The secondary range produced zero additional false flags across the 120 controls, including 27 with an incumbent false flag. The matched conditions share source groups. Ten swap views had neither a useful current nor a useful secondary interval. [Exact aggregate and model-free audit](evaluation-panels/video-short-return-secondary-transfer-2026-11-11.json)

Median IoU of the added correct ranges was 1.00. Median absolute endpoint error was zero and the largest was one native picture interval, 40 milliseconds. The CPU run verified 2,895 decoded pictures in 345 seconds with 345 MiB peak memory. The model-free reviewer independently checked the routing across 180 views, 608 native endpoint pairs, and 87 selected source anchors. The research scorer used exact native pictures assembled from matched inputs. Serving verification runs through the application and its own frame and evidence paths.

Exports containing the new range use `video-evidence-12`. Use the verifier from this release to validate the range, source-clock and playback coordinates, and committed source crops. The preceding evidence versions remain readable.

## Local serving verification

Enable the separate short-range cue with `VERITAS_VIDEO_FACE_REVIEW_MODE=appearance` and `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW=true`. The pilot covers upright native 1080p clips of 2–8 seconds at exact 25 or 30 Hz. It requires verified source timestamps and at most one observed face at each sampled position. The report shows the exact half-open secondary range and three original pictures and crops. The original confidence estimate and any incumbent range are retained.

The isolated Full-profile smoke used the real app face detection, tracking, appearance review and report pipeline on the pinned CPU weights. It limited the detector schedule to those paths. A stub simulated Stripe transport while the actual checkout and webhook logic ran. Checkout and an unpaid webhook queued zero analyses. A valid paid event queued one analysis, and repeated delivery of that paid event was idempotent.

The two positive smoke fixtures were selected using the serving sampler because its frame positions differ from the research comparator. Both rendered clips reproduced all 150 source pictures and their exact 25 Hz clocks. With the feature enabled, the absent-incumbent case produced **3.000–3.360 s** for a known **2.920–3.280 s** edit, with 0.636 IoU. The other case preserved the incumbent **2.840–3.360 s** and added **2.760–3.120 s**, the exact edit range. The disabled and enabled runs had identical incumbent aggregates and confidence decisions on eight inputs. Four matched originals and two overlapping swap controls produced zero supplemental ranges.

For each of the two emitted packets, the evidence API and JSON export matched, and all six original/crop PNG hashes and decoded pixels matched the uploaded source. The report PDF embedded the same six decoded images. The signed ZIP retained that PDF and evidence and passed independent replay of the source pixels, crops and timestamps. Scene similarity and appearance separation are measurements committed by the recorded detector run. The visible check and exported reports used model-neutral wording. [Aggregate app integration receipt](evaluation-panels/video-short-return-secondary-app-integration-2026-11-11.json)

## Reproduce

Use the retained archive and private freeze directories. The seven prior directories are the zero-margin replication and the six directories to which that replication binds. The separate texture external freeze supplies two additional previously used camera recordings. The local model-free reviewer recomputes the decisions and validates the exact source clocks using the native receipts.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_short_return_secondary_transfer freeze "$run_dir" \
  --margin-run "$margin_run" --prior "$replication_run" --prior "$face_return_run" \
  --prior "$texture_run" --prior "$boundary_run" --prior "$short_boundary_run" \
  --prior "$landmark_run" --prior "$chroma_run" --texture-external "$texture_run/external-freeze.json"
backend/.venv/bin/python -B -m scripts.video_short_return_secondary_transfer run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_short_return_secondary_transfer readout "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_short_return_secondary "$run_dir" \
  --archive "$archive" --receipts "$native_receipts"
```

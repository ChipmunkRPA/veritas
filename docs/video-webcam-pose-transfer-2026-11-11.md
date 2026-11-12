# Within-track geometry and webcam appearance flags

The local app previously flagged six ordinary delivery views from two genuine webcam recordings. They stayed within a tracked face and covered three deliveries of each recording. One recording showed a change in the ordering of detected nose and mouth points relative to the eye axis. In the other, visible eye spacing contracted while the detected nose shifted sideways. This test asks whether either pattern recurs in an independent camera cohort while preserving correct ranges on known face swaps. [Development observations](video-vcd2-remaining-webcams-v253-result.md) · [Earlier eye-width test](video-landmark-rerank-v259-result.md)

## Frozen comparison

Use the original top face-appearance range, its consecutive three-picture groups, and its exact source clock. The first candidate condition requires the nose and mouth to both project above the directed eye axis. The second requires that both landmarks still project below the eye axis, that eye spacing contract to at most 70% of the smaller flank median, and that the nose move at least 12% of face-box width in the same direction relative to both flank medians. In either case, at least two pictures per triplet and at least two thirds of the middle triplets must meet the condition. Each flank needs at least four pictures with normal landmark order and a median eye spacing of at least 20% of face-box width. The earlier eye-spacing-only comparison uses its original 60% point. The new fixed points came from development measurements and geometry, without held camera or swap measurements.

A qualifying range would carry explicit landmark uncertainty. These two-dimensional measurements describe visible geometry. They cannot identify the cause of a pose change or establish an occluding object. The existing interval provides the before, during and after pictures, and the candidate contributes zero weight to AI confidence. Combining appearance and motion cues is consistent with recent work on temporal feature variation. The candidate uses the project's existing commercially qualified face weights and does not reuse another paper's code or weights. [Datta et al., ICCV Workshops 2025](https://openaccess.thecvf.com/content/ICCV2025W/APAI/papers/Datta_PIA_Deepfake_Detection_Using_Phoneme-Temporal_and_Identity-Dynamic_Analysis_ICCVW_2025_paper.pdf) · [Weight terms](video-selected-landmark-reliability-2026-11-11.md)

The held camera cohort is 18 sealed Full-profile analyses from six distinct recordings, including unchanged, chroma and recompressed variants. The two camera groups already used in the previous detailed confound studies are excluded. The camera replay verifies the archive, previous selected packet, exact timestamp and decoded RGB of every selected native input, then reproduces its exact aligned crop. The native swap cohort consists of six distinct known original links and donor groups, with one untouched original and two matched swap methods per group. These sources have prior project evaluation exposure, and portrait attribution remains incomplete. This study uses their first outputs from the frozen production-tracked comparison. The current Full pipeline processes them with actual shot segmentation, tracking, native face detection and the incumbent face-appearance route. The experiment verifies the original source bytes, exact edit intervals and every selected frame commitment. [Camera terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Camera source records](video-vcd2-full-app-v202-method.md) · [Native source and rights review](video-common-source-preparation-v89.md)

A local pilot needs at least four genuine flags contextualized across at least two held camera recordings, no loss among correctly localized swaps, at least four correctly localized swaps across at least three native groups, and at least 90% geometry coverage among selected ranges. Correct localization means at least 0.50 temporal IoU between the incumbent's first and last supported centers and the native half-open labeled edit. Missing geometry keeps the previous decision. The candidate does not move an interval to another rank.

## Held webcam result

| Outcome | Views |
| --- | ---: |
| Measured without a selected appearance range | 14 |
| Incumbent unassessed | 1 |
| Measured with a false appearance range | 3 |
| False range contextualized by the previous eye-width rule | 0 |
| False range contextualized by either new geometric condition | 0 |

The three false ranges are matched deliveries of one genuine recording. All three had complete geometric support, and neither candidate condition occurred in any middle triplet. In a separate replay of the six development views, the new rule identified landmark inversion in three views of one recording. The other recording met the lateral-motion condition in six of ten triplets, one short of the frozen requirement. The older eye-width rule marked all six development views.

After scoring, inspection of three exact source pictures from the held failure showed the person lean to one side, tilt their head, and return to the center. The landmark orientation check did not capture that movement. This is one development example for a separate test of head translation and roll relative to stationary background references. The camera condition failed and the candidate stays experimental.

The three initial native worker attempts stopped before detector inference. The first omitted the isolated analysis identifier. The second used a content-store symlink where ingestion requires an ordinary staged upload. The third reached the detector stages without a model manager, so the relevant adapters performed no inference. The operational amendments supplied the identifier, used the existing upload function, and passed the installed model manager explicitly. The private run keeps every freeze version and each empty native journal. The held camera results, scoring rule, native inputs and gate stayed fixed.

## Paired swap preservation

The 18 current Full-profile analyses comprised 12 known swap views and six corresponding originals. The incumbent localized all 12 swaps at 0.92 IoU with their exact native edit intervals. The new rule preserved every range, and no original received a flag. A separate archived production replay preserved all four deliveries of a single known swap at 61/72 IoU. Those four deliveries represent one intervention.

Geometry was measurable for all 15 selected ranges in the primary comparison. The camera replay verified all 108 selected source pictures and aligned crops. The native workers verified 432 selected picture commitments over 273.8 cumulative seconds. The independent arithmetic reviewer checked all 36 outcomes and 27 exact endpoint or overlap comparisons. The held ordinary cases showed no reduction in false flags, so the comparison remains an evaluation result. [Aggregate and audit](evaluation-panels/video-webcam-pose-transfer-2026-11-11.json)

## Reproduce

Use the retained private v200/v202 camera preparation and archives, v253 development receipts, and v95 matched native source archive. Start in a fresh directory. The native worker runs one clip at a time and uses temporary storage for each byte-exact upload. The two arms can run concurrently.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_webcam_pose_transfer freeze "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_pose_transfer camera "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_pose_transfer native "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_pose_transfer readout "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_webcam_pose_transfer "$run_dir"
```

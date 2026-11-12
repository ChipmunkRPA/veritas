# Scene-referenced face motion in webcam review ranges

A previously flagged genuine recording showed a person lean sideways, tilt their head and return against a near-stationary scene. The earlier landmark-order check measured normal alignment there. This study asks whether the wider position and roll change can explain other genuine webcam flags while preserving correctly located swaps. The earlier recording supplied one development example. All three deliveries of that recording are outside the held camera comparison. [Prior webcam and swap study](video-webcam-pose-transfer-2026-11-11.md)

## Frozen test

Keep the incumbent face-review range, before/during/after sample roles and exact native timestamps. Use each saved center frame with its decoded RGB commitment. Resize source views to a 320-pixel analysis grid and exclude the union of expanded face boxes and the bottom 32% of the scene from reference tiles. This reduces the chance of tracking the moving face or shoulders as background. Select textured tiles from the last before frame. Search up to five pixels around each tile, require high normalized correlation and a distinct correlation peak, and fit a similarity transform with supported tiles spread in both axes. The local reference region may occupy just one side of the source scene. Require supported registration for every center and no more than six pixels of fitted motion at any corner of the analysis grid. A featureless scene or unreliable fit produces an unassessed result.

Use the fitted transform to express the face-box center and eye-line angle on the reference grid. The two flanks need reliable eye geometry. The primary check requires a return to within a quarter of the typical face-box width. At least 60% of the middle centers must move to the same side by at least 0.65 box widths relative to both flanks. The second check also requires the flanks to agree within five degrees of roll and at least half the middle centers to differ from both flanks by ten degrees on the same side. Both checks preserve the source interval. A qualifying measurement could add motion context without contributing to AI confidence. Similar motion can coexist with an actual edit, so correct swaps must remain visible.

The held panel includes 12 original, chroma-converted or recompressed deliveries of four genuine webcam recordings that received flags in earlier production analyses. Two sources had prior illumination and geometry investigation. The other two supplied the previously studied landmark flags. Their source motion measures are first exposure to the new cue. The study verifies exact original app measurements and native source pixels before replay. Six matched native face swaps with distinct source and donor keys run in the current Full-profile appearance path. The study uses one swap from each source, with the two swap methods alternating. These sources had prior project exposure and incomplete portrait attribution. [Webcam data terms](https://github.com/microsoft/VCD/blob/main/LICENSE-DATA) · [Webcam controls](video-vcd2-full-app-v202-method.md) · [Native source preparation](video-common-source-preparation-v89.md)

The separate camera cohort contains four participants from Gros and Straub's CC BY 4.0 controlled motion data. The videos contain standardized head movements under set lighting against a white screen. Two participants from each archive were selected by a fixed hash order, with four lighting choices fixed before evaluation. The study used bounded HTTP member reads and verified the publisher ZIP index, member range and CRC, then recorded the local SHA-256 of each extracted file. The archive-wide publisher SHA-256 identifies the source container. The full archives were not downloaded. The local 720p H.264 deliveries preserve every source-relative picture timestamp and omit the metronome track. These are real controlled movements, with limited demographic and scene variety. [Collection protocol](https://www.mdpi.com/2306-5729/4/3/130) · [Dataset part one and terms](https://data.mendeley.com/datasets/xgg8xcscr5/1) · [Dataset part two and terms](https://data.mendeley.com/datasets/f47pm7rwt3/1)

The gate requires at least four true incumbent false ranges contextualized across two recorded sources, no correctly localized swap range contextualized, at least four correct swap ranges across three sources, and at least three newly sourced genuine app analyses with incumbent appearance coverage. The two candidate checks are compared independently.

## Held webcam and swap results

| Result | Archived genuine webcam | Current app known swaps |
| --- | ---: | ---: |
| Incumbent ranges | 12 | 6 |
| Scene and face motion measured | 6 | 4 |
| Scene registration unavailable | 6 | 2 |
| Range contextualized by lateral movement | 3 | 0 |
| Range contextualized by lateral movement with roll | 3 | 0 |

All three qualifying webcam ranges came from one recording. Two other genuine recordings lacked stable enough texture or registration in the available scene region, and the remaining recording's movement did not qualify. The check kept all six useful swap ranges at 0.92 native temporal IoU. The measured sample from one qualifying webcam range was independently decoded into three exact source pictures and three matching crops. The source SHA-256, exact frame indices, PTS, and both PNG and RGB commitments remain in the private evidence packet. Both arms fall short of the required improvement across two sources.

## Fresh controlled camera results

All four current app analyses returned an incumbent appearance range on genuine head movement. None received new motion context. Three videos supplied zero qualifying stationary reference tiles and the fourth supplied seven reference templates without a reliable fit across the selected source pictures. The plain backdrops expose a coverage limit for a check that depends on visible reference texture.

The workers matched all 2,190 decoded delivery timestamps to the original source-relative MP4 clocks and verified 34 source pictures in the four selected ranges. The archived camera and current swap lanes verified another 224 pictures. The independent reviewer reproduced all 22 case outcomes, 28 exact interval checks, and 20 measured decisions across the two arms. Three focused tests covered a stationary scene, independent roll and lateral movement, and unavailable or moving background. The method stays experimental. [Aggregate and audit](evaluation-panels/video-webcam-motion-transfer-2026-11-11.json)

## Operational source preparation

The initial controlled camera worker timed out while probing all 4K source frames. The next attempt used the exact MP4 video packet PTS clock and stopped at an isolated-environment check before inference. A third attempt reached the upload check, where the lossless RGB delivery exceeded the application's 500 MB limit. The last amendment set the isolated environment before runtime import and used ordinary 720p CRF 16 H.264. All retained sources, candidate points, and study gates stayed fixed. The final decoded upload PTS grid was required to match every original source-relative presentation tick.

## Reproduce

Use the retained earlier camera and native files and the four separately acquired public motion samples. The intake runner binds the two Mendeley publisher inventory replies and exact ZIP suffixes. The archive URLs and case members stay in the private roster.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_gros_range_intake plan "$run_dir"
backend/.venv/bin/python -B -m scripts.video_gros_range_intake fetch "$run_dir" --destination "$gros_source"
backend/.venv/bin/python -B -m scripts.video_webcam_motion_transfer freeze "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_motion_transfer camera "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_motion_transfer positive "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_motion_gros_clock run "$run_dir"
backend/.venv/bin/python -B -m scripts.video_webcam_motion_transfer readout "$run_dir"
backend/.venv/bin/python -B -m scripts.review_video_webcam_motion_transfer "$run_dir"
```

The controlled-motion runner uses the retained pre-inference amendment. Run the final readout and independent arithmetic check after all three case journals are complete.

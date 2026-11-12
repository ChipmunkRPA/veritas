# Paired background changes near a face-review range

Two ordinary cuts in clips with similar backgrounds produced face-change ranges in the local Full worker. Both cuts lasted from **2.680 to 3.000 seconds**. The existing shot check found no boundary in either clip. The face review placed its ranges at **2.640–3.160 seconds** and **2.360–2.880 seconds**. Both findings came from the first face-appearance comparison. [Tracked reproduction](video-face-scene-continuity-2026-11-12.md#existing-shot-check-and-remaining-gap)

The candidate starts with an existing face-review range and checks nearby consecutive native pictures for two abrupt changes outside the tracked face. It measures a departure and return, with more similarity between the surrounding pictures and within the intervening scene. This follows the idea of comparing both within-scene and between-scene relationships around a transition, implemented here as a local measurement with no additional model or weights. [OmniShotCut study](https://arxiv.org/abs/2604.24762) · [Adjacent-frame comparison](https://openaccess.thecvf.com/content/ICCV2025/html/Hwang_Generic_Event_Boundary_Detection_via_Denoising_Diffusion_ICCV_2025_paper.html)

## Frozen measurement

The test uses 120 upright H.264 pictures at 640 × 360 and exactly 25 frames per second. Every picture has a verified presentation timestamp. Read up to nine pictures on either side of the first-pass range and two additional context pictures at each end. Consider an entry and exit separated by 5 to 12 pictures. At least half of the shorter of the proposed cut and the first-pass range must overlap. Require a unique matching pair.

Measure an 8 × 6 grid on a 160 × 96 thumbnail. Exclude every tile that touches any of the contributing face boxes after expanding those boxes by 35% on each side. At least 16 tiles must remain. For each remaining tile, compare mean-centered luminance patterns with a per-tile contrast normalization and compare mean absolute chroma differences. A tile supports the pair when the weaker of its two edge changes is at least three times the median of the four nearby within-scene changes. The larger of the differences between the two outside pictures and between the two inside pictures must be at most 55% of that weaker edge. For luminance, require 2.5 units of texture change, or at least 1.2 with a corroborating chroma change of at least .85. At least 35% of eligible tiles must support the pair, with at least two support tiles in each of three quadrants.

On the two known false ranges, the candidate located both exact cut boundaries with 16/28 and 15/28 supporting tiles. With those same ranges and face boxes substituted into the comparison as a diagnostic, it returned no pair on 14 earlier swap and ordinary-control deliveries or on the two other cuts. The two positive examples share the same source pair and were available during method design. The prospective panel excludes both recordings and all other known source and donor groups from the preceding localization studies. [Previous exposure and source rights](video-face-scene-continuity-2026-11-12.md#comparison)

## Held worker results

Source-only matching found 15 disjoint pairs under the frozen background similarity criterion among 38 eligible recording groups. The held panel used the eight closest pairs, with normalized outer-background distances from .0384 to .1100. The native source clocks included 29.97, 30, 50, 59.94, and 60 frames per second. Each output picture takes the nearest authenticated native picture with an earlier-timestamp tie break. No interpolated source pictures were inserted. The tracked worker used simulated signed Stripe payments and ran the Full video shot, face tracking, and appearance lanes.

| Held cases | Current useful swap ranges | Current coarse ranges | Background pairs accepted |
| --- | ---: | ---: | ---: |
| Genuine cross-recording cuts | — | 0/16 | 0 |
| Long face swaps | 15/16 | 15/16 | 0 |
| Short face swaps | 8/8 | 3/8 | 0 |
| Light, zoom, and original controls | — | 0/24 | 0 |

The shot stage found at least one exact boundary on 12 of the 16 genuine cuts. The four other cuts also produced no displayed face range. The candidate only acts on a selected coarse range, so all 18 evaluated inputs in this panel were swaps. It preserved all 23 useful displayed swap results, including 17 with a useful coarse range. The experiment could not measure whether it would remove new false alerts. The two known development examples remain the only observed corrections. The candidate remains experimental. The next study can screen separate genuine cut deliveries for false ranges through the tracked worker before exposing that challenge subset to the frozen candidate. That measures behavior conditional on an existing alert.

The independent authoring replay verified 2,308 native source pictures and 56 authored intervals and rebuilt all 64 MP4s byte for byte. The standard-library audit recalculated the exact timing rules and decision counts without inference. The worker ran in about 319 seconds. The candidate replay took about 9 seconds. The initial bootstrap failed at an unpaid test webhook before any scoring. A corrected runner verified the isolated database before starting the completed test. Both the authored panel and the candidate measurement stayed fixed. [Verified result ledger](evaluation-panels/video-native-background-edges-2026-11-12.json)

## Reproduce

Use the retained source archive, local qualified model assets, and a new private run directory. Pass the same optional `--archive` path to each native command when using an alternate archive.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_native_background_panel freeze "$run"
backend/.venv/bin/python -B -m scripts.evaluate_video_native_background_panel freeze "$run"
backend/.venv/bin/python -B -m scripts.video_native_background_panel prepare "$run"
backend/.venv/bin/python -B -m scripts.replay_video_native_background_panel "$run"
backend/.venv/bin/python -B -m scripts.evaluate_video_native_background_panel run "$run"
backend/.venv/bin/python -B -m scripts.review_video_native_background_authoring "$run"
python3 -B -I -S backend/scripts/review_video_native_background_results.py "$run" --export "$result"
```

The public replay uses the same final verifier with the published ledger as its only input. Source video, image crops, and private model outputs stay local.

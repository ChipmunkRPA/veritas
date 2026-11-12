# Paired H.264 video comparison on a second 25 fps group

The existing paired comparison localized 31 of 32 ordinary changes across eight source groups at 360 × 288 and exactly 25 fps. It showed no local finding for 48 neutral deliveries, 32 tandem encodes, or 16 unrelated earlier copies. The original two-bank H.264 trial failed its joint gate after three ordinary delivery flags in the 480 × 360 CREMA-D group. [Reserve aggregate and replay](evaluation-panels/video-reference-h264-grid-reserve-2026-11-11.json) · [Original transfer](video-reference-h264-earlier-transfer-2026-11-11.md)

## Frozen comparison

The reserve roster uses 16 GRID recordings from eight known source components separate from the original GRID development and held groups. The source receipt records CC BY 4.0 and 75 native pictures per recording. The project had previously extracted and verified these pictures. This is the first paired comparison on the reserve roster. The test checked the original source and picture hashes and the exact 40 ms relative picture clock. [GRID source provenance](video-mouth-jaw-transfer-2026-11-11.md)

Each earlier MP4 was encoded independently as H.264 4:2:0 at CRF 23 with the medium preset and a fixed 48-picture GOP. The submitted controls used independent CRF 28 or CRF 33 encodes, an independent whole-frame tone change, two tandem encodes of the earlier copy, and an exact-clock unrelated copy from a different source component. The submitted edit cases apply a fixed local brightness change or blur to a lower-central rectangle from 1.080 s to 1.880 s, using a half-open interval. The original point, shared-picture check and location criteria stayed fixed. A useful result needs at least half interval intersection over union and the fixed changed-pixel fractions in the selected native tile.

The frozen gate required useful tone locations on at least 14 recordings across six source groups, 80% assessed coverage for every condition other than unrelated references, at most one neutral flag, and zero tandem or unrelated flags. Blur was a separate challenge outside the gate.

## Results

| Held condition | Cases | Assessed | Local findings | Useful locations |
| --- | ---: | ---: | ---: | ---: |
| Local brightness | 16 | 16 | 16 | 16 |
| Local blur | 16 | 16 | 15 | 15 |
| Independent CRF 28 or 33 or global tone | 48 | 48 | 0 | — |
| Tandem CRF 28 or 33 | 32 | 32 | 0 | — |
| Unrelated earlier reference | 16 | 0 | 0 | — |

All eight groups contributed a useful local brightness location. The selected range matched both authored endpoints on five of the 32 edited cases. It serves as a review interval. All unrelated pairs abstained for insufficient shared pictures. Independent replay checked 128 exact picture-clock pairs, 16 native originals, 186 selected before, during and after crops, and the actual changed-pixel counts for 31 findings.

This second group tests one capture corpus, picture size, cadence, edit rectangle and range. It measures localization of visible differences between two copies. The previous group supplies the neural redraw results, with 24/32 useful locations and all seven source-compatible visible-motion redraws missed. The paired result provides no AI attribution. The 480 × 360 H.264-earlier case remains outside the supported scope.

## Reproduce

Use the private GRID reserve plan and the pinned initial transfer run with its independent audit. Use a fresh local output directory for `$run`. The media and per-source roster stay local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze held report; do
  backend/.venv/bin/python -B -m scripts.video_reference_h264_grid_reserve "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_reference_h264_grid_reserve "$run"
```

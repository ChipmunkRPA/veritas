# Fixed-position picture detail on neural mouth redraws

This comparison tests whether a short mouth redraw breaks the agreement of fine detail at the same native picture coordinates. It uses only the submitted pictures. Published forensic studies motivate matching spatial residuals and show how ordinary picture content, compression and stabilization can affect those measurements. The implementation uses the existing licensed face geometry asset and local arithmetic. No camera fingerprint or camera identity is established. [Fixed-position video research](https://www.tara.tcd.ie/tara8/server/api/core/bitstreams/2be474fb-d432-4087-a7c8-acee9e3a6e29/content) · [Compression and scene-detail study](https://arxiv.org/abs/2402.06669) · [Block-based video study](https://dfrws.org/wp-content/uploads/2025/03/Video-capturing-device-identification-thr_2025_Forensic-Science-Interna.pdf) · [Face geometry terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

## Frozen comparison

The method takes an 80 ms sample grid with verified native timestamps. Face geometry selects mouth and nearby reference areas. Every measured pixel remains in its native position. Eligible windows stay within one shot and limit drift in mouth position and eye distance. The calculation removes a smoothed copy of luminance and averages the residual within a middle interval and two four-sample flanks. The candidate interval lasts 0.32–0.80 seconds. Low-gradient pixels in the two flanks determine the comparison area. Both areas need at least 48 supported pixels and measurable agreement between the flanks.

The fixed-pixel arm measures how much mouth detail agrees between the two flanks and loses agreement in the middle. It subtracts the correlation observed at nearby offsets to reduce effects from broader image structure. The primary arm subtracts the corresponding deficit in the nearby face. It selects the strongest interval before applying the frozen review point. Exact source times, unwarped frames and mouth crops link the result to the upload. The audio is reopened for evidence review only.

The review point is the third-highest ordinary maximum across 56 development views from eight GRID recordings in four source components. All 56 were measurable. The held comparison uses eight GRID recordings from four separate components, ten CREMA-D confirmation groups, eight H.264 re-encodes and four whole-picture cuts. Ordinary views include originals, retiming, resampling, blur, brightness changes and audio-only delays. The two additional reserve speakers provide four neural redraws and four ordinary views. These retained media had earlier project exposure. The development and evaluation roles were frozen before measuring this cue. [GRID publisher terms](https://zenodo.org/records/3625687) · [CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) · [Matched held bank](video-mouth-temporal-detail-2026-11-12.md) · [Reserve edit and clock audit](grid-reserve-neural-mouth-2026-11-12.md)

A useful range overlaps the authored interval by at least 0.50 intersection over union. The comparison requires additions on different source groups and includes a separate requirement for redraws that preserve verified source-time mouth motion. The full rule also caps added ordinary, re-encode and cut flags and requires sufficient measured coverage in both classes. [Frozen criteria and results](evaluation-panels/video-mouth-fixed-residual-2026-11-12.json)

## Held results

| Outcome | Direct comparison | Fixed-pixel mouth | Mouth adjusted by nearby face |
| --- | ---: | ---: | ---: |
| Correct GRID redraw ranges | 1/32 | 1/32 | 2/32 |
| Misplaced GRID redraw ranges | 0/32 | 4/32 | 9/32 |
| GRID ordinary flags | 0/56 | 12/56 | 18/56 |
| Correct CREMA-D redraw ranges | 1/20 | 0/20 | 2/20 |
| Misplaced CREMA-D redraw ranges | 2/20 | 0/20 | 3/20 |
| CREMA-D ordinary flags | 8/40 | 0/40 | 9/40 |
| H.264 re-encode flags | 0/8 | 1/8 | 4/8 |
| Genuine-picture hard-cut flags | 0/4 | 0/4 | 0/4 |
| Correct reserve redraw ranges | — | 0/4 | 0/4 |
| Reserve ordinary flags | — | 0/4 | 2/4 |

Both candidate arms assessed 165/168 held and reserve views. The three unavailable CREMA-D views were two neural redraws and one ordinary control. Neither arm localized any of the seven qualified held GRID redraws or two reserve redraws that preserve source-time mouth motion. Before applying the review point, the primary arm selected a range with sufficient overlap in 9/32 GRID, 3/20 CREMA-D and 0/4 reserve positives. Its top-score AUC for neural versus ordinary views was 0.479, 0.610 and 0.500 in those banks. The primary arm's two correct CREMA-D results came from one source group. These failures in both range selection and score separation rule out a cutoff-only correction. The check stays experimental and carries no AI confidence weight.

One correct range spans **0.827–1.295 s** inside an authored **0.760–1.428 s** redraw. Its exact before, during and after pictures are at **0.760**, **0.994** and **1.295 s**, with 7,488 aligned source audio samples in the selected interval. The crop shows the area measured. The procedure does not estimate which individual pixels the renderer changed.

The primary held pass took 111.6 seconds with 179.1 MiB peak reported RSS. A separate readback checked 168 file hashes, 496 decisions, 171 exact native crops and 24 aligned PCM intervals. All eight original/audio-delay pairs produced identical picture results. The audit checked the saved measurements and selected evidence against the source. [Aggregate and verification receipt](evaluation-panels/video-mouth-fixed-residual-2026-11-12.json)

## Replay

Use the retained mouth-jaw comparison and admitted reserve as inputs. Media, individual measurements and screenshot/audio examples remain local.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held reserve report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_fixed_residual "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_fixed_residual "$run"
```

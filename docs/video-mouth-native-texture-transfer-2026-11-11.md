# Native mouth texture on short redraws

The native texture check localized eight of 32 GRID redraws and two of 20 CREMA-D redraws. The direct check localized one in each bank. The new check flagged 13 of 56 ordinary GRID views and missed all seven redraws with independently verified source-compatible mouth motion. It remains experimental. [Frozen aggregate and source replay](evaluation-panels/video-mouth-native-texture-transfer-2026-11-11.json)

## Fixed comparison

This single-file trial measures local picture texture on the native pixel grid. Recent image and video forensics research evaluates spatial and temporal residuals as candidate evidence of local editing. The present trial estimates relative texture differences with a compact CPU model and exact source-clock spans. [Spatiotemporal defect research](https://www.nature.com/articles/s41598-026-55334-x) · [Handcrafted residual research](https://www.nature.com/articles/s41598-026-72389-y)

A pinned face detector locates one visible face. Eye and mouth landmarks place three ellipses around the observed mouth. The ellipses select native pixels without resampling. At two Gaussian scales, the check compares low-gradient luma and chroma residuals inside the mouth and in a perioral band against a surrounding band. The descriptor includes residual magnitude quantiles, cross-channel correlations, and luma correlations at neighboring native pixels. A regularized linear direction learns which changes distinguish known redraws from matched ordinary processing. It uses the 32 neural and 48 ordinary measured intervals from eight development recordings in four known GRID source components. Each source contributes equal positive and ordinary contrast means. The third-largest ordinary maximum from the same development panel sets the point. [Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

The check finds the strongest fixed short interval whose median is higher than both measured flanks, using the native frame table and the prior direct comparator’s sampled shot guard and window lengths. Held inference uses the submitted pictures alone. An exact archived counterpart and known edit masks establish test truth. The archived motion qualification identifies seven GRID original-time redraws with source-compatible visible movement and unchanged audio. [Motion qualification](grid-mouth-motion-v158-result.md) · [Matched source and control protocol](video-mouth-jaw-transfer-2026-11-11.md)

The held test reuses eight GRID recordings from four separate known source components and ten separate CREMA-D source groups. These clips had earlier project exposure. Each comparison covers the exact same qualified files as the archived direct picture baseline, including eight H.264 recompressions and four genuine-picture hard cuts. A correct range needs at least 0.50 native time intersection over union with the known authored interval. The gate requires useful additions across both banks and on the source-compatible GRID subset, bounded ordinary flags, no added hard-cut flags, and at least 80% coverage of positives and controls in each bank.

## Held results

| Held outcome | Direct picture comparison | Native texture |
| --- | ---: | ---: |
| GRID redraws correctly localized | 1/32 | 8/32 |
| GRID redraws with a misplaced range | 0/32 | 1/32 |
| GRID source-compatible redraws correctly localized | 0/7 | 0/7 |
| GRID ordinary views flagged | 0/56 | 13/56 |
| CREMA-D redraws correctly localized | 1/20 | 2/20 |
| CREMA-D redraws with a misplaced range | 2/20 | 5/20 |
| CREMA-D ordinary views flagged | 8/40 | 9/40 |
| H.264 recompressed originals flagged | 0/8 | 0/8 |
| Genuine-picture hard cuts flagged | 0/4 | 1/4 |

The eight useful GRID detections were new relative to the direct check. The texture check lost the one direct detection. It found four original-time speech-conditioned redraws, three earlier-driven speech-conditioned redraws and one earlier-driven expression redraw. The original-time qualification is separate from the requested renderer time. All seven independently source-compatible expression redraws remained below the point. Two original GRID controls and their visually identical audio-only counterparts flagged. The other nine ordinary flags appeared on matched brightness, resampling and retiming controls. Both useful CREMA-D detections came from earlier-driven redraws, and the candidate lost the one direct detection there. Ordinary retouching and natural motion can produce similar local texture signals.

An independently replayed example reported **1.120–2.000 s** on an edit at **1.080–1.880 s**, for 19/23 temporal intersection over union. The before, during and after crops used the native picture timestamps **1.040**, **1.520** and **2.000 s**. The inspected native crop identifies the measured area. Exact edited pixels are available only from the authored test mask.

The single-thread held worker measured all 160 views and 5,498 sampled pictures in 87.4 seconds at approximately 132 MiB peak reported RSS. An independent replay verified the 160 file hashes, all 320 comparison decisions and 117 selected native source crops and picture timestamps. It replayed saved responses and their decision arithmetic. The replay keeps source media, identities and feature vectors local.

## Reproduce

Use a new private run directory and the exact retained mouth-jaw transfer records. The latter bind the previous direct scores, verified source ranges and media paths. The reviewer reopens the held inputs and recreates the decision arithmetic and crop hashes.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze development held report; do
  backend/.venv/bin/python -B -m scripts.video_mouth_native_texture_transfer "$phase" "$run_dir" --prior "$prior_run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_native_texture_transfer "$run_dir" --prior "$prior_run"
```

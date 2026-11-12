# Motion reconstruction on verified mouth redraws

The motion-reconstructed mouth check measured all 88 held views. It localized zero of 32 neural redraws and flagged four of 56 ordinary controls. A direct comparison on the same pictures localized one redraw with no false control ranges. The motion reconstruction stays experimental. [Aggregate and native replay](evaluation-panels/video-mouth-reconstruction-transfer-2026-11-11.json)

## Matched source panel

The retained GRID pilot contains 16 original recordings from eight known source components. Each recording supplies the original, four local neural redraws, four ordinary resampling or brightness changes matched to the two renderer branches, one genuine frame retime, and one audio-only retime. Earlier construction verified the native changed-pixel masks, picture timestamps, and PCM. This study independently verifies the frozen file bytes and sampled decoded pictures. The source components were ordered by a fixed hash and split equally between development and held scoring. No held response set a comparison point. The panel had earlier project exposure to other methods. The untouched reserve remains unrendered. [GRID publisher](https://zenodo.org/records/3625687) · [Construction and source commitments](grid-native-clips-v149-result.md) · [Renderer terms](video-timing-coverage-and-codec-holds-2026-11-11.md)

The positive interval is the half-open native frame range `[27,47)`, or 1.080–1.880 s on the output clock. The earlier independent mouth-geometry review found that seven of the eight held original-time expression-driven redraws reproduced the original visible mouth trajectory within its comparison criteria. Their original soundtracks were retained. This subset tests an observed source-time motion match. Independent phoneme-level agreement and listener-perceived synchronization remain unverified. The speech-conditioned redraws and earlier-driven variants have separate results. [Motion qualification](grid-mouth-motion-v158-result.md)

## Comparison

The study samples every second native frame. It registers each face to a 96-pixel grid using the two measured eye positions, and uses the observed mouth center to place a small mouth region and nearby face reference. The direct comparison measures the center frame against both neighboring frames. The candidate first warps each neighboring frame toward the center with dense optical flow. Both compare the 75th-percentile absolute residual inside the mouth and nearby face reference. They scan 0.40, 0.56, 0.72 and 0.88 s intervals and select the strongest middle response above both measured flanks. The score uses only the tested file. The matched original and archived changed-pixel mask supply evaluation truth. The commercial-admitted face detector is the only pretrained weight. [Counterfactual frame consistency research](https://arxiv.org/abs/2606.22772) · [Lip motion and pose research](https://proceedings.mlr.press/v306/she26a.html) · [Face weight terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Each arm's comparison point is the third largest measured ordinary development response, using strict exceedance. A correct range needs at least 0.50 temporal intersection-over-union with the authored interval. The gate requires at least six correct redraws across three held components, at least two more correct ranges than direct, at least two correct source-time-compatible redraws, and at most two added ordinary ranges. The scan, thresholds and held groups were sealed before held scoring.

## Held results

| Check | Assessed views | Correct neural ranges /32 | Misplaced neural ranges /32 | Ordinary flags /56 | Correct original-time compatible /7 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct comparison | 88 | 1 | 0 | 0 | 0 |
| Motion reconstruction | 88 | 0 | 3 | 4 | 0 |

The direct comparison flagged 1.280–1.680 s on one earlier-driven expression redraw, exactly 0.50 IoU. The motion reconstruction flagged 0.400–1.280 s on one original-time expression redraw with compatible visible motion, for 5/37 IoU against the true range. The two other misplaced ranges came from speech-conditioned redraws. Its four ordinary ranges came from three local resampling changes and one brightness change. Both checks gave identical observations on all eight held original and audio-only pairs.

The run verified 6,688 sampled native pictures across development and held. The separate reviewer checked all 88 held file digests, recomputed the decision counts, decoded 489 native evidence frames, and verified eight archived changed-pixel masks at selected mouth observations. One private example retains three exact before, during and after pictures and their corresponding native crop bounds. Its estimated range is marked misplaced.

## Reproduce

Use the retained archive with the `grid-native-clips-v149` and `grid-mouth-motion-readout-v158` directories and the existing pinned face weight.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
backend/.venv/bin/python -B -m scripts.video_mouth_reconstruction_transfer freeze "$run_dir" --archive "$archive"
for phase in development calibrate held readout; do
  backend/.venv/bin/python -B -m scripts.video_mouth_reconstruction_transfer "$phase" "$run_dir" || exit
done
backend/.venv/bin/python -B -m scripts.review_video_mouth_reconstruction_transfer "$run_dir"
```

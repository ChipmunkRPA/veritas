# Local speech-layer comparisons on short edits

The low-dimensional layer and boundary heads missed all ten edits with louder competing speech at their fixed operating points. The primary boundary head also lost both dry locations from the existing frame response.

| Response | Dry useful / genuine false | −5 dB useful / genuine false | +15 dB useful / genuine false | −5 dB best ranges overlapping the edit before applying a point | −5 dB edits ranked above both ordinary controls at the known interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing frame scores | 2 / 0 | 0 / 0 | 1 / 0 | 5 | 6 |
| Weighted-layer projection | 3 / 2 | 0 / 2 | 1 / 0 | 4 | 7 |
| Early-layer projection | 0 / 0 | 0 / 0 | 0 / 0 | 3 | 5 |
| Middle-layer projection | 4 / 2 | 0 / 0 | 0 / 0 | 4 | 6 |
| Deep-layer projection | 0 / 1 | 0 / 2 | 0 / 2 | 4 | 4 |
| Combined boundary distances | 0 / 0 | 0 / 0 | 0 / 1 | 2 | 2 |

Each held condition contains ten exact synthetic edits. The dry and +15 dB conditions have 20 genuine views each. The −5 dB condition has 34 genuine views, including unedited and isolated-background controls. The numbers describe repeated deliveries on ten prior host comparisons, with ordinary donor and background reuse. [Exact input construction, source terms and prior results](audio-partial-residual-transfer-2026-11-12.md)

## Fixed comparison

Reuse the exact licensed model, source roles and 316 previously scored waveforms. Capture early, middle and deep hidden states at indices 4, 12 and 24, alongside the serving weighted mixture. Keep the original 20 cells at 0.2-second intervals. One full four-second forward supplies every new measurement. Encoder context can extend beyond each cell. The time grid identifies a position in the input. [Checkpoint terms](commercial-model-audit.md) · [Cross-layer representation research](https://arxiv.org/abs/2611.05678)

Subtract the mean of left and right flanks from each cell representation. The flanks use cell offsets −3 and −2 and +2 and +3, clipping indices to the available track. Use one fixed signed 32-coordinate projection for every layer. The boundary head uses six normalized squared distances per layer, giving 24 dimensions. These distances measure a center cell relative to the flanks and immediate neighbors. An ordinary speaker transition or recording change can produce the same observation. [Method and freeze](evaluation-panels/audio-partial-layer-boundary-2026-11-12.json)

Fit all five small heads on the same 72 matched fitting views, using the exact synthetic-overlap fraction and regularized linear loss. The eight fitting hosts include clone and stock edits. Expressive edits occur only in confirmation. Seal each arm's point at the highest genuine score in the separate operating-point role. Select one 0.6-second range from its highest three-cell mean. Count a useful location at 30% IoU and 50% edit coverage. The primary candidate needed two net new useful −5 dB locations across two generator families, at most one extra noisy false pair, no lost dry useful location and no isolated-background flag. The original source panel was already exposed to earlier analysis. A positive result would still require a new source panel. [Partial-spoof localization research](https://arxiv.org/abs/2604.02913)

## Replay

The new run completed 316 forwards in 231.7 seconds. Every weighted cell and whole-clip score reproduced the prior result exactly. The native reviewer recalculated the five objectives and gradients, sealed points and 1,008 source-linked decisions. The standard-library reviewer reproduces the public time geometry, control counts, and advancement rule across 36 aggregate cells. It rejected a one-sample alteration to a reported range. Raw waveforms, features, coefficients and per-case scores stay local. [Public verification receipt](evaluation-panels/audio-partial-layer-boundary-2026-11-12.json)

```sh
python3 -I -S backend/scripts/review_audio_partial_layer_boundary_public.py \
  docs/evaluation-panels/audio-partial-layer-boundary-2026-11-12.json \
  docs/evaluation-panels/audio-partial-residual-transfer-2026-11-12.json
```

Use the completed prior local `parent` directory and exact installed `weights` directory to rerun native inference. Use a new output directory for `run`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
audio_layer_args=("$run" "$parent" "$weights")
backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary freeze "${audio_layer_args[@]}"
for role in fusion_fit confidence_cal; do
  backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary encode "${audio_layer_args[@]}" --role "$role" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary fit "${audio_layer_args[@]}"
for role in selection confirmation; do
  backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary encode "${audio_layer_args[@]}" --role "$role" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary report "${audio_layer_args[@]}"
backend/.venv/bin/python -B -m scripts.audio_partial_layer_boundary verify "${audio_layer_args[@]}"
```

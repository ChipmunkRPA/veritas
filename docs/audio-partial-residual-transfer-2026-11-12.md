# Training a temporal voice check on short edits with competing speech

The paired temporal head added one useful location under quieter competing speech. At −5 dB, all five arms missed all ten held edits at their fixed points. The paired head's strongest range overlapped one of those edits before applying a point, compared with five for the existing frame responses. The candidate stays experimental.

| Held condition | Known synthetic edits | Genuine views | Current clip flags | Current frame locations | Dry-trained locations | Mixture-trained locations | Paired-head locations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dry foreground | 10 | 20 | 1 | 2 | 4 | 3 | 4 |
| Competing speech at −5 dB | 10 | 34 | 0 | 0 | 0 | 0 | 0 |
| Competing speech at +15 dB | 10 | 20 | 0 | 1 | 3 | 1 | 2 |

All arms produced zero false flags on the 74 held genuine views. The +15 dB condition is a controlled mixture with quieter competing speech. No separator produced those inputs. The whole-clip arm has a four-second extent and serves as a file-flag comparator. The four other arms select a 0.6-second inspection range. The fixed range bounds are on the exact decoded sample clock and measure partial overlap with the known edit.

## Waveforms and source roles

The v81/v82 source receipts bind the local recordings and known source groups. Each matched comparison starts from the same genuine four-second host. Insert a different passage from that host, a genuine donor from another voice, or a synthetic donor at the same position. Apply the same 20 ms fades, match local level, and use the same interfering speech waveform and gain across all three arms. Each edit spans 0.64, 0.80, or 0.96 seconds. The exact PCM outside the edit is identical across matched arms. Unedited and background-only controls enter the strongest-interference condition. [Source qualification and terms](audio-complete-sources-v81-result.md)

The 316 views cover 30 unique host comparisons, split into eight fitting, six operating-point, six diagnostic, and ten confirmation groups. Exact source, canonical speaker, transcript and background ancestry were checked across roles. The underlying sources had earlier project exposure. The interventions and placements were newly constructed. The licensed competing speech comes from four distinct LibriSpeech groups per role. Recording rooms are unknown. Donors and backgrounds recur within roles, so repeated deliveries do not count as independent recordings. Clone and stock examples enter fitting. Expressive synthesis is present only in confirmation. [Exact source audit and outcomes](evaluation-panels/audio-partial-residual-transfer-2026-11-12.json)

## Fixed temporal comparison

Reuse the installed commercially reviewed encoder and the existing weighted layer mixture. Keep the existing clip response and apply its unchanged affine head to twenty 0.2-second cell means. One native four-second forward supplies both. The frame centers begin at sample 280 and advance by 320 samples. The encoder can use context outside a cell. Its grid provides source positions for review. [Checkpoint terms](commercial-model-audit.md)

The fitted arms use the same cell vectors and their differences from the recording-wide means. Fit a regularized linear head using the fraction of each cell covered by the synthetic insert. Compare fitting on dry audio only with fitting on dry and both mixture levels. The paired arm additionally rewards an edited score above each matched genuine score at the known interval, and penalizes differences away from the edit. Determine every operating point from the maximum genuine response in the separate calibration role and seal those points before computing diagnostic or held responses. The reported interval is the strongest three-cell average. It qualifies as useful with at least 30% intersection over union and 50% edit coverage. [Partial-spoof localization](https://arxiv.org/abs/2604.02913) · [Matched-condition training](https://arxiv.org/abs/2607.10345)

At −5 dB, the existing frame response ranked seven of ten edits above both matched genuine controls. The paired head ranked three. Even with the true interval supplied for a diagnostic comparison, the paired head ranked four of ten above both controls, compared with six for the existing response. The paired head added one held stock-voice location at +15 dB and preserved earlier useful flags. All eight noisy expressive examples stayed inconclusive in every arm. The study used eight fitting hosts and a 2,048-component representation. It supports measuring transfer to other sources before changing the app.

## Six-second meeting-audio replay

Apply the sealed heads and points to 48 earlier consumed dry and competing-meeting-speech recordings. Run exact four-second windows at 0 and 2 seconds, and retain the strongest result for each file. The additional window gives the checks two opportunities to flag. This replay uses earlier exposed waveform constructions for local evaluation. The model receives only submitted waveform samples and uses no reference channel. [Meeting construction and rights](audio-overlap-window-transfer-2026-11-12.md)

All four local arms recovered two of six dry held locations. The original frame scores also flagged one genuine dry clip. The new heads flagged none of the six genuine dry controls. With loud meeting speech, the paired head flagged one manipulated clip at an incorrect location and one genuine clip. The mixture-trained head flagged one manipulated clip at an incorrect location and zero genuine clips. Every arm missed all six useful locations. The independent reviewer reconstructed the 96 model-input commitments and checked all 20 decision cells from the saved flags and source coordinates.

## Independent replay

Waveform verification regenerated all 316 exact inputs, checked matched interference and source clocks, and rejected a one-sample alteration outside the edit. The CPU scored 316 forwards in about 222 seconds. A separate review rebuilt the native model-input hashes, checked the fitted objectives and gradients, and reproduced 840 decisions and 30 aggregate cells. The public reviewer recreates the location and outcome counts from de-identified decision flags and sample ranges. The waveform, learned feature, score and weight files stay local.

```sh
python3 -I -S backend/scripts/review_audio_partial_residual_public.py \
  docs/evaluation-panels/audio-partial-residual-transfer-2026-11-12.json
```

Use a new `run` directory with the pinned local v81/v82 source and audio paths for the native replay. Use the exact installed voice asset directory as `weights`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze build verify; do
  backend/.venv/bin/python -B -m scripts.audio_partial_residual_inputs "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_partial_residual_transfer freeze "$run" "$weights"
for role in fusion_fit confidence_cal; do
  backend/.venv/bin/python -B -m scripts.audio_partial_residual_transfer encode "$run" "$weights" --role "$role" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_partial_residual_transfer fit "$run" "$weights"
for role in selection confirmation; do
  backend/.venv/bin/python -B -m scripts.audio_partial_residual_transfer encode "$run" "$weights" --role "$role" || exit 1
done
backend/.venv/bin/python -B -m scripts.audio_partial_residual_transfer report "$run" "$weights"
backend/.venv/bin/python -B -m scripts.review_audio_partial_residual_transfer "$run" "$run/audit.json"
```

For the separate no-refit meeting replay, use the earlier local overlap-window output folder as `meeting_parent`.

```sh
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.audio_partial_residual_room_replay "$phase" "$run" "$meeting_parent" "$weights" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_audio_partial_residual_transfer "$run" "$run/audit-plus-room.json" --meeting-parent "$meeting_parent"
```

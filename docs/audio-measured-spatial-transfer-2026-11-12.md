# Measured-room cancellation for short voice edits

Measured-room cancellation improved foreground SNR and gave the same voice check one useful location on six held edits at the frozen points, with two false genuine groups in each arm. The original audio gave zero useful locations. The method needs the measured room and the target and competing playback positions. These inputs are available for the engineered mixtures in this study. [Evaluation receipt](evaluation-panels/audio-measured-spatial-transfer-2026-11-12.json)

## Sources and fixed comparison

The selection role uses the University of Surrey front and lateral measurements and two annotated AMI meeting sessions. The held role uses BBC Research & Development front and lateral measurements and two different AMI sessions. Both use the original publisher sample order and complete 48 kHz impulses. We resample those impulses to 16 kHz and retain every response coefficient. The Surrey measurements are by Jon Francombe at the University of Surrey, licensed CC BY 4.0. The BBC static WAVs are by Chris Pike, Michael Romanov, and BBC R&D, licensed CC BY-SA 4.0 at the pinned repository correction. [Surrey dataset](https://openresearch.surrey.ac.uk/esploro/outputs/dataset/IoSR-Listening-Room-Multichannel-BRIR-Dataset/99512517702346) · [BBC asset and rights qualification](audio-bbc-room-assets-v303-method.md) · [AMI terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

Each role contains six six-second edited examples with two clone, three expressive, and one stock insert. Every edited example has a genuine same-reader splice, a genuine other-reader splice, and an untouched control. Two further examples contain the room feed alone. Both roles use newly selected exact utterances, generated fragments and meeting intervals outside the earlier replication. The source banks have prior project exposure. The first two intake attempts stopped before freezing or measuring outcomes because the retained bank lacked a fourth unused stock voice. The balanced 2/3/1 roster retained all exact-source, reader, and transcript exclusions. [Source and model qualifications](audio-remote-neighbor-replication-2026-11-12.md)

The target stays in the front position. All competing speech comes from one lateral playback feed, with 410–960 ms of overlapping annotated speech from at least two talkers within each authored insert. Both matched render arms use the same full-length PCM and 16 kHz clock. The original arm averages the two channels. The calibrated arm uses the measured front response to cancel the target from a side channel, then uses the measured lateral response to reduce interference in the average. The filter uses at least two seconds of zero padding on either side, retains the original sample count, and receives no isolated speech stem or authored edit coordinates. The wrong-direction and wrong-room tests use the same calculation with different measured responses. [Transfer-function cancellation study](https://www.sciencedirect.com/science/article/abs/pii/S0167639306001932) · [Steering uncertainty study](https://www.sciencedirect.com/science/article/pii/S0003682X24004109)

## Acoustic outcome

| Median across six edited recordings per role | Surrey selection | BBC held |
| --- | ---: | ---: |
| Whole-clip SNR gain with matching calibration | +7.28 dB | +5.97 dB |
| Exact edit-interval SNR gain with matching calibration | +8.67 dB | +9.81 dB |
| Exact edit-interval SNR change with wrong side | −4.67 dB | −4.61 dB |
| Exact edit-interval SNR change with wrong rig | −1.60 dB | −2.17 dB |
| Isolated target preservation | 43.28 dB | 48.04 dB |

Both roles cleared the frozen median requirements of +4 dB whole-clip gain, +3 dB within the edit, and 35 dB target preservation. The 52 original and calibrated signals had zero clipped samples. A separate implementation re-decoded the room responses, reconstructed the stereo mix and filtered PCM, and matched all 260 committed PCM stems. The alternate FFT implementation reproduced every conditioned output sample after rounding. Independent AMI annotation replay also verified the edit overlap and distinct participant groups.

## Voice comparison

The pinned voice model and speech gate read the two-second sample windows at 1–3 and 3–5 seconds. Only genuine selection examples set the two fixed cutoffs. The shared-time arms keep a window when both original and calibrated audio passed the speech gate on it. The useful-location criterion is at least 0.30 temporal IoU and half of the edit covered. The wrong-direction and wrong-room diagnostics measure acoustics. [Pinned voice model and speech-activity assets](audio-pacc-portable-replay.md)

| Six held edits and 20 genuine examples | Original | Calibrated |
| --- | ---: | ---: |
| AI flags at frozen cutoff | 1 | 1 |
| Useful locations at frozen cutoff | 0 | 1 |
| False genuine groups at frozen cutoff | 2 | 2 |
| Useful locations at 0 / 1 / 2 equal false groups | 0 / 0 / 0 | 0 / 1 / 2 |
| Useful strongest window before cutoffs | 3 | 5 |
| Unassessed files | 0 | 1 |

The fixed-point location was a clone insert at 3.84–4.80 seconds. The two-second output at 3–5 seconds covered the whole edit at 0.48 IoU. Both arms chose that window, and the calibrated score passed its frozen point. At two retrospective false groups, calibration also found a clone insert at 1.28–2.08 seconds. Its output moved from the original arm's 3–5 second window to 1–3 seconds and covered the whole edit at 0.40 IoU. Both gains held when ranking only the windows admitted in both arms. The earlier fixed original-audio point yielded zero locations and zero false groups on this held set.

Neither result generalized to the three expressive or one stock edit at the held cutoffs. Calibration ranked two of six synthetic splices above their matched other-reader controls. The original ranked three above those controls. Neither arm flagged a room-only example. A room-only example became unassessed with calibration.

The live Full profile's pre-admission grid for a six-second file is 0–4 and 2–6 seconds. Even a perfect choice between those four-second windows reaches a maximum time IoU of 0.24 on these 0.80–0.96-second edits. This offline study used narrower windows and supplied the room, target, and side measurements. The holdout supports further testing in known microphone configurations.

## Replay

With the previously pinned local asset archives present, set `run` to a new private evaluation directory. The runner also accepts `--archive` when the measured room assets live under a different mount. Each role has its own exact score ledger and only the selection role sets the two operating points.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze selection fit held report audit; do
  backend/.venv/bin/python -B -m scripts.audio_measured_spatial_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B backend/scripts/review_audio_measured_spatial_transfer.py "$run" "$archive" \
  backend/data/independent-evaluation/spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip
```

The first scoring launch stopped before a model forward because the draft and model loader both set the Torch interop thread count. Removing the duplicate call reproduced the original roster byte for byte. The scored run made 98 selection and 100 held CPU voice forwards. The native audit replayed the exact source and result ledgers, and the independent reader checked all 208 candidate window bindings, including skipped windows. The public receipt contains aggregate results, byte commitments, and the method. The source recordings, engineered audio, and individual scores stay local. [Runner](../backend/scripts/audio_measured_spatial_transfer.py) · [Independent reviewer](../backend/scripts/review_audio_measured_spatial_transfer.py)

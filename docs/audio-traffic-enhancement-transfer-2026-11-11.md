# Speech enhancement under heavy traffic

The combined rule recovered 44 synthetic voice flags across all nine synthetic groups and all four traffic recordings. It also added 39 false AI flags across nine of ten genuine reader groups and all four recordings. This exceeded the frozen false-alarm limit. The existing voice analysis stays in service. [Exact aggregate and freeze](evaluation-panels/audio-traffic-enhancement-transfer-2026-11-11.json)

## Frozen comparison

The earlier two-view rule added five true and five false AI flags. It also left 35 of 48 synthetic traffic mixtures unassessed. This experiment tests whether a learned speech enhancement view can recover assessable speech and improve downstream decisions. The earlier v4 study used the same enhancer family on previously consumed, milder inputs and always took speech admission from the original view. Its frozen criteria selected no arm for confirmation. The present comparison keeps the current classifier, calibration and activity gate fixed and tests whether allowing the enhanced waveform through that same gate makes a difference. [Earlier heavy-traffic transfer](audio-strong-traffic-view-transfer-2026-11-11.md) · [Earlier enhancement experiment](audio-noise-view-v4-development.md) · [Earlier result](audio-noise-view-v4-evaluation.json)

The pinned author-distributed streaming GTCRN ONNX export runs locally at 16 kHz on CPU. It applies a learned complex mask to 512-sample square-root periodic Hann frames at a 256-sample hop. It uses reflect padding and resets recurrent state for each input. The output remains 48,000 samples, quantized to signed 16-bit without level normalization. The repository includes the checkpoint under its root MIT grant. The test binds the actual model and license bytes. The test does not establish training-source independence or validate the enhancer as a forensic reconstruction. Speech enhancement can improve intelligibility while changing features used by a detector. [Pinned checkpoint](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/stream/onnx_models/gtcrn_simple.onnx) · [Publisher license](https://github.com/Xiaobin-Rong/gtcrn/blob/502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73/LICENSE) · [Speech enhancement and detection study](https://arxiv.org/abs/2603.14767)

The 19 foreground groups include ten new LibriSpeech test-clean readers and nine prepared synthetic voices, three each from clone, expressive and stock cohorts. Known foreground identities and exact PCM are disjoint from the preceding traffic and two-background transfer rosters. The synthetic voices had earlier exposure elsewhere in the project. Each foreground has an exact three-second clean reference and eight mixtures, one at −5 dB and one at −10 dB for each of four independently recorded street scenes. A common gain keeps each mixture within the signed sample range. Three background recordings and their provisional slices had prior exposure. The fourth uses eight preselected three-second slices from Cori Samuel’s public-domain street recording. It contains traffic and other outdoor sounds. Two isolated excerpts per tape bring the total to 179 exact views. Source selections and all model inputs were fixed before inference. [LibriSpeech and CC BY 4.0](https://www.openslr.org/12/) · [Synthetic source preparation and terms](audio-short-edit-source-plan-v169.md) · [Prior traffic qualifications](audio-commons-slice-screen-v247-result.md) · [New recording and terms](https://commons.wikimedia.org/wiki/File:Sunday_in_the_city_street_noise1.ogg)

Apply the current full-profile speech gate and classifier to both original and enhanced samples. The combined test preserves current directional results and adds an AI flag only when the current direction is inconclusive or unassessed, the enhanced view qualifies and leans AI, and enhancement reduces total signal energy by at least 6 dB. The energy change is measurable from the input and enhanced output. Independent acoustic assessment compares both views to the known clean source with scale-invariant signal-to-distortion ratio (SI-SDR). The original clean reference is never a model input for its noisy counterpart. Advancement requires six additional correct AI flags across three foreground groups and two tapes, at most one added genuine false flag, no added clean or isolated flags, and at least three enhanced admissions for previously unassessed synthetic speech across two tapes. The acoustic minima are 12 dB median clean SI-SDR and 3 dB median noisy gain in at least three tapes at each noise level. [SI-SDR method](https://arxiv.org/abs/1811.02508) · [Frozen runner](../backend/scripts/audio_gtcrn_traffic.py)

## Local results

| Planned group | AI flags, current → enhanced → combined | Unassessed, current → enhanced → combined |
| --- | ---: | ---: |
| Clean genuine (10) | 0 → 0 → 0 | 0 → 0 → 0 |
| Clean synthetic (9) | 8 → 9 → 8 | 0 → 0 → 0 |
| Genuine −5 dB (40) | 1 → 27 → 25 | 11 → 2 → 5 |
| Synthetic −5 dB (36) | 5 → 28 → 29 | 11 → 2 → 3 |
| Genuine −10 dB (40) | 1 → 17 → 16 | 27 → 12 → 20 |
| Synthetic −10 dB (36) | 0 → 20 → 20 | 24 → 11 → 15 |
| Isolated background (8) | 0 → 0 → 0 | 8 → 8 → 8 |

Each synthetic case without an AI flag was inconclusive or unassessed. The combined rule missed seven of 36 synthetic mixtures at −5 dB and 16 of 36 at −10 dB. Enhancement by itself admitted 22 previously unassessed synthetic mixtures across all four tapes. The new street recording produced nine additional genuine false flags. False flags also rose in the three archived recordings. The result applies to controlled three-second additive mixtures. Voices, tapes, and conditions repeat within their matched groups.

The known-reference median SI-SDR improvement ranged from 6.71 to 12.90 dB by tape and noise level. Clean median SI-SDR was 22.68 dB. The enhancement met both acoustic criteria without clipping. On inputs assessed in both arms, the raw score's genuine-versus-synthetic ROC AUC changed from 0.676 to 0.582 at −5 dB and from 0.622 to 0.686 at −10 dB. These selected subsets contained 54 and 25 inputs, respectively. The extra acoustic clarity and larger speech coverage do not establish that the unchanged voice scores have useful calibration on the enhanced inputs.

The run completed 33,840 enhancement frame forwards, 33,652 speech-activity frame forwards, and 266 classifier forwards including stability probes. Preparation took 5.44 seconds, enhancement took 17.96 seconds, and scoring took 191.53 seconds on CPU. The final-code replay matched every original and enhanced PCM file, acoustic measurement, classifier response and decision from the initial completed run. The model-free auditor checked both sample commitments for every input, reconstructed all 152 mixtures sample-for-sample, and reproduced 358 speech decisions and the advancement result. The largest SNR error was 0.000066 dB. An initial preparation attempt stopped before case generation or inference because it read the prior freeze hash from the wrong level of the public receipt. The completed runs verified the hash at its recorded location. [Independent arithmetic auditor](../backend/scripts/review_audio_gtcrn_traffic.py)

## Local replay

Use the same Python backend environment, ONNX Runtime 1.23.2, FFmpeg, current local voice assets, archived source preparation, and two prior private run directories. In a fresh working directory, save the pinned ONNX file as `gtcrn_simple.onnx`, the publisher MIT notice as `LICENSE`, the new Wikimedia audio as `sunday.ogg`, and the matching Wikimedia `imageinfo` reply with `sha1`, `url`, `size`, and `extmetadata` as `sunday-source.json`. The source and model identities are bound before any model calls.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_gtcrn_traffic freeze "$run_dir" \
  --librispeech "$test_clean_archive" --prior-traffic "$prior_traffic" --prior-background "$prior_background"
backend/.venv/bin/python -B -m scripts.audio_gtcrn_traffic enhance "$run_dir"
backend/.venv/bin/python -B -m scripts.audio_gtcrn_traffic score "$run_dir"
backend/.venv/bin/python -B -m scripts.audio_gtcrn_traffic readout "$run_dir"
backend/.venv/bin/python -I -B backend/scripts/review_audio_gtcrn_traffic.py "$run_dir"
```

Use `--published` with the public aggregate to verify the exact retained run. Individual PCM, recordings, source IDs, checkpoint bytes and raw model scores stay local.

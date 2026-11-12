# An independent check on false AI speech flags under noise

An earlier two-room study found that the independent voice check ranked overlapping speech better than the incumbent, while its rescue rule added zero correct AI flags. Its rescue rule could only add AI flags. This experiment uses the independent score to test the opposite decision, withholding an existing AI flag when the second check gives a sufficiently low spoof response. [Earlier two-room study](audio-independent-background-transfer-2026-11-11.md)

## Frozen comparison

Use the existing Apache-2.0 Spectra-AASIST3 CPU INT8 derivative at revision `b56aed04853cb4e5bf825025c54c93d4bc345c61`, with the previous 0.97 pre-emphasis, repeated input preparation and spoof-logit direction. Both checkpoints assess the same decoded three-second mono waveform. The incumbent uses its saved production-profile speech gate and calibration. The independent check runs only on incumbent AI flags. A response strictly below the frozen point changes that direction to inconclusive. The experiment adds no AI flags and estimates no customer probability. [Pinned derivative and terms](https://huggingface.co/Limitless-8/spectra-aasist3-int8-audio-deepfake/tree/b56aed04853cb4e5bf825025c54c93d4bc345c61) · [Current voice checkpoint terms](commercial-model-audit.md)

Choose the highest observed point that preserves all clean and all but one noisy synthetic flag in the completed earlier Spectra comparison. The point uses responses from a separate cohort with 17 declared foreground groups and two AMI meetings. The held role uses the 116 native deliveries of the later stereo transfer, assessed on the incumbent mono downmix. It includes 16 clean foregrounds, 64 centered and 32 offset mixtures, and four natural room controls. The mixture input is the exact submitted PCM. The independent check receives no isolated stem. Reuse another 96 exact delivered views from the phase study, comprising clean and simultaneous centered mixtures round-tripped through Opus or a fixed denoiser and compressor. These deliveries repeat the held foregrounds. Exact clean PCM hashes, declared foreground groups, and AMI meeting IDs are disjoint between selection and held roles. Both roles use retained material with prior project exposure. [Held stereo sources and terms](audio-spatial-interference-2026-11-11.md) · [Exact delivery variants](audio-phase-corroboration-2026-11-11.md)

To advance, the rule must clear three or more centered false flags across at least two foreground groups and both rooms. It can lose at most two native centered synthetic flags, one native offset flag, and one noisy flag per alternate delivery. Every clean AI flag must remain in each delivery. Any successful transfer would still need a larger confirmation across independently sourced rooms and generators.

## Local results

The selection point preserved all seven clean and nine noisy synthetic flags. It also retained all 15 genuine false flags on the selection panel. It cleared two genuine flags in the held native centered mixtures, both in one room and from different foreground groups. The same comparison lost five synthetic flags, and it lost more after Opus delivery and restoration. A withheld result becomes inconclusive.

| Held delivery | Genuine false AI flags, incumbent → combined | Synthetic true AI flags, incumbent → combined |
| --- | ---: | ---: |
| Clean native | 0/7 → 0/7 | 9/9 → 9/9 |
| Native centered mixtures | 6/28 → 4/28 | 18/36 → 13/36 |
| Native offset mixtures | 1/14 → 1/14 | 3/18 → 1/18 |
| Opus centered simultaneous mixtures | 3/14 → 1/14 | 9/18 → 3/18 |
| Restored centered simultaneous mixtures | 6/14 → 2/14 | 14/18 → 4/18 |
| Clean restored | 0/7 → 0/7 | 9/9 → 8/9 |

All these rows were assessed by the incumbent. Clean Opus retained nine of nine synthetic flags and zero of seven genuine flags. The four native room controls retained two original, one inconclusive and one unassessed result. The frozen veto failed six of its seven advancement checks. These measurements describe two held rooms and repeated delivery variants of the same foregrounds. They show that better ranking in the earlier noisy panel did not establish a transferable score boundary. [Aggregate and independent replay](evaluation-panels/audio-independent-noise-veto-2026-11-11.json)

## Reproduce

The local CPU run used ONNX Runtime 1.23.2 and made 88 model forwards, including one exact repeat. It took 67.7 seconds with 0.89 GiB peak process RSS. The separate verifier reopened the original PCM or delivered media, rebuilt the exact graph inputs with independent preprocessing arithmetic, and reproduced the point, counts and decision. Raw audio, models and per-case readings remain local.

Use the completed Spectra comparison directory as `selection`, the completed spatial-interference run as `held`, the completed phase-corroboration directory as `delivery`, and the existing pinned Spectra checkpoint and publisher records as `assets`. Run the three measurement stages in a new directory with ONNX Runtime importable in the backend environment. The verifier uses FFmpeg and the standard library.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_independent_noise_veto freeze "$run" \
  --selection "$selection" --held "$held" --delivery "$delivery" --assets "$assets"
backend/.venv/bin/python -B -m scripts.audio_independent_noise_veto score "$run" \
  --selection "$selection" --held "$held" --delivery "$delivery" --assets "$assets"
backend/.venv/bin/python -B -m scripts.audio_independent_noise_veto report "$run" \
  --selection "$selection" --held "$held" --delivery "$delivery"
backend/.venv/bin/python -B -m scripts.review_audio_independent_noise_veto "$run" "$selection" "$held" "$delivery"
```

# Track changing background noise before a speech recheck

An adaptive noise profile added no useful speech edit locations and no false flags on 98 held recordings. The check stayed experimental. [Saved window responses and input commitments](evaluation-panels/audio-local-noise-profile-transfer-2026-11-12.json)

| Held speech condition | Current focused review | With adaptive noise profile | Extra genuine file flags |
| --- | ---: | ---: | ---: |
| Gain-only | 13/18 | 13/18 | 0/12 |
| Rain, −5 dB | 3/18 | 3/18 | 0/12 |
| Train, −5 dB | 0/18 | 0/18 | 0/12 |

## Frozen comparison

The earlier spectral subtraction estimated one noise floor for the entire recording. This comparison estimates a separate floor for every frequency and frame from its local 151-frame neighborhood, about 1.21 seconds. The floor is the 20th percentile of power. A 512-sample Hann window advances by 128 samples. Subtract 1.25 times the floor in power, keep at least 60% of amplitude and smooth over five time frames and three frequency bins. The reconstructed signal keeps the original phase and sample clock. No enhancement model or external code was used. [Earlier whole-recording noise profile](audio-noise-window-transfer-2026-11-12.md) · [Research on component-level audio testing](https://arxiv.org/abs/2608.09593)

Only inconclusive or unassessed original dense windows entered the new path. The usual activity check had to admit the new view. The operating point was the highest of 23 admitted responses from the 25 genuine and isolated selection files. A new range requires two adjacent positive responses with at least one from the adaptive view. The current ranges remain intact. The two study roles are source separated, and the exact held files have prior project exposure. [Source recordings and terms](audio-subband-partial-transfer-2026-11-12.md)

On the held files, 150 of 250 filtered windows passed speech activity. None exceeded the frozen point. All eight isolated background excerpts stayed unassessed. These are two particular background recordings with repeated speech sources across conditions.

## Replay

The public ledger reader recalculates the operating point and counts. Native replay also checks the exact filtered inputs.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.audio_local_noise_profile_transfer review \
  docs/evaluation-panels/audio-local-noise-profile-transfer-2026-11-12.json
```

Recreate `selection` and `held` using the [source instructions](audio-subband-partial-transfer-2026-11-12.md). Set `run` to a new directory. Inference uses the pinned local voice and activity assets.

```sh
export PYTHONPATH=backend VERITAS_AUDIO_DETECTOR_MODE=candidate VERITAS_BILLING_MODE=disabled
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_local_noise_profile_transfer run "$run" \
  --selection "$selection" --held "$held"
backend/.venv/bin/python -B -m scripts.audio_local_noise_profile_transfer review "$run/receipt.json" \
  --selection "$selection" --held "$held"
```

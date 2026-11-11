# Downstream scoring on speech recovered under strong traffic noise

Use every v269 fallback rescue at traffic −10 dB SNR and every original-gate-admitted comparison in the same two foreground arms. The roster includes one rescued genuine group and two rescued Alania groups, plus seven admitted genuine and 13 admitted Alania groups. Each group contributes its own baseline-admitted gain-matched clean waveform and its exact noisy waveform. The study scores 46 waveforms. It leaves Drifting and the remaining gate-rejected rows outside this downstream comparison. [Fixed strong-noise challenge](audio-noise-speech-gate-strong-noise-v269-method.md)

Verify the completed v269 freeze, resource receipt and saved speech eligibility for the entire source roster. Require every selected group and its own matched clean source. Preserve the original speech-gate outcome separately from the downstream classifier direction. Wrap each exact signed-16-bit payload in a mono 16 kHz WAV and use the same local decoder, classifier, reference calibration and stability probe as v261. Run one guarded offline classifier stage, with zero activity or second-opinion model calls and at most two classifier calls per waveform. The study leaves product behavior and calibration unchanged. [Earlier downstream comparison](audio-rescued-speech-retry-v261-method.md)

Report counts for leaning AI, leaning original and inconclusive by foreground arm, noisy admission group and condition. The two views per source group are paired and all noisy waveforms share one tape. Selection conditions on the observed activity outcomes. Treat the results as descriptive downstream behavior on this cohort. Keep individual waveforms, scores, speaker groups and spans private.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_noise_speech_rescue_downstream_v270 preflight
backend/.venv/bin/python -B -m scripts.audio_noise_speech_rescue_downstream_v270 freeze
backend/.venv/bin/python -B -m scripts.audio_noise_speech_rescue_downstream_v270 run
backend/.venv/bin/python -B -m scripts.audio_noise_speech_rescue_downstream_v270 readout
```

# Stronger-noise challenge for the fixed speech fallback

The completed v267 transfer found no coverage loss at 0 or +5 dB SNR on the prepared foregrounds. This separate stress test fixes two stronger levels, −5 and −10 dB, before scoring. It keeps the v267 foreground roster, source bytes, matched tape offsets, exact shared-residual construction and activity model. The same clips recur across both studies. Treat v267 as development history for the stronger-noise challenge. [Earlier transfer method](audio-noise-speech-gate-transfer-v267-method.md)

Recreate the four noise conditions and a clean control with the new common clipping-safe gain. Verify both strong-noise levels against the existing 0.1 dB post-quantization tolerance. Keep the known source-unavailable rows in the planned denominator. Record any new unsupported mix separately. Replay the v267 result and bind the stronger-noise inputs to the exact former source and tape excerpt fingerprints. The earlier 102 background-only controls remain historical. This run makes no new calls on those unchanged controls and establishes no new background false-alarm rate. [Earlier transfer result](evaluation-panels/audio-noise-speech-gate-transfer-complete-v267.json)

Use the current serving window builder and the same fixed fallback as v256. The original rule requires half of the three-second waveform to fall inside returned speech spans. The fallback also accepts one continuous returned span of at least one second. Report the number of newly admitted windows and their source-group counts per arm, environment and SNR. Preserve exact sample ranges and per-window identities privately. The result measures speech admission under stronger noise, with zero authenticity classifier calls and no serving or calibration change. [Frozen fallback](audio-noise-speech-gate-v256-method.md)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_strong_noise_v269 preflight
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_strong_noise_v269 freeze
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_strong_noise_v269 run
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_strong_noise_v269 readout
```

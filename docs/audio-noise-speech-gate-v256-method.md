# Paired speech admission under recorded background noise

Replay the exact 355 prepared three-second audio variants from v250 through the unchanged serving speech-activity model. Keep all 360 planned rows, including the five source-unavailable variants. Bind the published v250 result, freeze, and every saved serving batch receipt. Confirm the existing window builder reproduces all saved admission outcomes. Measure detected speech spans on the same waveform bytes. [Prior comparison](audio-matched-background-score-v250-result.md)

Compare the serving gate, which requires half the window to lie in returned speech spans, with one fixed rescue. Keep every currently admitted window. Also admit the original three-second waveform if the activity model returns a single span of at least one second. One second matches the serving window duration floor. Returned spans include the model's usual gap joining and padding. Count rescued inputs by foreground arm and noise condition, and count coverage losses among genuine speakers who passed on the matched clean input. Keep skipped and unavailable outcomes in their own denominators. No authenticity classification or confidence fitting is part of this diagnostic.

Check for additional admissions on complete three-second windows from the seven qualified background recordings. Use every window on the original nonoverlapping grid that was outside the fixed v247 screen roster. Exclude exact PCM duplicates of screened windows. Selection uses position and bytes only. This avoids conditioning the controls on the same speech model's earlier decisions. The new cuts share recordings with the main panel and could contain speech, so report background admissions by recording group and environment without treating them as proven false positives. [Earlier screen](audio-commons-slice-screen-v247-method.md)

Keep exact spans, window coordinates, fingerprints and the observation journal private. The public result contains counts and whole-artifact commitments. Run the existing local neural resource guard, with only the pinned activity model loaded. Preserve the run profile and reconstruct the aggregate from its exact private observations. The shadow policy leaves customer scoring unchanged. Recovered genuine coverage must be weighed against extra background admissions and independently checked downstream voice error rates before considering use.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_v256 preflight
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_v256 freeze
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_v256 run
backend/.venv/bin/python -B -m scripts.audio_noise_speech_gate_v256 readout
```

# Recover activity-matched crops from the same RUEG participants

Replaying all 15 unavailable v246 recordings reproduced the exact quiet-boundary rejection in each case. The original files and complete production decoding had passed. Preserve that result and use the same 24 frozen source recordings in a separate preparation. Keep the earlier nine exact crops and the existing crop durations, activity threshold, ranking and minimum active fraction. For the exact no-pause rejection only, rerun the same crop selector with zero required quiet frames at the boundaries. [Original intake](audio-rueg-genuine-v246-result.md) · [Existing crop rule](audio-voxforge-tr-genuine-v199-method.md)

This fallback admits continuous high-activity excerpts. It can cut an utterance mid-word. Preserve that distinction in the private evidence. Verify the source clock, exact waveforms, known byte and PCM exposure checks and all prior successful crops. The fallback makes zero model calls and keeps the same participant roster. The later fixed speech route still decides whether each mixed clip contains assessable speech.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1
backend/.venv/bin/python -B -m scripts.audio_rueg_activity_crop_v252 freeze
backend/.venv/bin/python -B -m scripts.audio_rueg_activity_crop_v252 qualify
backend/.venv/bin/python -B -m scripts.audio_rueg_activity_crop_v252 readout
```

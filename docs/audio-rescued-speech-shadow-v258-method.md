# Shadow scoring for speech recovered by the admission test

Verify the published v256 admission result against its guarded private observations and exact matched inputs. Select every three-second variant that the original gate skipped and the one-second-span fallback would admit. The frozen result yields six genuine noisy variants. Two contain kitchen noise at 0 dB SNR, one contains kitchen noise at +5 dB, and three contain traffic at 0 dB. Keep the waveform and speaker bindings private. [Admission method](audio-noise-speech-gate-v256-method.md)

Send each original 48,000-sample waveform through the existing serving classifier and its frozen calibration with the ordinary whole-window input and stability probe. The saved eligibility determines admission for this experiment. No new speech-activity forward, waveform trimming, score fitting, or second-opinion encoder runs. The resource guard keeps the scoring stage offline and records at most two classifier forwards per variant. Report the exact counts for leaning AI, leaning original and inconclusive by noise condition, plus the number of genuine source groups with false AI calls.

These variants come from the same development panel used to identify the coverage gap. They evaluate the classifier's behavior on rescued genuine speech only. Any improvement in conclusive coverage and any false AI call must be read with that scope. Separate source groups and known speech-free negatives are needed before changing serving behavior. The public output contains aggregate counts and whole-artifact commitments.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_shadow_v258 preflight
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_shadow_v258 freeze
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_shadow_v258 run
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_shadow_v258 readout
```

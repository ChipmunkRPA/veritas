# Shadow speech scoring retry

The v258 worker stopped before classifier construction. Both the worker and the existing voice loader configured PyTorch interop threads, and the second call raised an error. This successor verifies the original freeze, private input plan, failure log and terminal resource receipt. The original attempt remains recorded without classifier outcomes. [Original method](audio-rescued-speech-shadow-v258-method.md)

The retry uses the same six waveform bytes, the same classifier loader, and the same calibration and stability evaluator. The existing loader alone configures CPU threads. The original guarded, offline scope remains in force. Keep the per-variant scores and original failure log private, and report aggregate score directions and failure receipt commitments. This is a development-panel replay.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_retry_v261 preflight
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_retry_v261 freeze
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_retry_v261 run
backend/.venv/bin/python -B -m scripts.audio_rescued_speech_retry_v261 readout
```

# Background-noise false alarms on matched genuine speech

Score the 98 exact production WAVs admitted in
[PACC-P preparation v176](audio-pacc-sample-v176-result.md). Each microphone
input contributes its clean recording and the publisher's matched babble,
environmental noise and instrumental music versions at nominal 0 and +5 dB.
The nominal levels refer to full publisher clips. They are not measurements
of SNR within the first three seconds used by the scorer.

Verify the v176 guarded resource profile, published receipt, input roster,
WAV hashes, zero-origin sample range and production 16 kHz PCM hashes. Use
the same frozen full-profile voice route, VAD, calibration, local model
assets and original-side stability check as
[v174](audio-matched-edit-diagnostic-v174-method.md). Run one raw prediction
per available speech window and count any serving-requested stability
forwards separately. Record windows rejected by the speech gate as
unavailable. No repetition cue, new decision threshold or model fitting is
used.

Report the four AMI meeting contexts separately from five VCTK utterances
from previously catalogued speakers. VCTK supplies two microphones per
utterance, so report each microphone separately as well as the pooled VCTK
counts. The two microphones remain dependent. Each noise condition compares
the same microphone input with its clean result. Report the four decision
counts, speech gate status, scored and decided coverage, false AI tendencies
and all clean-to-noisy direction and route-status transitions. False-alarm,
transition and coverage rates use the planned case or paired denominator,
including inconclusive and unavailable results. A false AI tendency that
disappears into an inconclusive or unavailable result lowers the false AI
count without establishing an original tendency. Repeated variants and the
small numbers of underlying contexts limit generalization. This panel
contains genuine controls only and cannot measure misses on synthetic speech.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_false_alarm_v177 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_false_alarm_v177 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_pacc_false_alarm_v177 readout
```

The model worker runs offline under the existing v108 resource monitor.
Per-input scores and source metadata stay in the private run. The published
readout reports aggregate counts and commitments without source identifiers.

# Development-only short speech edit diagnostic

The diagnostic uses the verified v169c development manifest and its exact
three-second mono PCM WAVs. It checks the published source result against
the actual completed v108 binding receipt, then verifies each admitted
development WAV, decoded sample bytes, and zero-origin WAV sample clock.
The source boundary allows direct development WAVs and rejects links and
paths outside that directory. Holdout WAV access and scoring are prohibited
in this diagnostic. [Preparation result](audio-short-edit-materialization-v169c-result.md)

The planned denominator is 79 cases. Fourteen genuine recordings each
supply the untouched excerpt and three same-source 320 ms copy variants
with a hard boundary or a contained 20 ms or 50 ms fade. The destination is
samples `[17280,30080)` and the copied original donor is `[12160,24960)`.
The 23 AI recordings supply untouched cases across clone, expressive and
stock families. All cases contain 48,000 samples at 16 kHz when constructed.
Unsupported construction and unavailable speech windows retain their rows.

Use the exact v167 scoring function, mapped CPU model load, and existing
speech-window gate. Compare the original calibrated direction with the v167
symmetric band probe and an independent repetition-based comparator. For
the latter, a measured exact or approximate repeated range can change an
AI tendency to inconclusive. The comparator keeps every other direction.
The range check runs on the unchanged case PCM. Authored copy support is
used only to count matches during evaluation. Repetition is an inspection
cue. These comparisons involve no new calibration fit or customer rule.
[Score method](grid-audio-splice-probe-v167-method.md) · [Cue method](grid-audio-repetition-codecs-v170-method.md)

The guarded worker writes a private per-case journal and an aggregate
result. The readout verifies its v108 resource journal and model-forward
count, case order and summary. Public counts retain all directions and
cue outcomes by family and arm. Audio bytes, sample ranges, per-case
scores, file names and source identifiers stay private.

Run from the repository root.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_diagnostic_v173 freeze
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_diagnostic_v173 run
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_short_edit_diagnostic_v173 readout
```

This small diagnostic measures tradeoffs on development material only.
The future paired synthetic-insert and holdout comparison is needed to
estimate transfer to that setting.

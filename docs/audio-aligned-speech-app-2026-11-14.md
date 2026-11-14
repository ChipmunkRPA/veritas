# Locate speech changes against an earlier recording

Chipmunk Verify can now align two short recordings of the same speech and
show where they diverge. The earlier and analyzed copies each get their own
timestamps, waveform, and playable excerpt. The app shows the measured
difference inside the interval and the lower differences on either side.
The PDF and JSON use the same source samples. The JSON also includes both
excerpts and their SHA-256 hashes.

The trial needs continuous 16 kHz mono 16-bit WAV or FLAC files of
5.8 to 12 seconds. Both time origins must be verified. The files can have
different durations and modest timing differences. When a single gain or
filter with optional uniform resampling explains the clips, the check
suppresses the local flag. Re-recording and ordinary edits can also produce
differences. This evidence contributes zero weight to AI confidence.

## Held comparison

The test keeps the prior alignment settings and the processing guard.
Two separate SINE panels provided 27 labeled synthetic speech splices from
27 readers, excluding the preview's original reader. A further 12 same-word
resyntheses do not have a labeled changed-speech interval. The tests reuse
the same edits across delivery conditions. The panels were previously used
for different localization checks and share a publisher generator.

A useful location covers at least half the marked edit and reaches 0.30
temporal intersection over union. The previous app result includes exact
local differences, the dense same-clock speech comparison, and local
audio timing. Whole-recording differences do not count as localization.
[Prior alignment method and frozen operating point](audio-aligned-speech-difference-2026-11-13.md)
· [Source terms and inputs](audio-sine-resynthesis-transfer-2026-11-14.md#sources-and-controls)
· [Additional held readers](audio-narrow-flag-transfer-2026-11-14.md#new-readers-from-the-publisher-archive)

| Outcome | Current comparison | Alignment alone | Combined |
| --- | ---: | ---: | ---: |
| Clean synthetic splice located | 4/27 | 21/27 | 24/27 |
| Noise-masked synthetic splice located | 0/81 | 0/81 | 0/81 |
| Ordinary pair with a local flag | 4/312 | 0/312 | 4/312 |
| Same-word resynthesis with a local flag | 0/48 | 0/48 | 0/48 |

The added locations covered 20 readers. On 195 clean ordinary pairs,
alignment had enough similar speech on both sides to measure a candidate
in 151. On the 117 ordinary noise-masked pairs it had no measurable
candidate. It also left all 81 noise-masked splices unassessed.
Rain, train, and competing speech were mixed into the submitted copy
at 5 dB above the foreground level. The held result supports a local
earlier-copy trial for cleaner speech. Standalone detection and the
overall confidence score were outside the comparison.
[Per-file measurements](evaluation-panels/audio-aligned-speech-app-transfer-2026-11-14.json)

## Local flow

Set `VERITAS_AUDIO_REFERENCE_ALIGNED_SPEECH=true` alongside
`VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH=true`. The trial setting
defaults to off. In a completed retained audio analysis, open
**Compare with an earlier audio recording** and select
**The earlier file is a clean copy of the same recording**.

Three simulated USD 9.99 checkouts covered an edited recording, an
unchanged recording, and the edit masked by train noise. No analysis job
started before payment and none of the comparisons changed the saved
assessment. The clean example located **2.400–3.120 s** in the analyzed
file and **2.340–3.020 s** in the earlier copy. The API, PDF, JSON and
source replay agreed on the recorded intervals and waveforms.
[Prepaid app verification](evaluation-panels/audio-aligned-speech-app-checkout-2026-11-14.json)

## Replay

The numeric recount uses standard-library Python. For native replay,
prepare both SINE banks using the source links above and obtain the
two [pinned backgrounds](audio-focused-window-transfer-2026-11-12.md#frozen-comparison).
Use a new output path. The native replay reopens the actual source files
and runs the earlier-copy app comparator. It requires FFmpeg and SciPy
and uses no model weights. The payment smoke test uses the local worker
and its configured assets.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-aligned-speech-app-transfer-2026-11-14.json
python3 -B -S -m scripts.audio_aligned_speech_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_aligned_speech_transfer run "$new_receipt" \
  --full "$full_sine_bank" --short "$short_sine_bank" --sounds "$backgrounds"
backend/.venv/bin/python -B -m scripts.audio_aligned_speech_app "$local_app_output" \
  --bank "$full_sine_bank" --sounds "$backgrounds"
backend/.venv/bin/python -B -m scripts.compare_audio_references \
  earlier.wav submitted.wav comparison.json --verify
```

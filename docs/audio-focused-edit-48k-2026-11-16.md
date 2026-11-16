# Focused speech edit review for 48 kHz WAV and FLAC

The focused review can now inspect six-second, 16-bit mono WAV and
FLAC uploads recorded or exported at 48 kHz. It uses the app's
verified lossless decoding path and the existing six-second review
after the two-second check completes without a flag. The intervals
show both original 48 kHz sample positions and 16 kHz analysis sample
positions. The waveform and playable excerpt use the decoded audio.
The PDF and JSON preserve the same positions. The focused check
carries zero weight in the overall assessment.

## Frozen replay

Use all 136 deliveries from the matched voice-conversion test. These
include 48 staged synthetic edits, 72 ordinary recordings and 16 exact
isolated backgrounds. One rate-converted 48 kHz waveform per delivery
was saved as both WAV and FLAC. The test reads both original 48 kHz
sample streams and verifies their sample identity. It then decodes
each file through the app's regular 16 kHz path and runs the same
focused model and strict operating point.
[Source roles, recording terms and prior calibration](audio-joint-generator-transfer-2026-11-16.md)

All three locations from native 16 kHz remained at the same times on
both 48 kHz variants. The rate-converted files also produced one useful
location on a TTS edit under the −5 dB mixture. Each version flagged zero
of 72 ordinary submissions and zero of 16 isolated backgrounds. The
app's 16 kHz analysis samples differed between WAV and FLAC on five
deliveries despite identical 48 kHz source samples. Each container was
scored using its own decoded samples. Both containers agreed on every
reported interval and flag. These are paired deliveries of the existing
test recordings. Other recording hardware and codecs need their own
measurements. [Paired numeric results](evaluation-panels/audio-focused-edit-48k-2026-11-16.json.gz)

## Local app verification

Two completed prepaid test uploads, one WAV and one FLAC, showed
**2.800–3.300 s**. The original interval is **[134400, 158400)**
at 48 kHz, matching **[44800, 52800)** at 16 kHz. The marked waveform,
half-second clip, API, JSON and signed report agree. Ordinary and
isolated-background controls stayed clear. A 48 kHz file with one
extra source sample skipped the focused check. Seven simulated USD
9.99 checkouts queued zero jobs before payment. The same louder file
had the same overall assessment with the feature on and off. The
existing 16 kHz tests also replayed successfully.
[Local 48 kHz replay](evaluation-panels/audio-focused-edit-48k-app-2026-11-16.json)

## Replay

Enable `VERITAS_AUDIO_FOCUSED_EDIT_LOSSLESS_48K_REVIEW=true` alongside
`VERITAS_AUDIO_FOCUSED_EDIT_REVIEW=true` and the existing
`VERITAS_AUDIO_SHORT_LOSSLESS_48K_REVIEW=true` development options.
The 48 kHz focused-review switch defaults off. Use the same prepared
audio and pinned assets as the [native six-second replay](audio-focused-edit-local-2026-11-16.md#replay).

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
panel=docs/evaluation-panels/audio-joint-generator-conversion-2026-11-16.json.gz
receipt=docs/evaluation-panels/audio-focused-edit-48k-2026-11-16.json.gz
python3 -B -S -m scripts.audio_focused_edit_carrier review "$receipt"
backend/.venv/bin/python -B -m scripts.audio_focused_edit_carrier score \
  "$panel" "$media" models/weights/audio-upgrade "$output"
backend/.venv/bin/python -B -m scripts.audio_focused_edit_app \
  "$panel" "$media" "$app_output" --lossless-48k --browser
```

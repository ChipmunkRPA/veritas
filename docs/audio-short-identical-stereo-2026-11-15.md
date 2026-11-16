# Short speech-window coverage for two identical lossless channels

Full audio analysis now runs the existing short speech-window check on
qualified uploads with two identical channels. The source must contain
one continuous six-second stream, encoded as 16-bit 16 kHz WAV or FLAC.
Every sample in both original channels must match, and those samples must
also match the analyzed mono input exactly. The app leaves the short check
unassessed when the original channels differ, even if their downmix would
produce the same mono samples. The findings apply to the complete flagged
window and carry zero weight in AI confidence.

The browser and Standard PDF show that the two recorded channels contain
the same samples. The JSON binds the equality check to the original
channel PCM digest. Signed source replay decodes both recorded channels,
recomputes their sample equality, and verifies the selected clip and
waveform against the upload. Historical packages retain their original
coverage and remain readable.

## Local checkout and export verification

Four local uploads use the same known positive example from the
[nearby-window study](audio-short-edit-input-views-2026-11-15.md#local-evidence).
The mono WAV and the two identical-channel deliveries flag **4.000–6.000 s**.
Their selected WAV and waveform hashes, source timestamps and nearby
readings match. The fourth delivery has an opposite one-level sample
offset in each channel. It decodes to the same mono PCM and skips the
short check because its recorded channels differ. The two stereo positives
have identical durations and channel digests even though their container
hashes differ. The evidence API, JSON and Standard PDF agree, and
all four signed packages verified against their original uploads.
The fixture simulated four USD 9.99 payments and queued no unpaid jobs.
[Local app receipt](evaluation-panels/audio-short-identical-stereo-app-2026-11-15.json)

## Replay

Use the reconstructed `native` input directory from the earlier
[nearby-window replay](audio-short-edit-input-views-2026-11-15.md#reproduce).
The fixture reads and verifies the original study's PCM hash, writes
four local delivery files and sends them through the existing checkout
and analysis worker. Source audio stays local.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=1
receipt=docs/evaluation-panels/audio-short-edit-input-views-2026-11-15.json.gz
backend/.venv/bin/python -B -m scripts.audio_short_identical_app \
  "$receipt" "$native_inputs" "$run/paid"
```

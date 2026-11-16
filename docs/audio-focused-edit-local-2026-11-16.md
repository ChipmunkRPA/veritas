# Focused speech edit review in the local app

The local Full analysis can look for shorter edits when the two-second
speech check completes without a flag. It reviews continuous six-second
mono WAV and FLAC recordings delivered as native 16 kHz, 16-bit samples.
The check measures windows of 0.5, 0.75 and 1 second at 0.1-second steps.
It considers each window where at least half the samples contain detected
speech and reports the highest qualifying response. The original
two-second check keeps its existing flags. The focused review contributes
zero weight to the overall assessment.

A flagged result in **Audio intervals** shows the exact source times,
the marked waveform and a playable clip cut from the uploaded samples.
It links to any overlapping background observations from the mixed
recording. The waveform and clip represent that same mixture. Ordinary
editing and overlapping voices can affect the response. The evidence
API and JSON include source samples, window coverage and both assets.
The Standard PDF uses the same source times and waveform. The signed
package can be verified against the original upload.

The later [48 kHz extension](audio-focused-edit-48k-2026-11-16.md)
adds original and decoded sample positions for verified lossless files.

## Evaluation and local verification

The existing check plus the strict focused review located 5 of 32 noisy
edits in the separate eight-reader test, up from 3. Both additional
deliveries contain the same voice-conversion edit at different noise
levels. Both systems flagged 15 of 48 ordinary noisy deliveries and
zero of 16 background-only deliveries. On the 559 retained deliveries,
the combination increased useful locations from 10 to 20 of 224 and
ordinary alerts from 78 to 88 of 275. Neither version alerted on any
of the 60 isolated backgrounds. The earlier severe SINE misses remain.
[Full comparison and source terms](audio-joint-generator-transfer-2026-11-16.md)

On the 136 new deliveries, the serving implementation selected every
same window and agreed on every flag with the earlier study. The
largest absolute raw-score difference was 0.0000201 across 126 unique
inputs. The closest margin to the fixed operating point was 0.127.

The local checkout replay exercised seven uploads at a simulated USD
9.99. Analysis queued only after simulated payment. The same known edit
flagged at **2.800–3.300 s** in the two mixtures delivered as WAV and FLAC.
A matched ordinary recording and isolated background stayed clear.
The earlier flag stayed available, and a recording with one extra sample
stayed outside this review's scope. The browser decoded both marked
waveforms and confirmed 0.5-second playback. The PDF waveforms matched
the exported pixels, and signed verification cut the same clip and
recomputed the marked waveform from each original file. The louder
file's overall assessment agreed across the on and off runs.
[Application replay receipt](evaluation-panels/audio-focused-edit-local-2026-11-16.json)

## Replay

This review defaults off. It runs in development with
`VERITAS_AUDIO_FOCUSED_EDIT_REVIEW=true`, the existing candidate voice
configuration and verified local copies of the encoder and fitted head.
Use the [encoder pins and terms](audio-wavlm-v13-protocol.md#source-and-hypothesis)
and reproduce the head from the [two-generator fit](audio-joint-generator-transfer-2026-11-16.md#replay-and-compare-another-detector).
The loader checks all file sizes and SHA-256 pins. The head SHA-256 is
`432bb64ef1f2e16a052d1d9ae36089bf4910ca35c2d6958ec92b83a38e285ce8`.
Use the native voice-conversion preparation's output for `$media`. Use a
new path for `$output`. The replay uses an isolated database and simulated
Stripe transport.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cp "$fit/both.safetensors" models/research/wavlm-v13/joint-speech-head.safetensors
backend/.venv/bin/python -B -m scripts.audio_focused_edit_app \
  docs/evaluation-panels/audio-joint-generator-conversion-2026-11-16.json.gz \
  "$media" "$output" --browser
```

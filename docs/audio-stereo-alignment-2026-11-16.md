# Finding repeated background sound after stereo alignment

The direct stereo difference can expose repeated sound beneath speech
when both channels record the foreground identically. The new fallback
also checks when one channel carries the shared sound a fraction of a
millisecond earlier or at a different level. It shows the two matched
ranges, the original stereo excerpts, and separate waveforms and
playback clips after alignment. The ranges describe similarities in
sound. Loops, music, alerts, and ordinary edits can produce them.
They carry zero weight in the AI estimate.
[Direct stereo comparison](audio-channel-difference-2026-11-16.md)

## How the comparison works

This check uses continuous 2.5–4 second, native 16 kHz, 16-bit stereo
audio. Native PCM soundtracks and WAV or lossless FLAC qualify. It
looks for a consistent delay and gain in short, active windows. The
search considers ±12 samples of offset and a gain ratio of 0.5–2.
At least four windows and two thirds of the usable windows must
support the selected offset. It uses a bounded frequency-band search
and checks the same ranges against the waveform. No model weights
are required. The direct stereo comparison takes priority when it
already finds a repeat.
[Time-delay estimation method](https://ieeexplore.ieee.org/document/1162830)

The two reported time ranges use the native left-channel clock.
The adjoining source samples included in each original stereo
excerpt let the signed verifier reproduce the derived playback.
The original-file verifier reruns the alignment and compares the
saved excerpts and waveforms. Matches can cover part of an edit.

## Local comparison

Four separate AMI meeting recordings each contributed eight
previously unused clips. Each repeated background segment was 240,
400, or 640 ms long. The same scene supplied each mixture level.
The panned foreground comes from a separate meeting with an integer
delay of 3–8 samples in either direction and gains from 0.78 to 1.20.
Controls include unchanged stereo, ordinary panned foreground,
replacement with different background sound, and a changing foreground
pan. Results describe staged repeats under recorded foreground
sound. A correct location requires intersection over union of at
least one half at both ranges.
[AMI recording channels](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml) ·
[AMI source terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Edit condition | Correct, existing → with alignment | Files with any flagged range, existing → with alignment |
| --- | ---: | ---: |
| Centered foreground 10 dB louder | 32 → 32 of 32 | 32 → 32 of 32 |
| Panned foreground 6 dB louder | 2 → 12 of 32 | 6 → 18 of 32 |
| Panned foreground 10 dB louder | 1 → 16 of 32 | 2 → 18 of 32 |
| Panned foreground 18 dB louder | 0 → 16 of 32 | 0 → 18 of 32 |
| Four control conditions | — | 0 → 0 of 128 |

The alignment could assess 69 of the 128 controls. It abstained on
all 32 untouched recordings and most changing-pan controls where the
foreground condition was unstable. Across the panned positives, it
added six, five, and five individual proposals outside the scoring
criterion at 6, 10, and 18 dB respectively. Those include partial
and misplaced matches. The results share four recording groups and
describe this test panel. Native 44.1/48 kHz tracks, lossy stereo
encodes, and changing direction need separate evaluation before use.
[Frozen source locations and per-file results](evaluation-panels/audio-stereo-alignment-2026-11-16.json.gz)

## Application and replay

Set `VERITAS_AUDIO_STEREO_ALIGNED_REPEAT_REVIEW=true` in a development
backend. A saved analysis retains the method version it used. Standalone
audio uses `/stereo_repetition/aligned_measurement` in JSON. Video uses
the same location under `/audio`.

Five simulated USD 9.99 purchases exercised the analysis worker. Zero
jobs queued before a paid event. The WAV example located both ranges
at 0.430–0.680 and 2.351–2.601 seconds. The video example offset that
soundtrack by 0.400 seconds and the evidence retained the resulting
video playback times. Chromium loaded both waveforms and all four
clips in each example. The PDF, JSON, and signed packages matched.
The ordinary mix produced no range, and the centered mix used the
earlier exact comparison. Running the same WAV with the new check on
and off left the overall assessment unchanged.
[Local application receipt](evaluation-panels/audio-stereo-alignment-app-2026-11-16.json)

Use FFmpeg and the backend Python environment. Download the eight
credited WAVs from the source URLs in the receipt into a private
`$run` directory. Freeze with the release code and then replay the
comparison against sizes and hashes for those source files. The
receipt contains sample coordinates, digests, and outcomes.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_channel_alignment_panel freeze "$run"
backend/.venv/bin/python -B -m scripts.audio_channel_alignment_panel replay "$run"
diff -u <(gzip -dc docs/evaluation-panels/audio-stereo-alignment-2026-11-16.json.gz) \
  "$run/alignment-results.json"
```

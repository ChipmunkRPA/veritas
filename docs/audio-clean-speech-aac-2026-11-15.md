# Compare noisy AAC speech against a clean earlier copy

Chipmunk Verify can locate a drop in sound shared between a submitted AAC
recording and an earlier clean copy of the same recording. Previously the
local speech check required both files to be lossless. It now accepts a
six-second 16 kHz mono AAC M4A submission paired with a continuous,
six-second 16 kHz mono PCM16 WAV or FLAC. Both files must start at zero
on the same sample clock.

The check verifies the AAC presentation duration and excludes trailing
codec samples from this measurement. It applies the existing 960 ms
matching-sound rule, including the frequency-balance guard, without
refitting its point. Matching sound on both sides supports the selected
range. The browser and comparison PDF show the interval, both waveforms,
the amount of matching sound, and the AAC samples excluded. The browser
and JSON provide both playable excerpts. The JSON records the exact ticks
and source associations and can be replayed against both files.

The result identifies a local difference from the earlier recording.
Ordinary cuts and local level changes can produce the same finding.
The check carries zero AI-confidence weight and requires the earlier
copy. A recent controlled study likewise found that genuine edits can
produce cues seen on synthetic splices.
[Matched editing interventions](https://arxiv.org/html/2610.04479v1#S3.SS2)

## Local evaluation

We encoded two previously retained, separately sourced groups at
32 and 64 kbps AAC-LC with FFmpeg 7.0.2. All submissions included
recorded competing speech 5 dB above the foreground. The earlier copies
were dry. One group has 24 prompts from three readers with backgrounds
from three ICSI meetings. The other has eight held MLS readers with
backgrounds from three AMI meetings. Four generator families supplied the
staged replacements. There are 240 source pairs, each measured as PCM and
at both AAC bitrates. Versions of the same source are paired measurements.
[ICSI and same-word source preparation](audio-short-edit-detector-training-2026-11-14.md)
· [AMI and stock-voice source preparation](audio-random-clock-stock-2026-11-13.md)

The table counts an edit as located when the interval covers half of the
known edit and reaches 0.30 temporal intersection over union. The
prior app left the AAC speech check unassessed. Its existing lossless
comparison is the PCM reference below.

| Held result | WAV PCM | AAC 32 kbps | AAC 64 kbps |
| --- | ---: | ---: | ---: |
| ICSI synthetic replacements located | 43/48 | 41/48 | 43/48 |
| AMI synthetic replacements located | 14/16 | 14/16 | 14/16 |
| Intentional genuine splices flagged | 35/40 | 35/40 | 35/40 |
| Unchanged or same-word resampling controls flagged | 0/80 | 0/80 | 0/80 |
| Isolated backgrounds and wrong references admitted | 0/56 | 0/56 | 0/56 |

The codec test reused the earlier source clips and created new AAC
deliveries. The strong genuine-splice response makes the finding useful for
comparing versions. It also shows why detecting a difference here cannot
establish AI origin. This test leaves longer clips, unequal clocks,
other sample rates, independently remixed backgrounds and single-upload
deepfake detection open.

[ICSI measurement receipt](evaluation-panels/audio-clean-speech-aac-icsi-2026-11-15.json)
· [AMI measurement receipt](evaluation-panels/audio-clean-speech-aac-ami-2026-11-15.json)

## Local app and replay

Enable `VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH_AAC=true` with
`VERITAS_AUDIO_REFERENCE_COMPARISON_ENABLED=true` and
`VERITAS_AUDIO_REFERENCE_CLEAN_SPEECH=true`. From a retained audio
analysis, choose **Compare with an earlier audio recording** and mark
that the earlier file is a clean copy of the same recording. The AAC
setting defaults to off.

The local app replay checks a 32 kbps staged edit, matched untouched
recording and unrelated earlier recording. Two simulated USD 9.99
checkouts started zero jobs before payment. The edit shows
**1.960–2.920 s**, with both exposed waveforms and source excerpts.
The comparison did not create another job or alter the saved assessment.
The unchanged pair produced no range. The unrelated copy remained
unassessed. The JSON and PDF agreed on the source time. The verifier
decoded both files independently and checked the exact excerpt samples
and the AAC duration.
[Local comparison receipt](evaluation-panels/audio-clean-speech-aac-app-2026-11-15.json)

Use the receipts and earlier source commitments for a standard-library
recount. Native replay needs the earlier ICSI materialized `sources.json`
and `pcm/` directory, and the AMI stock bank with
`placements-freeze.json` and `placements-native.npz`.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipts=docs/evaluation-panels
python3 -B -S -m scripts.audio_clean_speech_aac_transfer \
  review "$receipts/audio-clean-speech-aac-icsi-2026-11-15.json"
backend/.venv/bin/python -B -m scripts.audio_clean_speech_aac_transfer \
  ami "$run/ami.json" --sources "$stock"
backend/.venv/bin/python -B -m scripts.audio_clean_speech_aac_transfer \
  icsi "$run/icsi.json" --sources "$icsi"
backend/.venv/bin/python -B -m scripts.audio_clean_speech_aac_app \
  "$run/app" --sources "$stock" --receipt "$run/ami.json"
backend/.venv/bin/python -B -m scripts.verify_audio_clean_speech_export \
  "$earlier" "$submitted" "$export"
```

## 48 kHz mono and stereo AAC

The local comparison also supports a six-second 48 kHz mono or stereo
AAC-LC submission with a single audio track. Its earlier copy is a
six-second 16 kHz mono PCM16 WAV or FLAC. Both recordings need continuous
timing from sample zero. The check reads the original AAC presentation
length, converts the soundtrack to 16 kHz mono, then excludes trailing
decoded samples outside the six seconds of playback. The browser and PDF
identify the original mono or stereo format. JSON preserves the native
48 kHz presentation length and the 16 kHz measurement and excerpt clock.
[FFprobe stream metadata](https://ffmpeg.org/ffprobe.html)

The measurement, frequency-balance guard and decision point carry over
from the earlier comparison. We re-encoded the two retained noisy source
banks at 64 kbps mono and 96 kbps stereo. In the stereo delivery the left
channel is the original mixture. The right uses a scaled direct copy and a
quiet 1.5 ms delayed copy. The source banks include competing speakers 5 dB
above the foreground. These are controlled re-encodes of the same 240
pairs. Microphones that independently capture the surrounding speakers
were outside this test.

| Held result | PCM | 48 kHz mono | 48 kHz stereo |
| --- | ---: | ---: | ---: |
| ICSI synthetic replacements located | 43/48 | 42/48 | 43/48 |
| AMI synthetic replacements located | 14/16 | 14/16 | 14/16 |
| Genuine splices flagged | 35/40 | 35/40 | 35/40 |
| Unchanged and same-word resampling controls flagged | 0/80 | 0/80 | 0/80 |
| Isolated backgrounds and unrelated references admitted | 0/56 | 0/56 | 0/56 |

The comparison marks where matching sound becomes weaker. The response
to genuine splices illustrates why it carries zero weight in the AI
assessment and requires the earlier copy.
[ICSI carrier receipt](evaluation-panels/audio-clean-speech-aac48-icsi-2026-11-15.json)
· [AMI carrier receipt](evaluation-panels/audio-clean-speech-aac48-ami-2026-11-15.json)

Use `--aac48` on either native panel command above. The portable `review`
command accepts receipts from either sample rate. Use `--carrier
aac48_stereo` on the app command with the 48 kHz AMI receipt to replay the
paid comparison and report export.

Two simulated USD 9.99 checkouts verified that work queued after payment.
The stereo edit showed **1.960–2.920 s**, with analysis ticks
**[31,360, 46,720)** and both playable excerpts. The six-second source
ended at tick 288,000 on the native clock. The comparison excluded 256
trailing samples from its 16 kHz decode. The API, PDF and JSON agreed.
Independent decoding reproduced the two excerpt hashes and waveform
values. The unchanged pair produced no local difference, and the
unrelated earlier copy stayed unassessed.
[Paid stereo comparison receipt](evaluation-panels/audio-clean-speech-aac48-app-2026-11-15.json)

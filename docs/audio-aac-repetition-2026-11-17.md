# Repeated quieter sound in short stereo AAC

Chipmunk Verify can locate repeated quieter sound in eligible AAC audio and
video soundtracks. The ordinary result shows both intervals, each original
stereo excerpt, the audio after reducing the loud shared sound, and two
waveforms of the compared frequency bands. The PDF and JSON use the same
locations. The finding carries zero weight in the AI estimate. Ordinary
loops, alerts, and edits can also repeat sound.

## Method and tested input

The check estimates a stable delay and level difference on the native
48 kHz channels and reduces the sound shared by them. The search compares
four frequency bands between 160 and 3,000 Hz. It proposes interval offsets
from the band signal and from the same band after local level normalization.
A reported pair needs agreement in at least two active bands throughout a
qualifying stretch of 40 ms windows. Both occurrences must last at least
150 ms. The displayed waveforms combine the four compared bands, with
each plot scaled to its peak. The detection adds no pretrained weights.
Published work on audio copy-paste localization motivated testing lossy
recompression explicitly. [Audio Engineering Society paper](https://aes.org/publications/elibrary-page/?id=22627)
· [AAC encoder documentation](https://ffmpeg.org/ffmpeg-codecs.html#aac)

Recent component-level audio research evaluates speech and environment
separately. This implementation likewise records its acoustic finding
separately from the voice assessment. [MixedTruth study](https://arxiv.org/abs/2610.14141)

The check is enabled locally with `VERITAS_AUDIO_STEREO_AAC_REPEAT_REVIEW=true`
in development. It requires a single continuous 2.5–4 second 48 kHz stereo
AAC-LC track in an MP4-family container and agreement between the decoded
clocks. It compares samples within the presented track extent, including
nonzero starts in video. The caller skips additional stereo capture after
a verified, asset-bound AI declaration.

## Fresh local comparison

Two earlier backgrounds supplied development examples. The operating point
was frozen before measuring four separate kitchen, field, roadway and metro
recordings. The test mixes 32 fresh background excerpts with foreground
speech from two separate AMI meetings. A known 240, 400 or 640 ms segment
of the background is repeated. Each edit and control is encoded at both
96 and 160 kb/s by the same FFmpeg AAC encoder. Both reported intervals
must overlap at least half the union of their corresponding labeled
intervals to count as correct. Controls include the original environment,
ordinary panned speech, a different background excerpt, a change of voice
position, a repeated foreground segment with unchanged background, and
the same channel delivered twice. [DEMAND recordings and CC BY-SA 3.0 terms](https://zenodo.org/records/1227121)
· [AMI recordings and CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Correct background repeat locations | Serving mono | Lossless stereo method on AAC | New lower-band stereo check |
| --- | ---: | ---: | ---: |
| 96 kb/s, voice 10 dB louder | 0/32 | 0/32 | 8/32 |
| 96 kb/s, voice 18 dB louder | 0/32 | 0/32 | 2/32 |
| 160 kb/s, voice 10 dB louder | 0/32 | 10/32 | 19/32 |
| 160 kb/s, voice 18 dB louder | 0/32 | 3/32 | 5/32 |
| Controls with a reported pair | 30/384 | 0/384 | 0/384 |
| Controls assessed | 366/384 | 173/384 | 214/384 |

The 30 existing mono highlights are repeated foreground speech. The new
check adds zero control highlights and assesses 214 of the 384 control
deliveries. It flags ten edited deliveries without correctly locating the
known pair. At the louder speech level the new check locates 7 of 64 staged
repeats. The two encodes share 32 source excerpts and one encoder. These
counts measure repeated-sound localization under the tested conditions.
[Held measurements and exact ranges](evaluation-panels/audio-aac-repetition-held-2026-11-17.json.gz)
· [Development comparison](evaluation-panels/audio-aac-repetition-development-2026-11-17.json.gz)

## Application and independent replay

Six simulated USD 9.99 purchases exercised the normal worker. No job queued
before confirmation. The app located repeats in two M4A files and in one
MP4 whose audio starts after the picture. The MP4 marked **0.808–1.218 s**
and **2.6525–3.0625 s** on the source playback clock. The browser loaded
the paired stereo excerpts and waveforms. The PDF, JSON and signed package
commit to the same source times and waveform pixels. A matched ordinary
file returned no pair, a 44.1 kHz file was outside scope, and an on/off
replay agreed on the AI assessment. [Local application receipt](evaluation-panels/audio-aac-repetition-app-2026-11-17.json)

The exact source roster lists the publisher archives, selected channel
members and their SHA-256 digests. Collect channels 1 and 8 into each scene
folder using the member basenames. Place both `Array1-05.wav` files at the
same level with `recorded-sources.json`. Source audio and rendered fixtures
stay local. The public receipts contain derived measurements, source
coordinates and hashes. [Held source roster](evaluation-panels/audio-aac-repetition-sources-2026-11-17.json)
· [ES2004a microphone directory](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2004a/audio/)
· [TS3004a microphone directory](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/TS3004a/audio/)

The recount uses only the Python standard library. The audit verifies the
exact WAV and M4A bindings, checks every decoded AAC frame timestamp using
FFprobe, and recomputes the new method from a separate raw decode. This
also verifies the frozen measurements with the released code, which
accepts an incomplete final decimation step on a video soundtrack.
[Held replay receipt](evaluation-panels/audio-aac-repetition-held-bindings-2026-11-17.json)
· [Development replay receipt](evaluation-panels/audio-aac-repetition-development-bindings-2026-11-17.json)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.audio_aac_repetition_panel
panels=docs/evaluation-panels
receipt="$panels/audio-aac-repetition-held-2026-11-17.json.gz"
python3 -B -S -m "$module" review "$receipt"
cp "$panels/audio-aac-repetition-sources-2026-11-17.json" "$recordings/recorded-sources.json"
"$runner" -B -m "$module" prepare "$recordings" "$run/held-wav"
"$runner" -B -m "$module" measure "$run/held-wav" "$run/held-aac"
"$runner" -B -m "$module" audit "$receipt" "$run/held-wav" "$run/held-aac" "$run/verified.json"
npm --prefix frontend run build
"$runner" -B -m scripts.audio_native_alignment_app "$run/held-aac" "$run/app" --aac --browser
```

Rebuild the earlier development WAVs with the `development` split of the
lossless comparison, then use that output with the same AAC measure and
audit commands. Byte replay uses the FFmpeg version pinned in each
measurement receipt. [Lossless replay procedure](audio-native-alignment-2026-11-17.md#app-and-independent-replay)

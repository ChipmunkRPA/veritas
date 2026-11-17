# Temporary timing changes in short stereo audio

Chipmunk Verify can highlight a brief change in the delay between the left
and right channels of an audio upload or the stereo soundtrack of a video.
The ordinary result includes the exact flagged interval, a delay chart, a
waveform from the original left channel, and stereo excerpts from before,
during, and after the change. The PDF and JSON retain the same source
coordinates. A change of speaker, microphone, or stereo mix can produce
this pattern. The observation carries zero weight in the AI assessment.

## Signal and scope

The check measures normalized cross-correlation in 60 ms windows spaced
20 ms apart after bandpass filtering at 250–3,600 Hz. It requires a stable
delay on both sides of a short, distinct delay that returns near the earlier
value. Windows with too little common sound or several plausible delays
contribute no measurement. The released check uses fixed signal-processing
code and no new model weights. Research on speech novelty motivated trying
short spectral comparisons first. That pilot rarely added correct locations
once control flags were constrained. [Speech splice localization](https://www.sciencedirect.com/science/article/pii/S1077314226002559)
· [Microphone time-of-arrival comparison](https://www.sciencedirect.com/science/article/pii/S0263224125029112)

The development switch is `VERITAS_AUDIO_STEREO_TIMING_CHANGE_REVIEW=true`.
The check accepts a 2.5–4 second, continuous 48 kHz, two-channel track
in PCM16, FLAC16, or AAC-LC in an MP4-family container. It requires a
verified single-stream audio clock. The AAC path measures the presented
extent and excludes encoder padding after the end of the track. No-match
and unavailable are distinct. Mono, longer tracks, and other codecs are
outside this implementation's scope.

## Held comparison

The development comparison used 32 excerpts from four recorded scenes.
It located 46 of 64 staged changes and flagged none of the 158 controls it
assessed out of 192 control files. The held comparison uses four different
recorded scenes and speech from two other meetings. The two staged changes
per scene excerpt are a short delay shift in otherwise identical speech
and a speech replacement with a changed delay. A correct result needs at
least 0.40 interval overlap by temporal intersection over union and delay
estimates within three native samples of each of the three known offsets.
[DEMAND background recordings and CC BY-SA 3.0 terms](https://zenodo.org/records/1227121)
· [AMI speech recordings and CC BY 4.0 terms](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)

| Held outcome | PCM | AAC 96 kb/s | AAC 160 kb/s |
|---|---:|---:|---:|
| Correct location, delay changed in the same speech | 17/32 | 16/32 | 17/32 |
| Correct location, speech replaced and delay changed | 18/32 | 18/32 | 18/32 |
| Correct location, text-to-speech splice with staged delay change | 0/8 | 0/8 | 0/8 |
| Correct location, voice-conversion splice with staged delay change | 1/8 | 1/8 | 1/8 |
| Controls assessed | 173/232 | 173/232 | 173/232 |
| Added control flags among assessed controls | 0/173 | 0/173 | 0/173 |

The 936 held deliveries represent 312 inputs in three encodings, including
matched ordinary audio, same-delay speech replacements, gain changes,
gradual pans, whole-track delay, recorded environments, and unedited
conversations. The check could not assess any of the 16 unedited conversation
excerpts resampled from the two recorded microphone channels or the 32
background-only excerpts in each encoding. The synthetic speech comparison
uses two generation methods from one family and previously retained word
edits. Its three correct deliveries are the same converted clip in three
encodings. These results support the specific stereo timing evidence.
They leave real-world short voice-edit recall and false-alarm rates unresolved.
[Held measurements and source coordinates](evaluation-panels/audio-stereo-timing-change-held-2026-11-17.json.gz)
· [Development measurements](evaluation-panels/audio-stereo-timing-change-development-2026-11-17.json.gz)
· [PhonemeDF paired speech](https://zenodo.org/records/19522979)

## Local app and independent replay

Eight simulated USD 9.99 checkouts covered WAV, two AAC bitrates, an MP4 with
an offset soundtrack, an ordinary control, an unassessable recording, an
unsupported sample rate, and a run with the check disabled. The app queued
zero jobs before simulated payment. Chromium loaded both plots and played
all three excerpts for each of the four flagged uploads. The flagged range
in the MP4 was **1.188–1.448 s** on the video playback clock. The JSON and
PDF match the exact excerpt coordinates, and the signed packages verify
against the original audio. Switching the new check off produced the same
AI assessment on the identical upload. [Local application receipt](evaluation-panels/audio-stereo-timing-change-app-2026-11-17.json)

The separate readback verifies all 256 development and all 936 held input
files, the WAV parent hashes, and the exact sample and delay measurements.
For AAC it compares 109,824 decoded frame timestamps and uses a separate
raw decode to verify the presented samples. Public receipts contain
coordinates, measurements, and hashes. [Development source readback](evaluation-panels/audio-stereo-timing-change-development-bindings-2026-11-17.json)
· [Held source readback](evaluation-panels/audio-stereo-timing-change-held-bindings-2026-11-17.json)

For a fresh rebuild, extract the held roster's WAV members into their scene
folders under `held_rooms` and copy the roster to `recorded-sources.json`
in the same directory. Put the `Array1-05.wav` and `Array1-08.wav` files
from both AMI meeting directories together in `held_speakers`, with meeting
IDs prefixed as shown in the held receipt. Use the previously prepared
`media` directory as `generators` to reproduce the optional synthetic rows.
For the 256-file development replay use the four earlier AAC scenes and
their `Array1-05.wav` files, `development` in place of `confirmation`, and
omit `--generators`. Exact-media replay uses FFmpeg 7.0.2.
[Held source roster](evaluation-panels/audio-stereo-timing-change-sources-2026-11-17.json)
· [IS1006b files](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/IS1006b/audio/)
· [ES2011b files](https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2011b/audio/)
· [Earlier AAC source inputs](audio-aac-repetition-2026-11-17.md#application-and-independent-replay)
· [Paired speech input preparation](audio-joint-generator-transfer-2026-11-16.md#replay-and-compare-another-detector)

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
receipt="$panels/audio-stereo-timing-change-held-2026-11-17.json.gz"
python3 -B -S -m scripts.audio_stereo_timing_change_panel review "$receipt"
cp "$panels/audio-stereo-timing-change-sources-2026-11-17.json" "$held_rooms/recorded-sources.json"
"$runner" -B -m scripts.audio_stereo_timing_change_panel measure \
  "$held_rooms" "$held_speakers" confirmation "$run/held" --generators "$generators"
"$runner" -B -m scripts.verify_audio_stereo_timing_panel "$receipt" "$run/held" "$run/native-readback.json"
npm --prefix frontend run build
"$runner" -B -m scripts.audio_stereo_timing_change_app "$run/held" "$run/app" --browser
```

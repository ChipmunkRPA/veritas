# Exact silence in one stereo channel

New audio and video analyses now locate 100 ms to 2 s of exact zero samples
in one decoded channel when the other channel retains measurable sound.
The check uses the aligned 16 kHz stereo prefix already produced by the app.
It evaluates up to the first 30 seconds in complete one-second units and
skips runs whose flanks cross a source clock break. It requires 100 ms
of measurable sound before and after in the silenced channel and at
least −50 dBFS RMS in each 100 ms block evaluated in the opposite channel.
The three longest qualifying runs are shown in time order.
[FFmpeg's per-channel silence detection](https://www.ffmpeg.org/ffmpeg-filters.html#silencedetect)

The browser plays the left, right and analyzed mono tracks separately on
the annotated waveform's time scale. Each clip includes the 100 ms of
context on either side. The PDF shows the same waveform and source times.
The JSON export binds the sample boundaries, channel, clock, levels and
clips. A signed source replay recomputes the runs and checks the published
waveform and playback. A gate, ordinary edit or channel automation can
produce the same observation. The check carries zero AI-authorship weight.

A simulated prepaid USD 9.99 analysis passed through the local worker and
reported a left channel mute at samples [17701, 21541), or 1.106–1.346 s.
The evidence API, PDF, JSON export, clips and signed source replay matched.
A separate video integration test confirmed the same evidence structure
with different soundtrack and video start timestamps.

## Matched local test

We tested 24 three-second excerpts interleaved from paired synchronized
microphones across two AMI meetings. We kept the original and inserted
one brief left mute, one brief right mute, one right attenuation, and one
whole-track mute per excerpt. The right mute also went through AAC at
96 kbps. The test used the normal stereo decoding path. A hit required
the exact injected start, stop and channel. The source recordings are
from the AMI corpus, published under CC BY 4.0. Media stays local.
[Intake and source credits](audio-ami-native-stereo-v291-method.md)

| Delivery | Exact single-channel recoveries | Clips flagged by the new check |
| --- | ---: | ---: |
| 120 ms left mute in PCM | 9/24 | 9/24 |
| 240 ms right mute in PCM | 10/24 | 10/24 |
| 240 ms right mute after AAC | 0/24 | 5/24 |
| Unedited | — | 0/24 |
| Right channel attenuated | — | 0/24 |
| Both channels muted | — | 0/24 |

The existing whole-track silence and one-second stereo level checks
identified none of the one-channel edits. The whole-track silence check
recovered 12 of 24 exact whole-track mutes. AAC preserved five qualifying
zero runs with shifted endpoints, so the displayed times refer to decoded
samples. These counts measure one recording operation on two source
sessions. They are insufficient to estimate real-world error rates or
AI-generation accuracy. [Aggregate receipt](evaluation-panels/audio-stereo-channel-silence-2026-11-14.json)

To repeat the evaluation after preparing the linked intake, pass its
qualified WAV directory and private qualification receipt.

```sh
PYTHONPATH=backend backend/.venv/bin/python -B -m scripts.audio_channel_silence_transfer \
  "$qualified_wav_dir" "$qualification_receipt"
```

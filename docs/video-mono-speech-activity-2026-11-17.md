# Speech-activity gaps in mono video

The app's previous speech-activity trial qualified stereo video. This
comparison tests whether those frozen settings also find short intervals
when one audio channel contains both the voice and room sound. The aim is
to locate a possible timing or sound change and present original evidence.
The method carries zero AI-confidence weight.

## Fixed comparison

The previous stereo speech-activity thresholds apply unchanged to PCM and
AAC. The candidate compares local speech activity in the original and a
noise-reduced reading. It requires independently measured speech on both
sides and ongoing mouth movement with before, during and after frames.
Both readings must place the decline on the same sample range.
[Earlier speech-activity comparison and weight licenses](video-speech-activity-gaps-2026-11-17.md)

The reservation uses 12 neutral CREMA-D speaker groups and 12 DCASE room
recordings outside all three earlier stereo studies. The held speakers
appeared in older timing studies. Their exact earlier source clips were
excluded. The new test samples one channel from each independent room take
and mixes it with the recorded foreground to make a genuine mono WAV before
video encoding. The study preserves source picture pixels and verifies their
original 30 fps presentation in each delivered file.
[CREMA-D license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) ·
[DCASE stereo scene archive](https://archive.org/details/dcase2013_scene_classification)

At each of two measured foreground-to-room levels (+8 and −5 dB), the
comparison uses two staged speech removals and five ordinary controls.
The ordinary controls are unchanged mixing, whole gain, a local gain dip,
local equalization, and a reduction in room sound. Clean originals and
clean staged mutes measure compatibility with the existing near-silence
review. Tests use both 16 kHz PCM and 48 kHz AAC-LC. A correct location
requires at least 0.4 temporal intersection over union on the presented
16 kHz sample clock. Both encodings share the underlying cases.

Local runtime admission requires at least four additional correct
loud-background locations across two or more speaker groups per encoding.
Up to four ordinary highlights per encoding at either background level,
across at most three speaker groups, qualify for review. The combined
policy gives the existing near-silence finding precedence and must preserve
the clean-edit hits. A successful comparison still requires matching source
timestamps and media in the browser, PDF and JSON during a local replay.

## Held results

The existing near-silence review located no removals during continuing room
sound. It located 11 of 12 clean mutes in each encoding. Giving that review
priority kept all of those locations. The following table shows the combined
policy on the same submitted files.

| Sound condition | Encoding | Correct locations | Misplaced edit ranges | Ordinary files flagged |
| --- | --- | ---: | ---: | ---: |
| Voice 5 dB below room | PCM | 2/24 | 0/24 | 0/60 |
| Voice 5 dB below room | AAC | 2/24 | 0/24 | 0/60 |
| Voice 8 dB above room | PCM | 11/24 | 1/24 | 1/60 |
| Voice 8 dB above room | AAC | 11/24 | 1/24 | 0/60 |
| Clean | PCM | 11/12 | 1/12 | 0/12 |
| Clean | AAC | 11/12 | 1/12 | 0/12 |

The two loud-background locations came from the hard and faded versions of
one speaker's recording. The check assessed 12 of 24 edited clips in PCM
and 10 of 24 in AAC at that level. The quieter level added locations across
six speakers in both encodings. The loud-background result fell short of
the preset minimum in both encodings. Mono extension stays in evaluation.

An independent comparison reproduced every reported count and verified all
384 file hashes and 12 parent hashes. FFmpeg and FFprobe separately verified
27,456 delivered frame timestamps, 858 conformed source frame hashes, and
14,643,200 mono samples against the receipts. These files use staged mutes
and separate room recordings. The results do not estimate performance on
natural edits or competing offscreen speakers.

## Replay

The [reservation](evaluation-panels/video-mono-speech-activity-plan-2026-11-17.json.gz),
[publisher inventory](evaluation-panels/video-mono-speech-activity-sources-2026-11-17.json.gz)
and [held measurements](evaluation-panels/video-mono-speech-activity-held-2026-11-17.json.gz)
retain the delivery hashes, native frames, measured conditions, all estimated
intervals, and audio positions. The clean scope is recorded as `100` in the
condition field. Media stays local.

```sh
export PYTHONPATH=backend
panels=docs/evaluation-panels
python3 -B -S -m scripts.video_mono_activity_panel review \
  "$panels/video-mono-speech-activity-held-2026-11-17.json.gz" \
  "$panels/video-speech-activity-gap-development-2026-11-17.json.gz" \
  "$panels/video-mono-speech-activity-plan-2026-11-17.json.gz"
```

Rebuild the source reservation with `scripts.video_mono_activity_sources`
and obtain the selected public corpus copies with
`scripts.video_covariance_sources download`. Use the archive and source
release documented for the [previous series](video-stereo-covariance-2026-11-17.md#reproduce).
The `measure` subcommand uses those sources, the pinned local model and
picture assets, and the previous stereo freeze. Its `--help` lists inputs.

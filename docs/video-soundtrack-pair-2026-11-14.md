# Local timing differences between two soundtracks in one video

Chipmunk Verify can compare the two audio streams in a short video without
an earlier upload. It looks for a distinctive sound that occurs at
different times in the two tracks while the surrounding sound matches
at the same times. A finding shows each track's stream index, source
clock, waveform and playable excerpt. The Standard PDF shows the two
waveforms and their source positions. JSON and the signed evidence export
carry the same positions and link both WAV clips. This comparison carries
zero weight in the overall AI estimate. The first track still supplies
the voice and mouth-timing checks.

Both tracks need a continuous matching decoded clock and a duration of
1.5 to 12 seconds. There must be exactly two audio streams. The check
applies the existing short-window waveform comparisons at 80 ms every
20 ms. A local shift needs four matching windows and two same-time
matches on each side. A second rule requires same-time matches in all
three parts of the clip and in at least 55% of matched windows. This
makes the app withhold a localization when most of the matched audio
has a common whole-track offset. The response describes a relative
timing difference without identifying an original track, an AI edit,
or a mouth mismatch. Single-track lip-sync localization remains open.
[Existing short-window scorer](video-short-soundtrack-2026-11-14.md)
· [Why articulation and timing need separate checks](https://arxiv.org/abs/2610.09223)

## Local test

Five previously evaluated Wikitongues recordings each supplied two new
four-second excerpts. One excerpt decoded with a trailing sample, which
was trimmed to the authored four-second extent. Each excerpt provided
clean shifts in both directions, an earlier shift with rain and a later
shift with train sound. Background
audio was added at 0.45 times the original speech RMS, about 7 dB below it
across the full clip. The relative level can be higher inside the edit.
Each excerpt also had matched controls, including gain, filtering, whole-track
delay, different speech, background-only, unequal audio clocks and
matching local delays in both tracks. The rendered PCM and AAC deliveries
reuse the same sources and edited ranges. Useful means at least 0.50
temporal IoU and the intended signed lag within 1 ms. The check used no
new model weights. [Sources and terms](video-short-soundtrack-2026-11-14.md#held-comparison)
· [Recorded backgrounds and terms](audio-focused-window-transfer-2026-11-12.md#frozen-comparison)

| Outcome | PCM in MOV | AAC in MP4 |
| --- | ---: | ---: |
| Clean edits localized | 18/20 | 18/20 |
| Edits with rain localized | 5/10 | 5/10 |
| Edits with train sound localized | 4/10 | 4/10 |
| Controls flagged | 0/120 | 0/120 |
| Controls outside the stream or clock scope | 40/120 | 40/120 |
| Controls with insufficient same-time support | 40/120 | 40/120 |

The shared-sound rule kept all 27 useful ranges and explicitly withheld
assessment on ten whole-track delays per format. The waveform bracket
rule alone already prevented those examples from reporting a location.
The panel does not measure performance on naturally edited dual-track
videos or on an AI-generated mouth or voice with a matching soundtrack.
[Portable per-delivery results](evaluation-panels/video-soundtrack-pair-2026-11-14.json)

Two simulated prepaid analyses exercised the full local worker. Neither
started an analysis job before payment. The positive case matched stream 1
at **1.100–1.400 s** with stream 2 at **1.300–1.600 s**. Both waveforms and
both WAVs matched the source replay and signed evidence export. The
different-speech control checked 198 windows and found no matching audio.
Its report showed insufficient shared sound and no highlighted interval.
[Local app verification](evaluation-panels/video-soundtrack-pair-app-2026-11-14.json)

## Run and reproduce

Set `VERITAS_VIDEO_SOUNDTRACK_PAIR_REVIEW=true` on a development instance
and start a new analysis of a supported two-track video. Open **Voice
evidence from this video** and find **Timing between audio tracks**.
The setting defaults to off. Test mode runs the same prepaid path as a
normal local analysis using simulated payment confirmation.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
receipt=docs/evaluation-panels/video-soundtrack-pair-2026-11-14.json
python3 -B -S -m scripts.video_soundtrack_pair_transfer review "$receipt"
backend/.venv/bin/python -B -m scripts.video_soundtrack_pair_transfer run "$run" \
  --bank "$published_recordings" --sounds "$pinned_backgrounds" --examples
backend/.venv/bin/python -B -m scripts.video_soundtrack_pair_app "$run" "$app_run"
```

The recording directory uses the five exact filenames and SHA-256 hashes
in the evaluation's `freeze.sources`. The background directory needs the
pinned `rain.ogg` and `train.ogg`. FFmpeg must reproduce the same muxed bytes
for byte-for-byte receipt comparison. The public evidence verifier reads
the signed package with standard-library geometry and asset checks. Add
`--source-video` and `--source-audio`, both pointing to the same original
video, to recompute the reported sound intervals, waveforms and WAVs.

# Repeated background sound under centered speech

Chipmunk Verify can check for sound that repeats beneath an identical foreground in
both stereo channels. It subtracts the right channel from the left and checks for
at least 150 ms of identical active samples in that derived signal. The result shows
both source ranges, a common waveform, and the original stereo excerpts. The browser
plays the excerpts and the channel difference separately. The Standard PDF and JSON
use the same ranges and artifact identifiers. Loops, alerts, edits, and off-center
speech can produce a match. This observation carries zero weight in the AI assessment.

## Local comparison

Four previously used AMI meeting recordings supply eight new timed scenes each.
The synchronized microphone pair provides the background track. A separate
excerpt from a different meeting supplies the added foreground. Each scene repeats
240, 400, or 640 ms of the two-channel background at a different position before
mixing in the foreground. In the centered cases, the same integer foreground
samples enter both channels. In the offset case, one foreground channel is delayed
four samples and reduced in level. The comparison uses the existing exact and
gain-adjusted separate-channel checks and the exact, two-band, and fallback
waveform checks on the mono input. The four endpoint errors must each be at most
four samples for a match to count. The method uses no added weights.
[AMI Corpus license](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) ·
[Synchronized recording channels](https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml)

| Condition | Deliveries | Correct ranges, existing | Correct ranges, channel difference | Difference alerts on controls | Difference unassessed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centered foreground, 10 dB above the underlying track | 32 | 0 | 27 | — | 0 |
| Centered foreground, 18 dB above the underlying track | 32 | 0 | 27 | — | 0 |
| Offset foreground, 10 dB above the underlying track | 32 | 0 | 0 | — | 0 |
| Untouched microphone pair, ordinary mix, or different-scene replacement | 96 | — | — | 0 | 0 |
| Same mono signal in both channels | 32 | — | — | 0 | 32 |

The same 27 scenes were detected at both centered levels. Across the 54 successful
deliveries, the maximum endpoint error was one sample. The repeated placements and
controls share source scenes, so the numbers describe this controlled comparison.
The matched passages contain recorded sound. This tests acoustic repetition and
does not estimate whether AI generated a voice or background. Small channel delays
and level differences defeated the check in this comparison.
[Per-delivery sample commitments and counts](evaluation-panels/audio-channel-difference-2026-11-16.json.gz)

## Use and reproduce

Set `VERITAS_AUDIO_STEREO_DIFFERENCE_REPEAT_REVIEW=true` in a development backend.
New analyses of continuous, native 16 kHz, 16-bit stereo WAV and FLAC between
2.5 and 4 seconds are eligible. An eligible PCM soundtrack in a video uses
the same check. The JSON records the projection as
`floor((left_s16 - right_s16) / 2)`. The signed package verifies that both
included recorded excerpts independently reproduce the same derived samples.
Supply the original file with `--source-audio` to also recompute the search
and check the reported ranges against the recording.

The local app replay exercised five analyses with simulated payment. No
analysis or job was created before payment. Fixed WAV and offset video
examples produced the same two anchors in the browser, PDF, JSON, and
signed evidence. Chromium loaded both original and both derived
excerpts, with no page errors. The ordinary control and the
off-center foreground case each showed no repeated range.

Use a clean checkout, the backend Python environment, and FFmpeg. Download the
eight first-party WAVs listed in the [publisher-file receipt](evaluation-panels/audio-gain-public-replay-2026-11-13.json)
to a private scratch directory using their listed filenames and set `review_dir`
to that directory. The replay verifies
their sizes and hashes. It checks 224 deliveries with a separate 32-bit
projection, compares the waveform and range candidates, and checks the
production decode on 28 deliveries. The published receipt includes hashes,
sample positions and outcomes. Source recordings and excerpts stay local.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.audio_channel_difference_panel freeze "$review_dir"
backend/.venv/bin/python -B -m scripts.audio_channel_difference_panel replay "$review_dir"
diff -u <(gzip -dc docs/evaluation-panels/audio-channel-difference-2026-11-16.json.gz) \
  "$review_dir/difference-results.json"
```

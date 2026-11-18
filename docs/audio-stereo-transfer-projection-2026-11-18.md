# Stereo projection for short edits in overlapping speech

Estimating two speech directions from a submitted stereo recording
improved the foreground-to-interference ratio on a controlled local
panel. The existing short voice check located zero edits on the
projected tracks and zero on the original mono mix. The projection
remains in evaluation.

## Fixed comparison

Both pilots use previously qualified six-second, lossless stereo renders
with the same measured BBC room response. Each four-scene panel has two
PhonemeDF and two stock-voice scenes, covering four synthesis families.
Each scene supplies a two-edit synthetic delivery, an untouched delivery
and a matched delivery with two genuine edits. The pilots use different
exact PCM deliveries. The PhonemeDF speakers recur, the stock reader
pairs are disjoint, and the same room recurs.
[Source preparation and terms](audio-two-voice-components-2026-11-15.md#sources-and-exact-controls)

The method estimates a left-to-right transfer from frequency bins
whose observed phase aligns with a candidate stereo delay. It then uses
a bounded linear projection to preserve the estimated target direction
while attenuating the alternative direction. Each projection stays on
the original sample clock. Both estimates are scored without selecting a
speaker. Known source tracks are used after scoring to measure the
change in foreground and interference levels. No new weights or author
code were used. Research on direction-informed beamforming describes
why errors in these estimates can suppress some of the desired speech.
[Primary beamforming study](https://link.springer.com/article/10.1186/s13636-024-00373-3)
· [Previous spatial-mask comparison](audio-spatial-mask-voice-transfer-2026-11-13.md)

The two window pairs start at 1.500/2.000 s and 3.500/4.000 s, with
each window lasting 750 ms. Each pair needs both windows to have at
least half their samples admitted as speech and exceed the same
previously frozen operating point. Neither point was fitted to these
stereo scenes. A proposal covers at least half the label duration and
at least 0.30 of temporal intersection over union to count. Seven of
eight labels in each pilot fit the fixed geometry individually, with
at most six matches when each proposal can be used once.
[Existing short-window operating points](audio-isolation-by-codec-2026-11-18.md#matched-delivery-comparison)

## Results

| Pilot | Files with a projection out of 12 | Model calls | Edit locations, mono / projected / either | Genuine alerts per view, untouched / matched edits |
| --- | ---: | ---: | ---: | ---: |
| Require exactly two observed delays | 0 | 46 | 0 / unavailable / 0 | 0 / 0 |
| Use the two strongest eligible delays | 12 | 144 | 0 / 0 / 0 | 0 / 0 |

Counts are identical at both operating points. In the first pilot all
12 files produced three eligible delay peaks even though the scenes
were constructed from two voice tracks. The exactly-two rule therefore
made no projected calls. Before inference on different exact scenes,
the second pilot froze the use of the two strongest eligible peaks.
It scored both projected tracks for every file.

In the second pilot, selecting the track with the best ratio for each
known edit after scoring gave a median improvement of **3.80 dB** over
mono across eight intervals. Median foreground and interference level
changes were −0.54 dB and −3.94 dB, respectively. That selection uses
benchmark reference tracks. All location results were still zero with
both projections available. These controlled stationary renders and
transferred points leave performance on independent rooms or natural
uploads unmeasured.

## Offline recount

The portable receipt holds anonymous sample ranges, window decision
bits and hashes of the native run files. Its acoustic values are numeric
study summaries. It contains no audio samples, waveform features,
speaker identifiers or local paths. The reviewer checks timing,
admission counts and fixed-point decisions. The native inputs and
measurements are required to reproduce the acoustic estimates.
[Numeric receipt](evaluation-panels/audio-stereo-transfer-projection-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_audio_stereo_transfer_projection \
  docs/evaluation-panels/audio-stereo-transfer-projection-2026-11-18.json.gz
```

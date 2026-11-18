# Stereo speech dropouts in loud room recordings

A check that measures the changing relationship between stereo channels
located none of 48 staged speech removals in fresh recordings. The existing
stereo candidate and the near-silence check also located none on this set.
The covariance method stays in the evaluation tools. It makes no contribution
to the app's findings or confidence.

## Held results

Twelve speakers and twelve room recordings were held out of both earlier
stereo comparisons. Each speaker contributed a hard mute, a mute with 25 ms
fades, and five ordinary mixes at each background level and encoding. The
ordinary controls included unchanged audio, whole-clip gain, a local gain dip,
a panning change, and a reduction in room sound. The numbers below are per
encoding. PCM and 48 kHz AAC gave the same counts on these observations and
share speakers, underlying audio and edit times. Correct ranges required
intersection over union of at least 0.4 on the presented sample clock.

| Voice relative to room sound | Check | Correct ranges | Wrong range on edited clip | Ordinary controls flagged |
| --- | --- | ---: | ---: | ---: |
| 5 dB below | Covariance | 0/24 | 1/24 | 7/60 |
| 5 dB below | Previous stereo candidate | 0/24 | 0/24 | 3/60 |
| 5 dB below | Near-silence check | 0/24 | 0/24 | 0/60 |
| 8 dB above | Covariance | 0/24 | 0/24 | 1/60 |
| 8 dB above | Previous stereo candidate | 0/24 | 0/24 | 4/60 |
| 8 dB above | Near-silence check | 0/24 | 0/24 | 0/60 |

At the louder level the seven covariance alerts came from five speakers.
The mouth-supported covariance check could evaluate 22 of 24 edited clips
at that level and 16 of 24 at the quieter level. Both covariance variants
returned zero correct ranges at the selected two-group development budget.
The phase-aware variant flagged nine ordinary loud-background controls per
encoding at that budget. Requiring movement before, during and after the
proposed interval gave the same counts as the audio-only version there.

## What was measured

The check estimates a two-channel covariance matrix in four frequency
bands. It learns which direction changes most between a candidate interval
and the recordings on either side. It measures that loss against the actual
signal power and requires the other measured direction to stay stable.
One variant keeps only in-phase relationships. The other also uses imaginary
cross power, which retains phase information. A controlled test confirmed
that the second variant can expose a pure phase difference and that diagonal
loading does not turn an identical gain change in both channels into
directional loss.

The trial uses no new trained weights. Its covariance method was informed by
[multichannel speech beamforming research](https://www.sciencedirect.com/science/article/abs/pii/S0885230816300730)
and [direction-preserving MIMO speech enhancement](https://arxiv.org/abs/2604.11179).
This comparison estimates local loss directly from the submitted channels and
uses the previously pinned mouth landmarks for the picture-supported arm.

Operating points were selected using the 280 deliveries from the original
ten-speaker development bank, then frozen before the 336 held deliveries were
scored. The choice was made separately for PCM (3.904 dB) and AAC (3.780 dB).
The in-phase arm won development selection for both encodings. It located one
loud-background edit and five quieter edits per encoding in development.
The previous stereo check found three and six respectively on the same bank.
The held clips use different exact source files from earlier timing trials,
although all twelve held speakers appeared in those earlier trials.

These are controlled removals of acted speech mixed with separately recorded
room sound. The comparison does not estimate performance on naturally edited
video, synthetic speech or other recording equipment.
[CREMA-D source and license](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt) ·
[DCASE stereo source archive](https://archive.org/details/dcase2013_scene_classification)

## Reproduce

The [frozen development receipt](evaluation-panels/video-stereo-covariance-development-2026-11-17.json.gz)
and [held receipt](evaluation-panels/video-stereo-covariance-held-2026-11-17.json.gz)
retain the ranges, settings, per-band measurements, and exact file hashes.
The [reservation](evaluation-panels/video-stereo-covariance-plan-2026-11-17.json.gz),
[publisher inventory](evaluation-panels/video-stereo-covariance-sources-2026-11-17.json.gz)
and [held input receipt](evaluation-panels/video-stereo-covariance-inputs-2026-11-17.json.gz)
record source exclusion, room selection, ground truth and the verified native
picture and sample clocks. Media and complete mouth coordinate streams stay
local. Recount the comparison from `backend/` with standard-library Python.

```sh
python3 -B -S -m scripts.video_stereo_covariance_panel verify \
  ../docs/evaluation-panels/video-stereo-covariance-held-2026-11-17.json.gz \
  --freeze ../docs/evaluation-panels/video-stereo-covariance-development-2026-11-17.json.gz
```

To perform a measurement from the original files, run
`scripts.video_covariance_sources` to reserve, download and prepare the media.
It checks the pinned publisher indexes, LFS pointers, archive checksum,
picture timestamps, sound samples and mouth-model checksum. Run
`scripts.video_stereo_covariance_panel measure --help` for the scoring command.
Use the held input receipt and its regenerated media, with the frozen
development receipt supplied through `--freeze`.

A separate local recount used the integer intersection and union of the
retained sample ranges to reproduce all three methods' counts across the
616 deliveries. It also verified the held input, mouth observation, source
inventory and frozen reservation hashes.

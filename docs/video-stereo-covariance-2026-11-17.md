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

### Compare another detector on the same files

The [independent comparison receipt](evaluation-panels/video-stereo-covariance-independent-2026-11-17.json.gz)
contains the portable manifest, exact ground-truth intervals, relative file
index, and three sets of predictions. The exporter checked all 336 delivered
videos and their parent video hashes. The manifest classifies the task as audio
because the labels apply to the soundtrack of each video. Intervals are
half-open sample ranges on the presented 16 kHz clock. Each second corresponds
to 16,000 samples. A copy of the mono prediction can verify interval counts.
Its coverage is unknown when no proposal was retained.

From the repository root, unpack the receipt and recount the selected system
with standard-library Python. Replace the system identity and prediction rows
in a new response file to compare your detector. Keep the manifest hash,
case IDs and input hashes. Supply `regions: []` for an assessed clip with no
proposal, or a reason and `response: null` for an unassessed clip.

```sh
review=$(mktemp -d)
receipt=docs/evaluation-panels/video-stereo-covariance-independent-2026-11-17.json.gz
python3 -B -S - "$receipt" "$review" <<'PY'
import gzip
import json
import sys
from pathlib import Path

with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    receipt = json.load(stream)
directory = Path(sys.argv[2])
parts = {key: receipt[key] for key in ("manifest", "truth", "file_index")}
for key, body in (parts | receipt["readings"]).items():
    (directory / f"{key}.json").write_text(json.dumps(body), encoding="utf-8")
PY
python3 -B -IS backend/scripts/independent_review.py count-events \
  "$review/manifest.json" "$review/truth.json" "$review/selected_covariance.json" \
  "$review/counts.json" --match 0.4 0.4
```

Use `scripts.export_video_covariance_review --help` to rebuild the receipt
after reproducing the original media in sibling `held` and `sources`
directories. It rejects an input receipt that differs from the frozen held
run and checks the exact files before exporting any comparison.

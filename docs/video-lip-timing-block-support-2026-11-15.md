# Checking whether a lip-timing finding holds across time

Three versions of one genuine camera recording accounted for three of the
five ordinary alerts that the ranked timing fallback added on the first
29.97 fps panel. The other two alerts came from different actors. This
test asks whether looking for support in alternating time blocks, or
checking the direction of mouth movement, removes those alerts while
keeping useful timing locations. It reuses the installed acoustic and
landmark assets and their existing terms. [Earlier evaluation and asset terms](video-lip-aperture-scale-2026-11-15.md)
· [Time-varying alignment research](https://arxiv.org/abs/1808.06250)

## Comparison

Keep any existing sparse finding with complete before, during and after
pictures. Otherwise use the ranked check at its existing tighter point.
At that exact candidate interval and lag, compare two additional rules.
The first splits measurements into alternating 40 ms or 80 ms blocks
from the start of the decoded soundtrack. Each split needs an observation
on each flank and two inside, with positive signed support in each region.
Both splits share a video, soundtrack, and smoothing, so they can share
an error.

The direction check was specified after viewing the first two rosters and
before scoring the separate native recordings. It compares the amount
and direction of change in the mouth-opening ranks over 80 ms, using the
nearest measured native pictures at the two endpoints. It needs positive
support for the proposed signed lag in each region. All rules keep the
same interval, picture frames and score point when they accept a candidate.

## Native 29.97 fps results

The two earlier groups contain 45 different actors. The separate group
reuses 15 digest-verified recordings from an earlier lag study, with actor
codes absent from both of those groups. All 15 passed the native picture
and mono audio clock checks. They share one spoken sentence. Each recording
yields two known 200 ms local audio shifts, two uniform shifts, and three
ordinary versions. Useful locations require at least 0.40 temporal
intersection over union and a signed timing error of at most 80 ms.
These staged shifts test localization on native camera footage. They
cannot measure detection of reanimated mouths or well-synchronized edits.
[CREMA-D publisher terms](https://github.com/CheyneyComputerScience/CREMA-D#license)
· [Earlier source acquisition method](../backend/scripts/prepare_video_lipsync_local_delay_transfer.py)

Each cell lists useful local ranges, ordinary alerts and uniform-delay
alerts for that group. Denominators for the first, remaining and separate
groups are 48/72/48, 42/63/42 and 30/45/30, respectively.

| Rule | First 24 actors | Remaining 21 actors | Separate 15 actors |
| --- | ---: | ---: | ---: |
| Sparse only | 2 · 0 · 1 | 2 · 0 · 0 | 0 · 0 · 0 |
| Add ranked fallback | 5 · 5 · 2 | 4 · 0 · 1 | 0 · 0 · 0 |
| Require 40 ms block support | 4 · 5 · 1 | 3 · 0 · 0 | 0 · 0 · 0 |
| Require 80 ms block support | 4 · 5 · 1 | 3 · 0 · 0 | 0 · 0 · 0 |
| Require direction support | 3 · 4 · 1 | 3 · 0 · 0 | 0 · 0 · 0 |

Both block checks removed the two extra uniform-delay flags and missed
two useful fallback locations across the earlier groups. Neither removed
an ordinary flag. Direction support dropped one ordinary flag and both
extra uniform-delay flags, while losing three useful fallback locations.
It also removed both misplaced local findings from the first group. On
the separate native recordings, the sparse check misplaced one interval.
None of the fallbacks added a finding. The app keeps its existing narrower
frame-rate coverage for this check. [Public decisions](evaluation-panels/video-lip-timing-block-support-2026-11-15.json)

The local worker now checks the required checkpoint files before
launching the timing subprocess. A missing file, including a broken link
to a disconnected drive, reports the existing unavailable-assets status.
With the pinned checkpoint restored, four simulated USD 9.99 checkouts
replayed the staged edit and its controls. The existing timing finding
and signed evidence verified, and no unpaid job entered the queue.

## Replay

The public receipt preserves counts per role, observation counts, signed
support categories and useful-location decisions. It omits source
identifiers, media, timestamps and model readings. The portable reviewer
reconstructs all eligibility decisions and totals from those categories.
Python 3.12 and 3.14 produce identical counts. A second native run
reproduced all 15 alternate captures byte for byte from the
digest-verified sources and original clocks. The raw measurements stay
with the local dataset.

```sh
export PYTHONPATH=backend
module=scripts.video_lip_timing_block_support
receipt=docs/evaluation-panels/video-lip-timing-block-support-2026-11-15.json
python3 -B -S -m "$module" review "$receipt"
```

Use the [prior capture instructions](video-lip-aperture-scale-2026-11-15.md#replay)
to generate the first and remaining native measurements. The separate
source directory uses the recorded [intake script](../backend/scripts/prepare_video_lipsync_local_delay_transfer.py)
and the exact roster digest in the public receipt. Use the copied local
source directory rather than regenerating the roster against later history.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
backend/.venv/bin/python -B -m "$module" capture "$sources" "$audit" \
  "$checkpoint" "$landmarker" "$runtime" "$run/alternate.json.gz"
python3 -B -S -m "$module" compare "$audit" "$first" "$remaining" \
  "$run/alternate.json.gz" "$run/results.json"
cmp "$run/results.json" "$receipt"
```

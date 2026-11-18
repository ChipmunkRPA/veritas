# Eye-region observations across six-second videos

Six overlapping windows found 11 staged eye-region holds after the opening
window. They added one ordinary alert over the opening-only comparison.
All six ordinary alerts came from variants of one camera recording. The
trial exceeded its frozen ordinary-alert and whole-picture-control limits.
The local app keeps its previously confirmed two-second scope.
[Two-second confirmation](video-eye-30fps-confirmation-2026-11-18.md)

## Frozen comparison

We checked 16 new Wikitongues recordings against the earlier eye-study
intake. Each admitted recording has a distinct publisher file and subject
group. We verified the publisher revision, media bytes and SHA-1, and original
30-fps frame order and timestamps. Six recordings failed the fixed clock or
crop eligibility rules. Two downloads returned a rate limit before the
freeze. The eight admitted camera clips use one fixed crop per recording and
one delivery encode to produce six seconds of upright 480 × 360 H.264 4:2:0
MP4. The study retains any repeated camera pictures present at intake.
Source notices and raw media stay local. The admitted source licenses were
CC BY-SA 4.0 and CC BY 4.0.
[Wikitongues collection](https://commons.wikimedia.org/wiki/Category:Videos_donated_by_Wikitongues)
· [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
· [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Every group supplied three clips with an intentionally held eye region and
six ordinary controls. The authors held earlier pixels from the same camera
at `[17, 31)`, `[52, 68)` or `[146, 164)`. The middle hold crosses a two-second
window boundary. The ordinary files were an unaltered transport, a second
encode, a whole-picture hold, a same-camera jump, a cross-camera cut, and a
local blur. The source roster, edit positions, detector and decision limits
were frozen before scoring.

The opening-only reference ran the unchanged 60-frame detector at frame 0.
The overlapping schedule ran it at frames 0, 24, 48, 72, 96 and 120. Each
window kept the existing face geometry, shortlist, texture and nearby-motion
requirements. This schedule can contain every supported 10–22-frame hold
within `[5, 175]` together with five flanking pictures on either side. It can
still abstain on an unstable face within an individual window. Proposals
formed groups when they pointed to the same eye and shared at least 25%
intersection over union on both the native time range and the box. The
first window in each connected group supplied the exhibited range.

A useful location requires the correct eye, temporal intersection over union
of at least 0.50 and at least 75% of the proposed crop inside the staged
region. The trial requires at least eight useful later locations across four
subjects, no more than two ordinary files with any alert, zero whole-picture
control files with an alert, and exact replay of every first-seen context
frame. The source limits require at least eight distinct groups and at least
six assessed originals.

## Results

| Clip or control | Opening only | Overlapping windows |
| --- | ---: | ---: |
| Early holds with a useful location | 5 / 8 | 5 / 8 |
| Boundary-crossing holds with a useful location | 0 / 8 | 6 / 8 |
| Late holds with a useful location | 0 / 8 | 5 / 8 |
| Ordinary files with an alert | 5 / 48 | 6 / 48 |
| Source groups with an ordinary alert | 1 / 8 | 1 / 8 |
| Whole-picture control files with an alert | 1 / 8 | 1 / 8 |

All 72 delivered files had an assessed window. The detector measured 403 of
432 windows. Eleven later locations span seven camera groups. The 27 window
proposals reduce to 24 interval-and-box groups. An independent original-file
decode confirmed all 72 before, during and after anchor clocks and their
original RGB and crop hashes. The detector localized useful ranges with some
endpoint error. For example, a hold at `[146, 164)` produced `[145, 163)`
with original context frames 144, 154 and 163. No useful range matched both
staged endpoints exactly. On the local machine the six windows took a median
0.702 s and a maximum 0.927 s for scoring. The slowest measured decode,
scoring and evidence replay together took 1.345 s.

The unaltered transport and four ordinary edits from one group all returned
the same opening interval `[9, 20)`. The whole-picture hold in that group
was planted at `[104, 120)`, outside the flagged interval. The twice-encoded
ordinary file from that group returned `[132, 142)` and accounts for the one
additional alert over opening-only. That combination explains why counting
six affected files and counting one affected recording answer different
questions. The frozen decision uses the file count. The relative gain of 11
useful later locations for one additional alerted ordinary file makes the
approach worth checking on another source-disjoint camera panel.

The serving check currently skips all 72 six-second files. The opening-only
comparison is an experimental reference. These edits preserve earlier
pixels and support testing eye-region stability. They do not validate AI
authorship or coverage of neural eye redraws.

## Portable replay

The compressed receipt contains anonymous group numbers, roles, boxes, frame
ranges, window outcomes and first-seen transport clocks. The standard-library
reviewer reconstructs the counts and the frozen decision. Checking decoded
pixels independently requires the licensed recordings and local delivery
files. No images or participant details ship with the receipt.
[Numeric receipt](evaluation-panels/video-eye-long-windows-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_eye_long_windows \
  docs/evaluation-panels/video-eye-long-windows-2026-11-18.json.gz
```

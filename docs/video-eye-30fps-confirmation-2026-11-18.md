# Independent confirmation of eye-region holds at 30 fps

The frozen shortlist reproduced useful eye-region locations in a second
CREMA-D actor cohort and a separate Wikitongues corpus. Both panels met
their own predeclared checks. The result supports measuring a temporary
eye-region stability pattern in the tested delivery profile. The earlier
25 fps trial still records an alert on an ordinary picture jump.
[Earlier cadence comparison](video-eye-cadence-transfer-2026-11-18.md)

## Source and decision rules

The actor confirmation reserves 16 new CREMA-D actors whose identifiers
are disjoint from all 15 actors in the earlier intake. The independent
corpus supplies five separately documented Wikitongues speakers. All
source windows keep 60 actual camera pictures in order, with verified
original presentation timestamps. The delivered two-second clips use
upright 480 × 360 H.264 4:2:0 video at exactly 30 fps. The Wikitongues
inputs use verified, fixed crops from the originals, followed by one
delivery encode. The designated tandem roles add a second encode.
[CREMA-D media terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt)
· [Wikitongues collection](https://commons.wikimedia.org/wiki/Category:Videos_donated_by_Wikitongues)
· [License verified on the selected Wikitongues source files](https://creativecommons.org/licenses/by-sa/4.0/)

Each source contributes a short and a long hold made by preserving earlier
eye pixels, plus a tandem delivery of the long hold. The seven ordinary
controls include an original, an encoded copy, a whole-picture hold, a
same-camera jump, a cross-actor cut, local blur, and local gain. The actor
and corpus rosters, edit ranges, control roles and natural-motion witnesses
were frozen before scoring. The shortlist retains the eight proposals, exact
30 fps window and geometry rules, and the same texture and nearby-motion
points as the previous study. A useful range uses the correct eye, reaches
0.50 temporal intersection over union, and has at least 75% of its crop
inside the authored region.
[Guarded shortlist method](video-eye-shortlist-context-2026-11-18.md)

The actor confirmation requires eight useful locations across four actors,
at most two ordinary alerts, and zero whole-picture alerts. The smaller
corpus comparison requires three useful locations across two speakers and
zero whole-picture alerts. Its ordinary counts are descriptive.

## Results

| Measure | CREMA-D actor confirmation | Wikitongues corpus comparison |
| --- | ---: | ---: |
| Useful locations, raw → top-only → shortlist | 43 → 21 → 32 / 48 | 10 → 7 → 7 / 15 |
| Ordinary alerts, raw → top-only → shortlist | 11 → 1 → 1 / 112 | 8 → 0 → 2 / 35 |
| Whole-picture alerts, raw → top-only → shortlist | 2 → 0 → 0 / 48 | 2 → 0 → 0 / 15 |
| People with a useful shortlist location | 12 / 16 | 4 / 5 |
| Ordinary controls assessed | 105 / 112 | 31 / 35 |
| Cross-actor cuts assessed | 9 / 16 | 1 / 5 |

All 63 staged edits were assessed. Both guarded arms raised zero alerts on
the 22 assessed original and tandem files whose ordinary blinks, eyelid
movement or head movement were marked before scoring. The shortlist flagged
two twice-encoded originals and one local blur elsewhere in the controls.
It added 11 useful locations over the top-only arm in the actor cohort. In
the smaller corpus, it added two ordinary alerts and no useful location.
The shortlisted ranges carry limited temporal precision. None matched both
authored endpoints exactly.

One useful shortlist-only result spans native frames **[16, 32)**, or
**0.533–1.067 s**. The staged edit spans frames **[17, 30)**. The recorded
source context uses native frames 15, 24 and 32 for before, during and
after. The paired unaltered view and ordinary picture-jump control produced
no shortlist match. The native audit matched 126 exact source frame-and-crop
pairs and confirmed 575 changed parent pictures across 42 authored
regions. These staged changes use earlier pixels from the same recording.
Accuracy on smoothly redrawn eyes and AI authorship remain outside this
measurement.

## Local application

`VERITAS_VIDEO_EYE_LOCAL_HOLD_REVIEW=true` enables the observation in a
development instance. The app verifies 60 consecutive native timestamps on
an exact 30 fps clock and requires upright 480 × 360 H.264 4:2:0 MP4 video.
It distinguishes an input outside that scope, unavailable measurement, an
unstable face, and an assessed input without a match. The overall confidence
does not use this observation.

For a match, the evidence view and Standard PDF identify the source clock
range, measured crop and original frames from before, during and after the
hold. The JSON and signed package retain the same data. Original-file
verification checks all 60 decoded RGB hashes and timestamps, the six
exported images, and the recorded texture and motion measurements. Positive
measurement verification uses the native pixels and requires no face weights.
The source ranges identify unusual stability in the measured eye region.
Ordinary motion, compression and editing can produce similar measurements.

The application implementation reproduced all 210 predeclared decisions and
all 42 selected measurements across the two banks. Three of those matches
were ordinary controls. Five local app cases exercised simulated USD 9.99
checkout, with zero jobs before payment. Chromium displayed the three source
frames and matching crops in the staged hold. In that case, frame 15 served
as context before the measured range `[16, 32)`, and frame 32 served as the
first picture after it. All five signed exports verified against their
original files. The untouched clip and the ordinary jump produced no flag,
and the 25 fps control was outside scope. Turning the check on for the same
upload preserved its assessment and confidence inputs.
[Local worker, Chromium, and export receipt](evaluation-panels/video-eye-local-app-2026-11-18.json)

## Portable replay

The numeric receipt contains anonymous group numbers, exact clock offsets,
candidate ranges, scalar responses and frame and crop commitments. Source
recordings, picture crops, participant identifiers and original-file paths
stay local. The reader recounts the two frozen decisions using only the
standard library. Full native replay additionally needs the original
licensed recordings and source preparation.
[Numeric confirmation receipt](evaluation-panels/video-eye-30fps-confirmation-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_eye_30fps_confirmation \
  docs/evaluation-panels/video-eye-30fps-confirmation-2026-11-18.json.gz
```

# Source-bound face transitions in four-second 720p clips

The sustained face-transition comparison located all 12 staged swaps
at the exact frame across six source groups. The existing 720p
short-return comparison found none of those transitions. The new
reading marked one ordinary focus change among 30 controls. It
remains an optional local observation with zero AI-assessment weight.

## Method

The local scope is a four-second, upright 1280 × 720 H.264 AVC1
MP4 with exactly 120 frames at 30 fps. The original stream must start
at zero with exact 1,000-tick steps in the 1/30000 time base. The
measurement averages each 2 × 2 native RGB block with integer
half-up rounding. This yields a 640 × 360 input to the existing
sustained-transition measurement at its previously frozen point.
The short-return component keeps its existing schedule and role.
[Earlier 30 fps comparison and model terms](video-face-single-boundary-30fps-2026-11-18.md)

The marked timestamp refers to the exact first frame of the
observed transition. The three supporting frames and crops use the
original 720p pixels. Crop coordinates are twice the corresponding
coordinates in the measurement view. The JSON record commits both
resolutions and the original stream timestamps. A change in focus
can trigger a similar reading.

## Held comparison

The six held groups are separate from the prior sustained and
sparse transition groups by known clips and donors. Two groups
have native 720p source recordings and four use 1080p source
recordings rendered down once to 720p. The source timing stays at
30 fps. Each group has an edited prefix and suffix, and five
ordinary controls covering untouched footage, exposure, focus, a
same-recording jump and a cross-recording cut. The two supplied
renderers are balanced. Wider prior project exposure or shared
identities that lack source links cannot be ruled out.

| Held outcome | Existing short-return component | With sustained transition check |
| --- | ---: | ---: |
| Useful edited locations, within one frame | 0/12 | 12/12 |
| Exact frame locations | 0/12 | 12/12 |
| Exact locations from native 720p groups | 0/4 | 4/4 |
| Ordinary files highlighted | 0/30 | 1/30 |
| Cross-recording cuts highlighted | 0/6 | 0/6 |
| Wrong locations on edited files | 0/12 | 0/12 |

The new comparison left seven ordinary controls unassessed. Six
were cuts and one was a focus control. A separate three-recording
camera control replay found no transition. Those camera sources
had earlier exposure through a different resolution and frame-rate
test. The frozen held gate passed. The comparison covers the existing
720p short-return component, including its second return range, and
the new sustained check. The table does not measure the accuracy
of every Full video check or the origin of an entire recording.

## Original-file replay

An independent calculation verified 5,040 held native frames, 528
substituted source frames and all 39 original and measured frame
anchors for the reported transitions. The final backend reproduced
all 45 file outcomes, including the three camera controls. The
numeric receipt omits source and creator identities, source hashes
and media. The checker derives the frozen gate from the recorded
case outcomes using only the standard library.
[Anonymous evaluation receipt](evaluation-panels/video-face-boundary-720p-2026-11-18.json)

```sh
python3 -I -S backend/scripts/review_video_face_boundary_720p.py
```

## Local application verification

The same edited clip completed simulated USD 9.99 checkout and
signed source replay with the option disabled and enabled under
the same Full detector policy. The assessment agreed across the
two runs. A separate six-case checkout, with two unrelated slower
frame classifiers disabled for that replay, checked the browser,
PDF, JSON and signed files on the edited, ordinary, focus, hard-cut
and alternate-clock inputs. The browser loaded all 12 original
and crop images for the two marked clips with zero page errors.
No analysis started before payment. The local flag is
`VERITAS_VIDEO_FACE_SINGLE_BOUNDARY_NATIVE720_REVIEW`.

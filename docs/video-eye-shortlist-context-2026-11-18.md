# Nearby motion in the eye-region shortlist

Checking nearby motion on the existing shortlist found **10 of 27** staged
eye holds in a new holdout, compared with **9 of 27** when the check considered
only the first proposal. Both approaches raised **zero alerts among 68 ordinary
controls**. The extra location came from one capture group. The frozen
advancement rule required three additional locations across at least two
groups. The shortlist remains experimental.

## What changed

The existing eye texture experiment ranks areas where fine detail stays steady
while the rest of the face moves. It considers up to eight proposals, checks
the eye position at three source frames, and returns the first qualifying
proposal. The top-only collar comparison then tests whether the picture
changes more around that crop, both horizontally and vertically. The new
comparison applies that same nearby-motion test to each landmark-qualified
proposal in its original order and returns the first to pass. It keeps the
same texture point, nearby-motion point of 0.100, whole-picture continuity
checks, and eight-proposal limit. Research on dependencies between facial
regions and the effect of natural motion on residuals motivates this comparison.
It uses the project's existing face locator and pixel measurements.
[Regional motion research](https://arxiv.org/abs/2512.04175)
· [Multiscale residual research](https://arxiv.org/abs/2611.12508)
· [Original top-only comparison](video-eye-context-guard-2026-11-18.md)

The top-only comparison previously located 13 of 30 and 7 of 27 staged holds
on two separate capture rosters. Both rounds fell short of their original
15-location requirement. The raw eye texture check, top-only collar, and
shortlist all remain in evaluation. The application’s picture and mouth hold
checks located zero of the staged eye holds in this comparison.

## Held results

Each table entry gives useful locations followed by ordinary alerts. The
historical rows are exact replays of the two previously measured panels.
All earlier raw proposals and their adjacent-motion readings reproduced.
The fresh panel used ten previously unmeasured capture groups. Nine supplied
27 eye edits, and all ten supplied ordinary files. The 68 controls include
ten cross-capture hard cuts. Untouched and re-encoded originals from six
groups showed blinks, asymmetric closure, head movement or gaze changes in a
visual review recorded before scoring. All 12 of those files were assessed and
raised zero shortlist alerts.
[Pinned source terms and attribution](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

| Panel | Unguarded top proposal | Top-only collar | Guarded shortlist |
| --- | ---: | ---: | ---: |
| First historical | 17/30 · 8/60 | 13/30 · 0/60 | 15/30 · 1/60 |
| Second historical | 12/27 · 7/58 | 7/27 · 0/58 | 9/27 · 0/58 |
| Fresh holdout | 16/27 · 14/68 | 9/27 · 0/68 | 10/27 · 0/68 |

A useful result selects the correct eye, covers at least 0.50 temporal
intersection over union, and places at least 75% of its selected crop within
the authored patch. The new comparison kept all 29 useful top-only locations
across the three panels. In the two historical panels it added four useful
locations, one misplaced location on an edited file, and one alert on a
twice-encoded authentic clip. It raised zero whole-picture hold, jump or cut
alerts in all three panels. On fresh media it measured all 27 edits and 57
of 68 ordinary controls. The median added calculation took 0.110 seconds
after decoding. It did not recover both exact authored endpoints on any of
the accepted locations.

The fresh rule allowed at most two added ordinary alerts, zero added
whole-picture alerts, and one additional misplaced range. Those limits held.
The shortlist missed the useful-location requirement. The fresh roster and
delivery recipe were fixed before scoring. An independent replay
verified 216 physically changed pictures inside the 18 pre-encode edit
intervals. It also matched 114 before, during and after frame-and-crop
commitments across the three panels to decoded sources. Participant overlap
across the archive remains unknown. The results apply to staged eye holds in
short 480 × 360 H.264 files at 25 fps. Accuracy on generated or smoothly
redrawn eyes remains unmeasured. The experiment supplies no AI authorship score.

## Replay the counts

The receipt contains anonymous group numbers, numeric locations, source
timestamps, frame and crop hashes, and decision inputs. Source recordings,
image crops and publisher capture identifiers stay local. The reviewer
recounts location overlap and control alerts, checks witness clocks, and
reproduces both advancement decisions. The optional historical arguments tie
each original raw proposal and top-only decision to its published panel.
The command works with Python's standard library. Replaying native detection
and verifying image hashes also takes the original licensed recordings.
[Numeric receipt](evaluation-panels/video-eye-shortlist-context-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_eye_shortlist_context \
  docs/evaluation-panels/video-eye-shortlist-context-2026-11-18.json.gz \
  --historical docs/evaluation-panels/video-eye-context-held-2026-11-18.json.gz \
  docs/evaluation-panels/video-eye-context-confirmation-2026-11-18.json.gz
```

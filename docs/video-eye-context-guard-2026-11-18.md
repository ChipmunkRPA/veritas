# Distinguish a held eye patch from a naturally steady eye

The earlier normalized texture check found many staged eye holds.
It also flagged ordinary footage, where one eye region stayed
nearly unchanged while more distant face regions moved. This test
compares the eye patch with nearby picture details to see whether
the separation is more specific to a local edit. Research on facial
motion motivates comparing neighboring regions. The implementation
uses its own pixel arithmetic and the existing face locator.
[Earlier tests and false alerts](video-eye-texture-hold-2026-11-18.md) ·
[Research on motion dependencies between facial regions](https://arxiv.org/abs/2512.04175)

## Development replay

We treated all three prior held panels as development data for this
comparison. The replay used 270 deliveries from 30 groups, including
90 staged holds and 180 ordinary or other-edit controls. It verified
the delivery hash whenever a video was rerendered. Every arm starts
with the normalized texture check's highest-ranked proposal and can
withhold it based on the added measurements.

The first arm compares texture inside the proposed box against four
short bands outside it. Each band uses the median texture change from
one native picture to the next within the proposed range. The arm
requires a stronger change outside the box in both a horizontal and
a vertical direction. The second arm checks whether both ends of
the proposed region show stronger texture transitions than at the
other eye. A third arm requires both comparisons to agree. The scan
uses the same before, during, and after source frame positions as the
initial proposal.

| Development rule | Useful locations | Ordinary alerts | Alerts on whole-picture holds or jumps | Misplaced ranges on edited files |
| --- | ---: | ---: | ---: | ---: |
| Normalized texture alone | 64/90 | 35/180 | 8 | 7 |
| Selected nearby-band comparison | 49/90 | 2/180 | 0 | 2 |

The selection retained at least half of the useful development
locations and prioritized reducing ordinary and whole-picture alerts.
The retained rule uses the nearby bands only. The opposite-eye
comparison and the combined rule both lost too many useful
development locations. The selected value and admission logic were
fixed before reading the new held outcomes.

## First held result

Ten newly reserved publisher recordings supplied 30 staged holds and
60 controls. The source keys and byte hashes are disjoint from all
development rosters. The same 50 native 480 × 360 H.264 pictures at
25 fps, authored interval labels, and evaluation rules apply. Each
located crop and the three before, during, and after source pictures
were decoded again and hashed. The recordings come from RTFS-10K
under CC BY-SA 4.0. [Pinned RTFS-10K dataset and attribution](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)
[Frozen development measurements](evaluation-panels/video-eye-context-development-2026-11-18.json.gz)
· [Held source and frame commitments](evaluation-panels/video-eye-context-held-2026-11-18.json.gz)

| Held result | Existing native holds | Normalized texture alone | With nearby bands |
| --- | ---: | ---: | ---: |
| Holds located | 0/30 | 17/30 | 13/30 |
| Capture groups with a useful range | 0/10 | 7/10 | 6/10 |
| Ordinary flags | 0/60 | 8/60 | 0/60 |
| Whole-picture hold or jump flags | 0/20 | 4/20 | 0/20 |
| Misplaced ranges on an edited delivery | 0/30 | 1/30 | 0/30 |

The selected comparison removed all eight control alerts and the
misplaced range. It also withheld four useful locations. It located
11 of 20 holds after the first encode and two of ten after a second
lossy encode. The rule fell two edit locations short of the frozen
minimum of 15, while meeting the other gates. The recorded range
identifies where the pattern was measured. It can differ from the
underlying edit boundaries. The check assessed all 30 edits and
59 of the 60 controls.

The median nearby-band contrast was 0.275 on useful unfiltered
locations and 0.046 on ordinary alerts. Every ordinary alert measured
below the frozen 0.100 value. This indicates that the original
distant-face check could flag stable eye pixels even when nearby
pixels showed little change relative to the eye.

The receipt’s timing field came from the earlier eye measurement in
the paired replay. It gives no timing estimate for the nearby-band check.

## Confirmation with the same rule

Ten more source groups were reserved before scoring. Nine supplied
staged eye holds and one supplied ordinary controls only because its
eye geometry was unsuitable for authoring a hold. The same frozen
cutoff and gates applied to the 27 edited deliveries and 58 controls.
[Confirmation receipt](evaluation-panels/video-eye-context-confirmation-2026-11-18.json.gz)

| Confirmation result | Existing native holds | Normalized texture alone | With nearby bands |
| --- | ---: | ---: | ---: |
| Holds located | 0/27 | 12/27 | 7/27 |
| Capture groups with a useful range | 0/9 | 6/9 | 4/9 |
| Ordinary flags | 1/58 | 7/58 | 0/58 |
| Whole-picture hold or jump flags | 0/20 | 1/20 | 0/20 |
| Misplaced ranges on an edited delivery | 0/27 | 2/27 | 2/27 |

The single flag from the existing checks was a mouth-region observation
on a recompressed original. The nearby-band rule removed all seven
ordinary alerts from the normalized eye check. The strongest ordinary
contrast was 0.073, while the median on useful unfiltered locations was
0.123. The drop in that separation relative to the first held panel
left more edited ranges below the frozen value. The check assessed
25 of the 27 edits and 53 of the 58 controls.

Across the two held tests, the new rule brought ordinary
eye alerts down from **15/118 to 0/118** and useful locations from
**29/57 to 20/57**. It removed all five whole-picture hold or jump
alerts. Both tests fell below the frozen useful-location requirement.
The eye comparison remains evaluation-only.

## Reproduce

```sh
PYTHONPATH=backend python3 -B -S -m scripts.video_eye_context_guard_panel verify \
  docs/evaluation-panels/video-eye-context-held-2026-11-18.json.gz \
  docs/evaluation-panels/video-eye-context-development-2026-11-18.json.gz
PYTHONPATH=backend python3 -B -S -m scripts.video_eye_context_guard_panel verify \
  docs/evaluation-panels/video-eye-context-confirmation-2026-11-18.json.gz \
  docs/evaluation-panels/video-eye-context-development-2026-11-18.json.gz
```

These commands recompute the frozen development selection, held
decisions and counts. They check the format of recorded frame and
crop commitments and verify their timestamp arithmetic. Recomputing
hashes from decoded pictures and replaying the measurements also takes
the pinned source files and face locator from the earlier study.
Raw media and frame images stay local.

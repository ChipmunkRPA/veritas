# Distinguishing short local softening from face changes

A previous 29.97 fps trial found 17 of 18 staged short face swaps at their exact
frame boundaries. A separate ordinary-control panel then showed that the same
check could localize four frames of ordinary blur. The new offline comparison
asks whether two unedited source crops can jointly explain the middle crop
through local softening, while preserving evidence for face swaps that occur
during blur.

The method reads the original crops already attached to the detected interval.
It aligns each flank crop to the middle using a small integer image shift and
a bounded luminance correction. A fixed set of Gaussian scales then tests
whether each flank predicts the middle's fine detail better while preserving
its coarse structure. Both flanks must satisfy the same thresholds. No
detector was retrained. The rule acts after the existing timing check and
keeps the measured interval and picture bindings intact.

The reading was developed with two known ordinary blur alerts and 13 older
exact-30 swap findings. The prospective panel used other camera and swap
source groups, newly placed three- and four-frame interventions, two different
retained swap renderers, and both 30,000- and 90,000-tick delivery clocks.
Four RTFS groups contributed native exact 29.97 footage, and two other
genuine camera recordings supplied one-to-one re-clocked pictures. Some source
recordings appeared in earlier project studies. Training exposure and
cross-dataset participant overlap remain unknown. Circular defocus and
composite swap plus blur were absent from development. All footage remains
private.

| Held condition | Files | Raw exact alerts | Exact alerts retained | Unassessed |
| --- | ---: | ---: | ---: | ---: |
| Untouched camera window | 6 | 0 | 0 | 4 |
| Local Gaussian blur | 6 | 2 | 0 | 1 |
| Local circular blur | 6 | 2 | 0 | 3 |
| Short face swap | 4 | 3 | 3 | 1 |
| Face swap plus simultaneous local blur | 4 | 1 | 1 | 3 |

The rule explained all four exact ordinary blur alerts from two source groups
and both clocks. It preserved all three located swaps and the single located
compound swap. The frozen check required at least two observed compound
findings, so this comparison left that condition unqualified. Separately,
the rule preserved 17 earlier exact NTSC swap ranges and one alternate-clock
remux of an already counted range. Those repeated placements contribute no
new held compound outcome.

Independent reading rechecked 748 original-frame mappings and all 24 localized
crop/clock witnesses. The result supports discrimination of staged digital
softening in these measured cases. Optical focus, broader compressed uploads,
and a reliable combined swap-and-blur assessment remain open. The method
remains offline.
[Anonymous held observations](evaluation-panels/video-face-short-local-blur-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_face_short_local_blur.py \
  docs/evaluation-panels/video-face-short-local-blur-2026-11-19.json
```

## Matched compound follow-up

A separately frozen trial used three additional lineage-disjoint native
29.97 fps recording groups. Each contributed three previously untested
positions for a three- or four-frame comparison. At each position, the native
pictures, source clock, encoder and delivery clock match across a clean swap,
ordinary local Gaussian blur, and the same swap with exactly the same blur.
The design rotated three blur strengths across each group and held the rule
fixed.

| Matched condition | Files | Exact candidate ranges | Exact retained | Measured without a range | Unassessed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean swap | 9 | 7 | 7 | 2 | 0 |
| Ordinary local Gaussian blur | 9 | 1 | 0 | 3 | 5 |
| Swap and simultaneous Gaussian blur | 9 | 1 | 1 | 3 | 5 |

Both authored endpoints reached the fixed shortlist with their six native face
views in seven clean swaps, one ordinary blur, and one compound swap. The sole
compound finding was preserved. The separately frozen requirement was three
localized compounds across at least two recording groups, so the follow-up
also left suppression unqualified. A decision made after localization cannot
recover an unmeasured interval. The comparison identifies shortlisted timing
and usable face views in blurred pictures as the next practical limitation.

An external FFprobe pass independently verified 3,240 delivered picture
timestamps across the 27 files, including the mapped clocks for all nine
matched source interventions. The localized cases supplied 27 reproducible
native crop and timestamp witnesses. These results describe authored local
Gaussian blur on the tested carrier and leave optical focus and generalized
accuracy open.
[Matched compound observations](evaluation-panels/video-face-short-local-blur-compound-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_face_short_blur_compound.py \
  docs/evaluation-panels/video-face-short-local-blur-compound-2026-11-19.json
```

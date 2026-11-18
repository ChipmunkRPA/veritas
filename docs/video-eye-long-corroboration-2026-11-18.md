# Testing a second window for six-second eye-region observations

Requiring a second view of the same native eye region suppressed the ordinary
alerts in both panels. It also removed useful edit locations. The fresh camera
panel kept one useful later location out of 22 staged later holds, compared
with two using the unguarded six-window schedule. The trial missed its frozen
recall requirements. The local app keeps the confirmed two-second scope.
[Earlier six-second study](video-eye-long-windows-2026-11-18.md) ·
[Two-second confirmation](video-eye-30fps-confirmation-2026-11-18.md)

## Frozen check and source panels

The same 60-frame detector first ran at frames 0, 24, 48, 72, 96 and 120.
Each group of overlapping ranges and eye boxes supplied its earliest proposal.
The check chose one alternate 60-frame window that retained five context
pictures on either side of that proposal. Its start had to shift by at least
eight frames. It chose the farthest lawful start and broke ties toward the
earlier start. Both windows had to select the same eye, with both endpoints
within three native pictures and at least one-third spatial intersection over
union. It reused cached windows. Near either file edge, some proposals lack
enough pictures for a shifted window. The windows share source pixels and
measure repeatability under a change in context. Research on temporal
boundaries and joint spatial and temporal traces informed the comparison.
[BA-TFD+ benchmark](https://arxiv.org/abs/2305.01979) ·
[MVFNet](https://arxiv.org/abs/2503.20991)

The fresh panel used 11 eligible original-camera captures from 12 archived
Microsoft VCD2 recordings. The selected recordings and camera types are
distinct and all were outside the earlier eye studies and their reserve
intakes. They had exposure in other project video studies. Participant overlap
within the VCD2 archive is unknown. We verified archived bytes and exact
native frame order, retained consecutive pictures and encoded once into the
same upright 480 × 360 H.264 4:2:0 MP4 profile. One capture lacked the frozen
opening crop anchor. One admitted capture lacked a usable early eye label. The
roster, earlier-source exclusions, labels and rule were sealed before scoring.
[Camera-native dataset](https://arxiv.org/html/2603.26763v3) ·
[Publisher data license](https://raw.githubusercontent.com/microsoft/VCD/3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68/LICENSE-DATA)

The frozen gate required eight source groups with eight assessed originals,
at least six useful later locations across four groups, retention of at least
two-thirds of the unguarded useful later locations, at most two ordinary
alerts in one camera group, zero alerted whole-picture controls, and exact
native replay for both contexts of every supported match. A useful location
uses the correct eye, temporal intersection over union of at least one-half,
and at least three-quarters of the proposed crop inside the planted box.

The exploratory comparison reused the earlier Wikitongues panel. It recreated
all 22 deliveries with any original-window proposal, matched their prior MP4
hashes and reproduced the unguarded proposals before applying the same rule.
The other 50 historic files had no prior proposal to corroborate. These older
clips informed the rule's design and serve as a separate diagnostic.
[Earlier Wikitongues evaluation](video-eye-long-windows-2026-11-18.md)

## Results

| Measure | Fresh VCD2, opening → six windows → corroborated | Historic Wikitongues, opening → six windows → corroborated |
| --- | ---: | ---: |
| Useful early location | 0 → 0 → 0 / 10 | 5 → 5 → 4 / 8 |
| Useful location crossing the two-second boundary | 0 → 2 → 1 / 11 | 0 → 6 → 3 / 8 |
| Useful late location | 0 → 0 → 0 / 11 | 0 → 5 → 2 / 8 |
| Ordinary clips with an alert | 0 → 1 → 0 / 66 | 5 → 6 → 0 / 48 |
| Whole-picture controls with an alert | 0 → 0 → 0 / 11 | 1 → 1 → 0 / 8 |

In the fresh panel, 23 of 32 staged clips and 46 of 66 ordinary clips had an
assessed window. The remaining clips had unstable face geometry across all six
initial windows. The check measured 298 of 588 baseline windows and assessed
eight of the 11 originals. By comparison, all 72 historic files had at least
one assessed window. The lower base sensitivity on the fresh webcam sources
limits the value of a second-window filter.

The retained fresh edit was planted at frames `[52, 68)`. Two shifted windows
measured `[48, 68)` and `[49, 69)`. We independently decoded both sets of
before, during and after pictures and verified all six native timestamps,
original RGB hashes and crop hashes. The other useful base proposal lost its
alternate. The fresh blur control's unguarded alert `[73, 84)` was outside the
planted blur `[52, 68)`. Its alternate returned no match. The historic panel
verified 54 native context anchors for the nine retained proposal groups.
Fresh scoring took a median 0.356 s and a maximum 1.978 s, with a maximum
2.443 s including decode and context replay on the local machine.

The fresh panel missed the frozen minimum later count, camera-group count and
two-thirds retention. The historic panel retained five of 11 useful later
locations. Those losses come alongside fewer ordinary alerts. The result
supports keeping the six-second rule experimental. The planted regions use
earlier pixels from the same recordings and measure a visual stability pattern.
The evaluation did not use neural eye redraws.

## Portable replay

The compressed receipt contains anonymous recording groups, proposal windows,
boxes and native clock coordinates. It omits media, participant identifiers
and local paths. The standard-library reviewer rebuilds both tables and the
held decision. Independent pixel replay needs the licensed local inputs.
[Numeric receipt](evaluation-panels/video-eye-long-corroboration-2026-11-18.json.gz)

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_eye_long_corroboration \
  docs/evaluation-panels/video-eye-long-corroboration-2026-11-18.json.gz
```

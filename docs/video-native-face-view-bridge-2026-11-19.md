# Recovering face views around short blurred edits

In a 29.97 fps compound test, the stock face comparison found just one of
nine staged swaps with simultaneous blur. Seven of the eight misses lost
at least one expected edge before face measurement. The remaining miss had
both timings shortlisted and lacked the complete face views.

We froze two approaches to that gap. The first estimates the geometry of
unreliable middle frames from measured flank landmarks while keeping the
existing pixel-motion shortlist. The second admits an additional bounded
set of three- and four-picture pairs where a measured local detail drop
sits between low-frequency-consistent flanks. Both approaches read the
actual original pictures and preserve the existing appearance threshold.
They neither use the authored edit locations for proposals nor replace
stock findings. A separate arm applies the previous fixed local-softening
explanation to the new detections.

The 27-file development panel adds five exact compound ranges with the
second approach and four ordinary blur alerts. The separate softening
explanation withholds those four ordinary alerts and retains all five
development gains. The independent bridge holdout used existing footage
from four distinct retained swap groups and two genuine camera groups.
These sources remained outside bridge development. Their authored locations
were used previously for the separate softening study, so this comparison
leaves broader recording or training reuse open.

| Held bridge outcome | Stock timing and estimated geometry | Detail-drop proposal and estimated geometry | Detail-drop proposal, geometry and softening explanation |
| --- | ---: | ---: | ---: |
| Exact compound intervals with usable views, out of 4 | 1 | 4 | 4 |
| Added exact compound intervals, out of 4 | 0 | 3 | 2 |
| Added ordinary alerts, out of 18 | 0 | 6 | 2 |
| Ordinary files with usable candidate views, out of 18 | 5 | 10 | 10 |
| Wrong new positive intervals | 0 | 0 | 0 |

All three new unguarded compound findings matched the planted native
interval exactly. The softening explanation withheld one of them. Its
remaining two ordinary alerts came from one capture with different blur
methods at the same source location. The frozen qualification allowed at
most one new ordinary alert, so neither adding the unguarded detection
nor applying the previous rule as a veto qualified. Independent decoding
verified 27 unique original-frame and crop witnesses. The recovered
intervals are useful for developing physical evidence of a local appearance
change. They do not establish whether the image was generated or altered by
AI. This work remains offline.

[Anonymized held observations](evaluation-panels/video-native-face-view-bridge-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_face_view_bridge.py \
  docs/evaluation-panels/video-native-face-view-bridge-2026-11-19.json
```

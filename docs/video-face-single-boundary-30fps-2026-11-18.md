# One face-appearance transition at 30 fps

The current Full analysis and the added transition check reviewed the same
98 H.264 video files. They came from seven camera source groups and two
swap renderers. Each carrier used the same source groups and edit times.
The direct carrier used one AVC encode. The tandem carrier used an
additional AVC encode with stronger compression.

The added check uses the installed face localization and appearance
weights at the same pinned versions as the 25 fps check.
The publishers list MIT and Apache 2.0 terms for the respective releases.
[Localization weights](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) · [Appearance weights](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

The films contained 120 native pictures on a verified 30 fps clock. Each
staged edit changed the beginning or the end of one continuous shot. The
comparison used the earlier operating point without tuning it on these
files. Source review verified 11,760 timestamps and 1,314 substituted
native pictures before running either check. Known sources and donors were
separate from the earlier sustained-transition study. The same sources had
appeared in an earlier brief-edit study. Unknown upstream training exposure
and the identities of all depicted people were outside the comparison.
[Earlier sustained-transition study](video-face-single-boundary-2026-11-18.md)

| Held result | Direct AVC | Tandem AVC |
| --- | ---: | ---: |
| Current Full, edit boundary within one frame | 0/14 | 1/14 |
| Added check alone, exact boundary | 14/14 | 12/14 |
| Current Full plus added check, within one frame | 14/14 | 13/14 |
| Added ordinary alerts | 1/35 | 1/35 |
| Added alerts on ordinary cross-recording cuts | 0/7 | 0/7 |
| Wrong displayed boundaries on edited files | 0 | 0 |

The one additional ordinary alert per carrier came from the same focus
change. The new check assessed both missed tandem edits and found no
qualifying transition. Current Full located one of those edits. An ordinary
tandem control lacked a precise boundary and was left unassessed. A new
transition observation carries zero AI confidence weight. The original
frames around the marked timestamp show the appearance change for review.

## Local application

The opt-in Full check covers four-second 640 × 360 H.264 AVC MP4 files
with exactly 120 verified upright native frames. It uses 30.000 fps and
retains the separate 25 fps behavior. The web view and reports show the
last frame before the change, the first frame at the change, and a later
supporting frame. Exact source frame numbers and timestamps are recorded
with the three face crops.

The server implementation reproduced the complete decisions and raw
evidence on all 98 deliveries. It verified 84 original frame anchors.
Eight simulated analyses covered positive, unchanged, focus-change and
ordinary-cut cases with the option on or off. Browser, PDF, JSON and
original-file verification passed. The PDFs retained all 18 original
frame and crop images at their original pixel values. All three matched
pairs preserved the same overall assessment. One additional simulated
checkout repeated a baseline whose face check had been skipped under
memory pressure. All seven earlier signed 25 fps packages also replayed
against their source files.
[Local checkout and report receipt](evaluation-panels/video-face-boundary-native30-app-2026-11-18.json)

## Replay the counts

The [portable case records](evaluation-panels/video-face-boundary-native30-2026-11-18.json.gz)
record both checks, the expected edit boundaries, whether each candidate
result was assessed, and the native positions of the displayed frames.
They contain no source pictures, face descriptors, names, or media filenames.
Run the standard-library reviewer from the repository root.

```sh
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_boundary_native30 docs/evaluation-panels/video-face-boundary-native30-2026-11-18.json.gz
```

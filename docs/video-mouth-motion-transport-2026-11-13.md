# Picture motion and fine-detail tracking around the mouth

A continuously redrawn mouth can move throughout an edit and stay aligned
with the soundtrack. A check for briefly held pixels or mistimed speech
can miss that case. This experiment measures two picture signals from one
video, without using its audio or a reference copy. It evaluates the signals
separately and also measures whether they rise together at the same frame.
The signals share face geometry and motion estimates. Their results cannot
be interpreted as independent probabilities.
[Held mouth region check](video-mouth-local-hold-2026-11-12.md) ·
[Repeat-speech timing transfer](video-lipsync-repeat-fresh-actors-2026-11-13.md) ·
[Earlier-copy contrast](video-paired-native-mouth-transfer-2026-11-13.md)

## Fixed method

The existing MIT-licensed face geometry puts observed frames on a common
eye-aligned grid. The candidate estimates bidirectional flow using blurred
pictures. It keeps pixels where a forward and reverse step agree, then
compares lip-adjacent skin with similarly textured areas above and beside
the mouth. One signal measures abrupt local bending in the flow. The other
compares how consistently the signed fine detail moves between frames, using
fine detail omitted from the motion fit. The joint signal requires both
effects at a given frame. Each arm searches 8, 12, 16 and 20 native picture
ranges in one shot and tests against both neighboring flanks. All frame
times come from the submitted presentation timestamps. The receipt binds
the last picture before the interval, its first and middle pictures, and
the first picture after it.
[Face geometry terms](https://github.com/opencv/opencv_zoo/blob/f12e12798e8314f7c074a6656816c048dcc95b7a/models/face_detection_yunet/LICENSE)

Recent benchmark work identifies the need to score visual and audio edits
on their respective timelines and to include visual edits that preserve
genuine audio. The BabelFake dataset is restricted to noncommercial research;
the experiment used the published methodology context without using its
recordings or weights. The earlier native archive supplies the positive
cases under their already reviewed source and generator terms.
[Separate visual and audio evidence timelines](https://arxiv.org/html/2610.33413v1) ·
[Authentic audio and BabelFake dataset access](https://arxiv.org/html/2610.06339v2#S7) ·
[Native source and renderer terms](lombard-fresh-musetalk-2026-11-13.md)

The 56 ordinary GRID development files set the third-largest observed
maximum with a floor for each arm. The operating points reproduce at
0.46206669 for deformation, 0.10000000 for detail transport, and
0.29455331 for agreement. The 146 held files contain 23 neural edits, 123
ordinary controls, ten strict source-time LivePortrait edits and one
independently timing-qualified MuseTalk edit. These positive source files
had earlier project exposure and were disjoint from this experiment's
development recordings. A separate H.264 stress panel transcodes 17
source-time edits and their 17 matched originals on the same clocks.
The fixed gates require four strict LivePortrait locations across at least
three source groups, the one strict MuseTalk location, at least 70% assessed
coverage for each strict family and controls, at most two ordinary flags,
and zero flags on untouched footage, recompressions, cuts, altered
soundtracks or cadence controls. Every correct range overlaps at least half
of the known interval by temporal intersection over union.
[Source roster and matched controls](video-mouth-track-cycle-2026-11-13.md) ·
[Input and method commitments](evaluation-panels/video-mouth-motion-transport-plan-2026-11-13.json)

Four additional genuine recordings from Wikitongues and NASA supply 16
ordinary specificity controls. The original, 320 ms shifted
soundtrack and silent soundtrack deliveries have identical video clocks and
decoded pixels. The fourth delivery uses the same pictures with alternating
32 ms and 48 ms native frame steps. All 12 original, shifted and silent
deliveries reproduced identical evidence within each source and had zero
findings. All four alternating-cadence videos returned unassessed. An
initial cadence export exposed 69 of its 70 encoded pictures because of
the final timing/edit-list interaction. The corrected lossless export
verified all 70 decoded timestamps and exact decoded picture bytes.
[Conversation sources and attribution](evaluation-panels/video-lipsync-repeat-public-speech-sources-2026-11-13.json)

## Held results

| Outcome | Deformation | Detail transport | Agreement |
|---|---:|---:|---:|
| Neural edits assessed | 23/23 | 23/23 | 23/23 |
| Correct ranges on any neural edit | 0/23 | 0/23 | 0/23 |
| Correct strict LivePortrait ranges | 0/10 | 0/10 | 0/10 |
| Correct strict MuseTalk ranges | 0/1 | 0/1 | 0/1 |
| Ordinary alerts | 12/123 | 1/123 | 9/123 |
| Ordinary clips assessed | 123/123 | 123/123 | 123/123 |
| Correct ranges after H.264 re-encode | 0/17 | 0/17 | 0/17 |
| Ordinary alerts after H.264 re-encode | 0/17 | 1/17 | 1/17 |

Deformation located one strict LivePortrait finding outside its actual
edited interval, and its ordinary flags included originals and routine
recompressions. Requiring agreement still produced nine ordinary flags and
recovered no correct ranges. The single detail-transport ordinary flag was
an ordinary delayed-pixel edit. The independent MuseTalk range was
**1.038–1.702 s**. Detail transport and agreement each proposed
**1.204–1.536 s**, reaching an interval IoU of 0.50 before applying their
operating points. Both scores fell below those points. All 34 H.264
deliveries were assessed. The same original flagged in the detail transport
and agreement arms. The three arms stay outside the app.
[Portable result and evidence ledger](evaluation-panels/video-mouth-motion-transport-2026-11-13.json.gz)

## Replay

Use the retained archive from the [chroma-grid experiment](video-mouth-chroma-grid-2026-11-13.md)
and the four pinned [conversation recordings](video-lipsync-repeat-fresh-actors-2026-11-13.md#replay).
Create `$work` as a new private directory. The plain Python replay does not
need video, NumPy or OpenCV. Python 3.11 and 3.12 reproduced identical
totals. In the native review, an independent FFmpeg readback verified
1,236 held and 275 H.264 source pictures and their crop hashes. The
public ledger contains no video or rendered crops.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 -B -S -m scripts.video_mouth_motion_transport freeze "$archive" > "$rebuilt_plan"
cmp "$rebuilt_plan" docs/evaluation-panels/video-mouth-motion-transport-plan-2026-11-13.json
for phase in development held codec_stress; do
  backend/.venv/bin/python -B -m scripts.video_mouth_motion_transport "$phase" "$archive" "$work" || exit 1
done
backend/.venv/bin/python -B -m scripts.video_mouth_motion_transport ordinary_extra "$conversation_cache" "$work"
python3 -B -S -m scripts.video_mouth_motion_transport review "$work"
python3 -B -S -m scripts.video_mouth_motion_transport native_review "$archive" "$work"
python3 -B -S -m scripts.video_mouth_motion_transport review docs/evaluation-panels/video-mouth-motion-transport-2026-11-13.json.gz
```

# Brief face changes in eight-second portrait videos

The test extends the short portrait comparison to a continuous sequence of
240 native pictures. It keeps the established face distance, background
comparison and decision point. The scanner reads every other picture and
compares five sampled faces before and after the proposed interval. An
adjacent-picture shot check rejects a proposal crossing a cut in its
support. The analysis uses the original delivery timestamps for the range
and for the three before, during and after pictures.

## Frozen source comparison

Eight newly reserved publisher captures supply two staged replacements and
five ordinary controls each. Source selection excludes known prior capture
links, donor keys and exact publisher media hashes. The eight-second crop
comes from the native publisher pictures with verified source timestamps.
The delivered H.264 MP4 uses 360 × 640, upright 8-bit 4:2:0 pictures, `avc1`,
exactly 30 fps starting at zero and the direct CRF 23 encode.

The ordinary controls are the unchanged clip, a local gain change in the
picture, local face blur, a same-recording picture jump and a cut to
another recording. The staged replacement positions exercise both halves
and the area across the four-second mark. A location needs at least 0.50
temporal intersection over union with the labeled edit. The fixed decision
requires seven correct ranges from four groups, including at least three
edits beginning in the second half. It allows at most two ordinary alerts,
one misplaced range and zero alerts on cut or jump controls.
[Frozen test plan](evaluation-panels/video-face-portrait-eight-second-plan-2026-11-18.json) ·
[Source roster](evaluation-panels/video-face-portrait-eight-second-sources-2026-11-18.json)

The source roster keeps the acquisition helper's four-second recipe header.
The frozen test plan defines the eight-second deliveries used in this study.

The test uses RTFS-10K at its pinned release under the publisher's CC BY-SA
4.0 grant. The locator weights use MIT terms and the appearance weights
Apache 2.0. The panel measures controlled partial replacements. Its scope
covers this raster, codec, clock and duration. Different known donor keys
alone give limited information about the underlying people.
[Dataset and licence](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md) ·
[Locator licence](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance licence](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## Held results

| Outcome | Result |
| --- | ---: |
| Correct locations on all edited deliveries | 9/16 |
| Correct locations on edits beginning in the second half | 4/8 |
| Edits with a comparable proposal | 12/16 |
| Ordinary flags | 0/40 |
| Ordinary files with a comparable proposal | 31/40 |
| Misplaced ranges | 0/16 |

The nine correct locations cover five capture groups. Two matched both
authored boundaries exactly. All 56 deliveries and their 13,440 timestamps
passed independent clock verification. A separate FFmpeg pass reopened
129 source frame images used by proposals and verified their stored hashes.
[Held range and frame commitments](evaluation-panels/video-face-portrait-eight-second-held-2026-11-18.json.gz)

## Running local app

The existing optional short portrait mode also checks eight-second
360 × 640 `avc1` MP4s with 240 upright pictures at exactly 30 fps starting
at zero. Enable both `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW` and
`VERITAS_VIDEO_FACE_PORTRAIT_RETURN_REVIEW` in the local development
environment. The finding describes face-appearance changes at zero
weight in the overall AI assessment.

The production worker tested four second-half matches. Its existing face
checks already found each one, so the new scan withheld duplicate ranges.
The next held clip improved the best earlier overlap from 0.44 to 1.00
and returned the exact 0.667–1.067 s edit. Its before, during and after
frames are 18, 26 and 32. Both an unchanged clip and an ordinary cut
produced no new ranges. The option preserved the overall assessment in
the edit and both control pairs.

Ten isolated $9.99 simulated purchases queued zero jobs before payment.
The web view, PDF and JSON agreed on all three source frames and crops.
Chromium presented the complete evidence and sought to 0.866667 s with
zero page errors. Signed verification replayed the submitted source file.
[Local app and browser receipt](evaluation-panels/video-face-portrait-eight-second-app-2026-11-18.json)

## Replay

The standard-library review binds each range to the planned treatment and
checks recorded frame indices, timestamps and shot-guard decisions. The
full replay needs the same pinned models as the
[four-second comparison](video-face-portrait-2026-11-17.md#reproduce).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
base="$panels/video-face-portrait-eight-second"
roster="$base-sources-2026-11-18.json"
plan="$base-plan-2026-11-18.json"
result="$base-held-2026-11-18.json.gz"
runner=backend/.venv/bin/python

python3 -B -S -m scripts.video_face_portrait_eight_seconds review "$plan" "$roster" "$result"
run=$(mktemp -d)
"$runner" -B -m scripts.video_face_temporal_panel acquire "$roster" "$run/sources"
"$runner" -B -m scripts.video_face_portrait_eight_seconds run "$plan" "$roster" "$run/sources" "$run/held"
npm --prefix frontend run build
"$runner" -B -m scripts.video_face_portrait_eight_seconds_app "$run/held/receipt.json.gz" "$run/held" "$run/app"
```

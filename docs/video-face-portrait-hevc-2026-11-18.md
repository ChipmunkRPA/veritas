# Brief face changes in short HEVC portrait video

The optional local portrait face review now covers upright 360 × 640 MP4
using 8-bit HEVC Main with the `hvc1` tag. The source needs 120 presented
pictures at exactly 30 fps from a verified zero origin. The check can
locate a temporary change in one visible face and show exact uploaded
frames and face crops before, during and after the range. The timestamps
follow the original picture clock. The finding contributes zero weight to
AI confidence.

With original playback unavailable, the evidence page links to the
existing silent context clip. Its recorded clock maps frames back to the
submitted file. The surrounding and selected source images remain
available for inspection.

## Paired codec transfer

The [frozen plan](evaluation-panels/video-face-portrait-hevc-plan-2026-11-18.json)
used all 126 direct H.264 files in the 360 × 640 held portrait panel.
The source groups contribute 36 staged face replacements and 90 ordinary
controls, including unchanged files, exposure and blur changes, jumps and
cuts. Each replacement has known half-open boundaries. An exact-clock
FFmpeg encode carried the same four seconds of pictures into HEVC using
x265 CRF 24. The encoder preserved frame order and presentation times.
Both deliveries used the previously selected H.264 operating point and
adjacent-frame shot guard. An interval counts as located when it reaches
0.50 temporal intersection over union with the edit.

| Held delivery | Edits located | Edits with a comparable proposal | Ordinary clips flagged | Ordinary clips with a comparable proposal |
| --- | ---: | ---: | ---: | ---: |
| H.264 direct | 20/36 | 24/36 | 0/90 | 63/90 |
| HEVC from the same clips | 19/36 | 24/36 | 0/90 | 63/90 |

HEVC preserved 19 of the 20 H.264 locations and added zero false or
misplaced ranges. Independent FFmpeg and FFprobe reads verified all 15,120
delivered frame timestamps and the three original picture hashes for each
proposed interval. This result supports the measured carrier and size. The
two encodings share the same 18 source groups. The panel uses a controlled
re-encode of previously evaluated clips. Variable cadence, rotated uploads,
10-bit HEVC, other sizes and general face-swap accuracy on natural camera
recordings were outside this test.
[Case measurements and frame commitments](evaluation-panels/video-face-portrait-hevc-transfer-2026-11-18.json.gz)

The retained RTFS-10K comparison has publisher-declared CC BY-SA 4.0
terms. The assets used for face location and appearance use MIT and
Apache 2.0 terms. Only measurements, ranges, and hashes are in the receipt.
[RTFS-10K dataset](https://huggingface.co/datasets/godmodes/rtfs-10k) ·
[Locator license](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet#license) ·
[Appearance license](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface#license)

## App verification

Six simulated $9.99 checkouts exercised the measured edit, its unchanged
counterpart and an ordinary cut, each with the portrait review enabled
and disabled. Each job started after payment confirmation. The enabled edit
located exactly 0.533–0.933 s. The PDF and JSON agreed on source frames
14, 22 and 28 and their crops. Signed replay independently verified the
submitted HEVC file. The feature added no range for either control and
every paired run preserved the overall AI assessment.

Chromium could not play the uploaded HEVC file in the isolated browser.
The fallback opened the source-timed silent clip and sought to 0.733333 s.
Chromium displayed all three original frame images and all three crops.
[Checkout and browser receipt](evaluation-panels/video-face-portrait-hevc-app-2026-11-18.json)

## Local replay

Enable `VERITAS_VIDEO_FACE_SHORT_RETURN_REVIEW=true` and
`VERITAS_VIDEO_FACE_PORTRAIT_RETURN_REVIEW=true` in the development
environment, as with the existing H.264 portrait feature. Review the
measurements using Python 3.11 or later without model dependencies. To
re-create the HEVC deliveries, prepare the direct H.264 held panel and
configure the pinned local weights as described in the
[earlier portrait replay](video-face-portrait-2026-11-17.md#reproduce).

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
panels=docs/evaluation-panels
plan="$panels/video-face-portrait-hevc-plan-2026-11-18.json"
held="$panels/video-face-portrait-held-2026-11-17.json.gz"
freeze="$panels/video-face-portrait-calibration-2026-11-17.json.gz"
roster="$panels/video-face-portrait-sources-2026-11-17.json"
result="$panels/video-face-portrait-hevc-transfer-2026-11-18.json.gz"

python3 -B -S -m scripts.video_face_portrait_hevc_transfer review "$plan" "$held" "$freeze" "$roster" "$result"
backend/.venv/bin/python -B -m scripts.video_face_portrait_hevc_transfer run \
  "$plan" "$held" "$freeze" "$roster" "$prepared_h264_held_640" "$new_hevc_run"

npm --prefix frontend run build
backend/.venv/bin/python -B -m scripts.video_face_portrait_hevc_app \
  "$result" "$new_hevc_run" "$new_checkout_run"
```

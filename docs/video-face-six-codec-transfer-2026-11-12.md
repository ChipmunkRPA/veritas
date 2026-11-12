# Six-picture face-change review on VP9/WebM and HEVC/MP4

Both codecs recovered all eight held six-picture swaps. Each also added a range to two genuine whole-picture cuts. The same pair of recordings produced those two cuts in both codecs. Both transfers failed the frozen cut gate and remain experimental. [Aggregate and independent replay](evaluation-panels/video-face-six-codec-transfer-2026-11-12.json)

## Frozen comparison

The local face appearance check was qualified on H.264 for several exact clip lengths. This study tests the same guarded rank fallback on 4.8-second **VP9/WebM** and **HEVC/MP4** files. Both have 120 upright 640 × 360 pictures, 4:2:0 color sampling and an exact 25 fps native clock. When the earlier 8 Hz scan measures a candidate and yields no distinct range, the fallback uses every second native picture, the same strict separation point, the same surrounding-scene requirements and the existing duplicate check. [Prior six-picture results](video-face-six-picture-transfer-2026-11-12.md) · [H.264 duration transfer](video-face-six-duration-transfer-2026-11-12.md)

The panel uses all four remaining eligible source groups with known original links and donor keys disjoint from the 37 prior localization groups and from each other. Two earlier development groups are reused for execution coverage at the fixed point. Two held sources have the publisher's head-rotation tag. Some recordings had separate project exposure, and the separation claim covers documented source links and donor keys. The local comparison uses the previously admitted weights and the retained publisher media. [Rights and archive scope](video-common-source-preparation-v89.md) · [Native source preparation](video-native-inserts-v95-method.md)

Each held codec has eight six-picture swaps across both publisher families and four source groups, 20 matched ordinary deliveries and four whole-picture cuts. Each swap is a **0.240-second** insert at **2.160–2.400 s** or **2.680–2.920 s**. The two codecs encode the same authored pictures at the same times. Matched controls include an original, stronger tandem re-encoding, local illumination, local blur, and local time-step changes. Each cut switches to another genuine source and returns at the recorded boundary. The recipes use VP9 CRF 32 and 44 and HEVC CRF 26 and 38. These are matched inputs and documented encoder settings. The comparison uses directly observed faces and the serving appearance calculation, with the complete application tracking path outside the benchmark.

The frozen gate applies separately to each codec. It requires at least four additional correct intervals across three source groups, at least 90% fallback coverage on swaps and ordinary deliveries, up to one added ordinary range, and zero added ranges on cuts. A correct interval needs at least 0.50 temporal IoU. An earlier exact-30 fps HEVC and AV1 test used longer, 11-picture swaps and measured different coverage. [Earlier 30 fps comparison](video-short-return-codec-transfer-2026-11-11.md)

## Held outcomes

| Held outcome | VP9/WebM | HEVC/MP4 |
| --- | ---: | ---: |
| Correct swap locations, incumbent → combined | 0/8 → 8/8 | 0/8 → 8/8 |
| Sources with added correct locations | 4/4 | 4/4 |
| Ordinary false flags, incumbent → combined | 0/20 → 0/20 | 2/20 → 2/20 |
| Cut false flags, incumbent → combined | 1/4 → 3/4 | 1/4 → 3/4 |
| Measured fallback views | 32/32 | 32/32 |

Each codec produced four exact ranges and four ranges with both endpoints 40 ms later than the authored interval. The median IoU was 6/7 and the minimum was 5/7 in both codecs. Both selected **2.160–2.400 s** exactly in the same held example, with native before, during and after crops at **2.080**, **2.240**, and **2.400 s**. The held CPU pass measured 6,336 sampled positions in 178.1 seconds.

The independent audit rebuilt all 64 held files byte for byte from 600 archived native source pictures. It checked 7,680 delivered pictures, both actual carrier headers and 60 selected face crops, including the false cut ranges. The app's existing measurement function separately reproduced the exact swap and cut ranges on one held input of each type and codec. It returned no range on the paired genuine originals. All six measurements completed within 2.7 seconds of the existing 12-second budget. This private measurement preview used the frozen sample grid and left the product eligibility scope at the already validated H.264 profiles.

## Matched H.264 cut diagnostic

After scoring, we encoded the same eight swaps, four cuts and four originals in H.264 at CRF 23. The unchanged fallback recovered **8/8 swaps**, flagged **2/4 cuts** and left **0/4 originals** flagged. The same two cut directions produced the fallback's false flags in all three codecs. This reproduces the cut failure inside the existing H.264 delivery profile. The earlier H.264 transfer used different source pairs.

The diagnostic reopened 120 native boundary pictures, verified their decoded hashes and presentation timestamps, and recomputed the existing background correlations outside the detected face. For each interval it took the lower result across its two boundaries. The observed ranges below describe only the eight correctly selected swaps and two selected false cuts per codec.

| Delivery | Swap median tile correlation | False cut median tile correlation | Swap lower-quartile tile correlation | False cut lower-quartile tile correlation |
| --- | ---: | ---: | ---: | ---: |
| VP9/WebM | .940–.997 | .674–.711 | .914–.993 | .265–.291 |
| HEVC/MP4 | .942–.997 | .664–.683 | .913–.994 | .276–.290 |
| H.264/MP4 | .938–.998 | .714–.727 | .913–.992 | .292–.296 |

The observed cuts changed more of the non-face background at their boundaries. This suggests a stricter scene-continuity check for a separate, prospectively frozen evaluation. That evaluation needs fresh sources and ordinary cuts with similar backgrounds. This trial keeps the frozen guard and the serving scope unchanged.

## Reproduce

Use the retained native archive and private prior run folders. The independent audit checks native presentation timestamps and crops and recreates the authored files byte for byte.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze prepare development held report; do
  backend/.venv/bin/python -B -m scripts.video_face_six_codec_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_six_codec_transfer "$run"
backend/.venv/bin/python -B -m scripts.diagnose_video_face_six_codec_cuts "$run"
```

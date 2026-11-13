# Find a short picture change without a face finding

Chipmunk Verify can now locate a short change in the surrounding picture
even when the person on screen stays the same. In the local app, open
**Media Evidence** and look for **Short picture change and return**. The
browser, PDF and JSON use the same source interval and three exact decoded
frames from before, during and after the change. This observation carries
zero AI confidence weight. An ordinary edit can produce the same result.

The browser and Standard PDF also show a marked copy of the during
frame alongside the clean source image. Each outlined area corresponds
to a measured tile that supports the start and end of the interval.
The JSON lists the same tiles under `/packets/{index}/scene_return/tiles`.
Tile indices count from zero, left to right and top to bottom on a
10-column by 6-row grid. The measurements support the interval timing.

New analyses also record whether all 120 frames were checked, whether
a matching range was found, and why a check was unavailable when it
could not be completed. Media Evidence, the Standard PDF and JSON use
the same record. The optional observation has its own source commitment.
An older analysis keeps its original recorded evidence.

The optional Full and Deep check covers upright 1280 × 720 MP4 files
with H.264 (`avc1`) or HEVC (`hvc1`), 8-bit 4:2:0 chroma and exactly
120 verified frames at 30 fps.
The check looks for a change lasting 6–24 frames followed by a return to a
similar picture. It can miss edits outside that range, cuts without a return,
and changes in low-texture areas. To enable it on a development instance,
set `VERITAS_VIDEO_SCENE_RETURN_REVIEW=true` and restart the worker.

## Method and test

Published shot-transition work measures changes across time and considers
spatial motion separately from color. This implementation compares local
normalized luminance texture in 60 tiles and checks for brief entry and
exit changes with stable neighboring frames. Tiles support the result only
when the picture agrees across the two outside frames and across the two
inside frames. The rule requires support in all four quadrants and checks
for camera translation using phase correlation. The existing shot plan
and face detector inputs keep their original shot boundaries.
[TransVLM](https://arxiv.org/abs/2604.27975v2) ·
[OmniShotCut](https://arxiv.org/abs/2604.24762v2) ·
[Motion and cross-correlation study](https://arxiv.org/abs/2502.09202)

The consumed development panel had 32 deliveries. The standalone check
localized eight of eight layout changes and five of eight combined
speaker-plus-layout changes, with no ranges on the 16 unchanged or
speaker-only controls. The frozen confirmation used four different pairs
of CREMA-D actors, each rendered in both directions. All eight actors
were excluded from development and both older CREMA-D cut banks.
The source recordings had earlier project exposure. The confirmation used
new geometric layouts, durations and lighting and camera-shift controls.
[Previous routing test](video-scene-only-cut-transfer-2026-11-13.md) ·
[CREMA-D source terms](https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eebd53610a08b1dc/LICENSE.txt)

| Held delivery | Files | Existing shot check, both exact boundaries | Standalone check, exact range |
| --- | ---: | ---: | ---: |
| Layout changes with speaker pixels unchanged before encoding | 8 | 0 | 8 |
| Speaker and layout both change | 8 | 0 | 6 |
| Unchanged, speaker only, camera shift, color grade, full camera | 40 | 0 | 0 |

## H.264 transfer

Two H.264 encodes of the same source panel preserved all 14 exact
ranges, the supporting tiles, and both misses. Neither encode flagged
any of the 40 matched ordinary controls. The CRF 21 encode also flagged
zero of 18 supplemental ordinary clips derived from distinct retained
sources, including 12 one-way cuts and six continuous retimes.
The same source material appears in both encodes. This compares delivery
formats and does not add independent positive examples.
[Paired carrier ledger](evaluation-panels/video-scene-return-h264-transfer-2026-11-13.json)

## Local application verification

The 14 displayed ranges matched both authored boundaries, including
the exclusive end. FFmpeg independently reproduced all 6,720
decoded timestamps and 42 selected source images. The positive and control
ran through the local Full face pipeline with simulated payment. The
Standard PDF embedded all three exact source images. The public evidence
package verified its signature and reproduced the interval and tile support
from the original video. Chrome loaded the same three source images in
the plain view and matched the JSON to the evidence API. The test
isolated the face/tracking lanes and the new check. The same video with
the option disabled produced the same overall decision and omitted the
picture-change card. All three analyses queued after simulated payment.
[Scalar ledger](evaluation-panels/video-scene-return-transfer-2026-11-13.json) ·
[Local application receipt](evaluation-panels/video-scene-return-local-2026-11-13.json)

A separate local check exercised the four coverage outcomes that arise
from normal uploads. The range with a match displayed three source
frames, the ordinary camera movement reported a completed scan with
no match, a 360p H.264 control was reported as unsupported,
and the Quick profile said the check had not run. Chrome, the PDF and
JSON agreed on the outcomes, including a matching range on both supported
carriers. All analyses queued after simulated payment and all signed
packages verified.
[Coverage receipt](evaluation-panels/video-scene-return-coverage-2026-11-13.json)

## Reproduce

The standalone reader uses the published ledger and Python standard
library. Add `--media "$run/media"` to reproduce exact source pixels and
timestamps using FFmpeg if you have the retained deliveries.

```sh
python3 -B -IS backend/scripts/review_video_scene_return_transfer.py \
  docs/evaluation-panels/video-scene-return-transfer-2026-11-13.json
```

To recreate the local study, use a private scratch directory, the
retained earlier freeze and the publisher source assets documented in
the previous routing test.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.video_scene_return_transfer freeze "$run" --prior "$prior"
backend/.venv/bin/python -B -m scripts.video_scene_return_transfer prepare "$run"
backend/.venv/bin/python -B -m scripts.video_scene_return_transfer score "$run"
```

For the paired H.264 test, use that completed HEVC run and separate
scratch directories for the two encoder recipes.

```sh
for recipe in veryfast21 medium29; do
  for phase in run audit; do
    backend/.venv/bin/python -B -m scripts.video_scene_return_h264_transfer "$phase" \
      "$scratch/$recipe" --hevc "$run" --recipe "$recipe" || exit 1
  done
done
```

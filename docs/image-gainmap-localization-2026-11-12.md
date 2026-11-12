# Local differences between an Ultra HDR JPEG and its gain map

Two checks compared the main image with the HDR gain map carried inside the same JPEG. The edge check located four of 12 held conventional edits and highlighted none of 36 ordinary views. The brightness check also located four edits, with alerts on six variants of one ordinary source. Both missed their frozen requirements. A retrospective union located eight of 12 edits while retaining the one flagged ordinary source group. The checks stay outside the app. [Audited result](evaluation-panels/image-gainmap-localization-2026-11-12.json)

An Ultra HDR gain map records how much to brighten parts of the main image on an HDR display. Editing the main image can leave outdated detail in the map if an editor preserves the map without updating it. The first check measures strong map edges without nearby edges of matching orientation in the main image. The second measures an unexpectedly dark patch relative to other areas of the same gain range. Both rank a compact rectangle relative to surrounding support. They require a bounded grayscale gain-map JPEG declared by the XMP container and carrying gain-map metadata of its own. Both checks measure disagreement, with zero AI authorship weight. [Ultra HDR format](https://developer.android.com/media/platform/hdr-image-format) · [Google Research on stale gain maps](https://research.google/blog/hdr-photo-editing-with-machine-learning/)

Ten published original JPEGs from one Pixel 6 Pro supplied separate calibration and holdout photo groups. Four groups set each point to the largest ordinary control response. The six holdout groups contributed an untouched camera file and five ordinary variants each. Those variants include an unchanged base re-encode, JPEG75, global tone adjustment, a resize round trip and sharpening. Another six files omit the gain map. Each positive darkens and smooths a local base-image rectangle with feathering while keeping the publisher gain map unchanged. One rectangle is chosen from high-gain areas and one from a deterministic position independent of the measurement. The edited-pixel masks come from exact differences before JPEG encoding. All derived test files omit the EXIF thumbnail. [Published photos and attribution](https://github.com/MishaalRahmanGH/Ultra_HDR_Samples/tree/de109721ca479b2b0f31cc709e70b614531b413f)

| Held outcome | Orphaned gain-map edges | Dark relative to nearby gain | Retrospective union |
| --- | ---: | ---: | ---: |
| High-gain edits located /6 | 1 | 2 | 3 |
| Arbitrarily placed edits located /6 | 3 | 2 | 5 |
| Ordinary views highlighted /36 | 0 | 6 | 6 |
| Ordinary photo groups highlighted /6 | 0 | 1 | 1 |

Useful localization requires at least 30% intersection over union and 50% recall. Each individual check needed at least three high-gain edits and two arbitrarily placed edits with at most one ordinary source flagged. The union was evaluated after seeing the results. The six ordinary alerts all occurred on one photo, including its unchanged camera JPEG. The independent readback rebuilt all 54 held inputs, verified 96 native rectangles and confirmed that 48 held views lacked an IFD1 thumbnail. The photos share one camera and contributor. The edits test a deliberately stale map under controlled conditions. Those conditions leave transfer to other cameras and to unknown uploads unestablished.

## Replay

Use the pinned original files and the corresponding `acquisition.json` values under `intake` in the linked receipt. Preserve the original bytes. Set `run` to a new local directory. Source photos and derived images stay local.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze calibrate held; do
  backend/.venv/bin/python -B -m scripts.image_gainmap_localization_transfer "$stage" "$archive" "$run"
done
backend/.venv/bin/python -B -m scripts.review_image_gainmap_localization_transfer "$archive" "$run"
```

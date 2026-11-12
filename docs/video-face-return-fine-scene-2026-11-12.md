# Fine background support for brief face changes

The existing short face check requires eight textured background tiles at each proposed boundary. Two previously studied swaps had one face throughout and strong face-change responses, with only seven usable tiles at both boundaries. The current check could not assess scene continuity there. A finer grid measures smaller regions of the background while requiring evidence across several parts of the picture. [Earlier eligibility and native evidence results](video-face-return-eligibility-2026-11-12.md)

## Fixed method and source scope

At 192 × 128, divide the image into 8 × 16 tiles and exclude tiles touching either measured face box with the existing 10% margin. Retain the luminance texture floor of four levels. The new edge check requires at least 16 textured tiles across four macro sectors, and at least three quadrants with three tiles apiece and a lower-quartile correlation of at least .82. The overall median correlation must reach .90. Across all unmasked tiles, including flat tiles, the 90th percentile of mean absolute luminance and chroma differences must stay at or below eight and six levels respectively. [Spatial edge change research](https://ecommons.cornell.edu/entities/publication/28f739f0-d4db-4487-990f-89204f65452a)

The finer measurement fills only an edge the existing check could not measure. An observed correlation below the incumbent requirement still rejects the edge. The Full study reruns the exact dense grid only when the current short check produced no range after complete valid source measurement. It uses the remaining original inspection budget, preserves prior ranges, and keeps the existing face geometry, rank, duration and overlap rules. A candidate needs at least one finer edge. The production app and evidence versions stay at their previous revisions during the study.

The two earlier misses and their cut pair were development examples only. The prospective panel uses three other known source groups. Source selection used only incumbent background-tile availability on existing genuine pictures. Each group was exposed to earlier project studies and remains separate from the development pair by documented source and donor keys. The panel uses seven known short swaps, eight genuine cross-recording cuts with insufficient original background support at both authored edges, and sixteen ordinary controls. The controls include unchanged originals, compression, lighting, a camera punch-in and genuine same-recording jumps. [Source and rights](video-common-source-preparation-v89.md) · [Earlier paired source comparison](video-native-background-edges-2026-11-12.md)

All held files are upright 640 × 360 H.264 with 4:2:0 chroma, exactly 120 native pictures at 25 fps. Five swap cases, all cuts and all ordinary controls reuse prior native files. The two added swaps use authenticated publisher pictures over **2.160–2.400 s** and **2.680–2.920 s**, with each source picture in the authored interval verified to differ from the original. The sealed gate requires three or more newly localized swaps across at least two source groups, with at least 0.50 temporal intersection over union, at most one additional ordinary flag, zero additional cut flags, and preservation of earlier results.

## Held outcomes

| Held cases | Current | With finer background support |
| --- | ---: | ---: |
| Useful short swap ranges | 1/7 | 5/7 |
| Genuine cut alerts | 2/8 | 2/8 |
| Other ordinary alerts | 0/16 | 0/16 |

The four newly localized swaps came from two groups. Two selected **2.160–2.400 s** exactly. Two selected **2.720–2.960 s** against **2.680–2.920 s**, one native picture later at each endpoint. Their before, during and after pictures were at **2.080**, **2.240**, **2.400 s** and **2.640**, **2.800**, **2.960 s**, respectively. All selected edges had measured faces and six or seven usable original tiles. The finer grid supplied 26–32 textured tiles across three qualifying quadrants and four or five macro sectors.

The two remaining missed six-picture swaps came from the third group. The gate passed on the declared low-texture panel. Both genuine cut alerts were existing coarse-stage ranges and remain visible in the counterfactual. The result measures local inspection ranges and provides no new estimate of AI confidence or broad false-alarm rates.

The Full face stages completed 31 cases in 154.4 seconds. Independent replay reopened all 3,720 delivered pictures, reconstructed both newly authored files byte for byte, and verified 44 measured face inputs, 16 edge pictures and 12 selected source crops. It recalculated all 31 decisions. [Aggregate and native replay](evaluation-panels/video-face-return-fine-scene-2026-11-12.json)

## Reproduce

Use the retained source archive and pinned prior development receipts. The trial runs the existing local Full video face stages in process with an in-memory detector receipt. The independent reviewer reads the exact submitted native pictures and checks selected face crops, timing and background support. No analysis checkout or persistent database is used.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in prepare score report; do
  backend/.venv/bin/python -B -m scripts.video_face_return_fine_scene_transfer "$phase" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_video_face_return_fine_scene_transfer "$run"
```

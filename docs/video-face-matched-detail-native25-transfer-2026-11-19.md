# Matched-detail short face review on native 25 fps video

This frozen carrier comparison checked 30 upright 720p AVC MP4 files, each with 120 native pictures over 4.8 seconds. Six retained capture groups supplied separate donor, source-link and component keys from each other and the preceding brief-face and blur cohorts. Other project exposure and participant or model-training overlap remain open. The source footage stays in the local archive. [Pinned source terms](https://huggingface.co/datasets/godmodes/rtfs-10k/blob/bca865e675852666a426368b2f5614abf1c1926a/README.md)

Each group contributed a three- or four-frame staged face swap, an ordinary local blur over the same region and interval, and the same swap with the matched blur. Untouched and hard-cut camera controls used the same carrier. The three- and four-frame widths, retained swap renderers, and Gaussian and circular blur treatments were balanced before scoring. An independent reader verified 3,600 decoded RGB frames and native presentation timestamps, including 21 distinct authored source-frame positions.

The comparator ran the installed Full face-analysis stages with the current opt-ins enabled. The comparison adds the unchanged matched-detail review on the verified 25 fps carrier and holds its appearance point and native picture schedule fixed. An existing brief interval takes precedence and earlier valid findings stay intact. The timing budget and shot-boundary exclusions remain in effect. The native comparison is a physical appearance observation and carries zero weight in the AI assessment.

| Held condition | Files | Full exact | Full plus fixed check exact | Unassessed by fixed check |
| --- | ---: | ---: | ---: | ---: |
| Clean brief swap | 6 | 0 | 0 | 6 |
| Brief swap with blur | 6 | 0 | 4 | 2 |
| Matched ordinary blur | 6 | 0 alerts | 0 alerts | 1 |
| Untouched camera window | 6 | 0 alerts | 0 alerts | 6 |
| Ordinary hard cut | 6 | 0 alerts | 0 alerts | 6 |

The four additions cover three three-picture intervals and one four-picture interval across four capture groups. Both Gaussian and circular blur and both retained swap renderers contributed. All four additions have exact native boundaries. Full produced one earlier coarse range with one frame of overlap and no useful location in the compound set. That range was preserved. The candidate introduced no wrong positive range and no ordinary alert, with comparable views in five of the six matched ordinary blurs.

The frozen rule required six ordinary files with comparable views. Five qualified, so the transfer remains experimental. Unassessed controls contribute no evidence about ordinary false-alert behavior. A source-disjoint follow-up should add more naturally difficult blur, rapid head movement and refocus controls, select them before measuring the response, and preserve the original decision point. The current comparison does not establish detection performance for clean swaps or other 25 fps input lengths and carriers.

Independent original-file replay reconstructed all 37 measured face crops supporting the four findings, including the 12 before, during and after display frames. The anonymous receipt preserves the interval and control evidence without source images, identifying details or model weights. [Anonymous evaluation receipt](evaluation-panels/video-face-matched-detail-native25-transfer-2026-11-19.json)

```sh
python3 -I -S backend/scripts/review_video_matched_detail_native25.py \
  docs/evaluation-panels/video-face-matched-detail-native25-transfer-2026-11-19.json
```

# Short face-change checks on H.264 MP4 and VP9 WebM

The existing eight-second short face-change check admits HEVC MP4. This study runs the unchanged three-window measurement directly on H.264 MP4 and VP9 WebM. It compares the same media with the current Full face stages. Native times matter because ordinary WebM stores this cadence in whole milliseconds, including frame intervals of 33 and 34 ms. A result is localized face-appearance evidence with no overall AI confidence weight. [Existing eight-second HEVC study](video-face-short-prefix-transfer-2026-11-12.md) · [Primary codec robustness research](https://pmc.ncbi.nlm.nih.gov/articles/PMC13337803/) · [Short-interval localization research](https://arxiv.org/abs/2606.07090)

## Frozen comparison

Two development and six held source groups use eight continuous seconds from the qualified RTFS archive. Documented source URLs, donor keys and exact source components are disjoint within this study and from the eight preceding HEVC groups. The archive had earlier project exposure. Each authored positive inserts a six-picture range at the corresponding times in a publisher face-swap video. One view has both an early and a late change. Ordinary controls include the original, a repeated whole-picture interval, a jump to later pictures from the same recording, exposure changes and a brief cut to a separate genuine recording. The ordinary cut and late swaps use the same authored boundaries. [Source qualification and rights](video-common-source-preparation-v89.md)

Both carriers are created independently from the same source pictures. Primary H.264 uses CRF 23 and VP9 uses CRF 30. Tandem copies use CRF 35 and 41 respectively after the primary encode. Each delivery has 240 upright 1280 × 720, 8-bit 4:2:0 pictures, with exact 30 fps for MP4 and the rounded 30 fps lattice in 1 ms ticks for WebM. The check uses overlapping windows beginning at frames 0, 60 and 120. The appearance and surrounding motion comparison uses its prior 0.10 operating point. Source-specific times and RGB commitments are retained for the before, during and after pictures.

The per-carrier gate requires nine additional useful late locations across at least four groups, six additional exact late locations, at least 75% fully measured coverage, at most one new ordinary range, no new whole-picture loop or genuine-cut ranges, and at most one wrong range on an edited file. A useful localization needs at least half temporal intersection over union with the authored range.

## Held results

| Carrier | Edits localized by existing Full | Edits localized with the added check | Added exact late edits | Ordinary files flagged by existing Full | New ordinary flags | Fully assessed files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H.264 MP4 | 0/30 | 27/30 | 15/18 | 0/30 | 0/30 | 54/54 |
| VP9 WebM | 0/30 | 30/30 | 18/18 | 5/30 | 0/30 | 54/54 |

Both carriers met all frozen criteria, recovered exact late edits in all six source groups, and added zero wrong edited ranges. Every added location used the two authored frame boundaries. The three H.264 misses followed the second compression pass. Full already flagged five ordinary VP9 variants of the same source group, with the same broad range in each. Those existing findings remain. The development counts were 7/10 H.264 and 8/10 VP9 exact locations and 0/10 new ordinary flags for each.

An independent FFprobe and FFmpeg pass verified the 5,040 source timestamps and 81 original RGB anchor frames for all 27 H.264 locations, and the 5,760 source timestamps and 90 original RGB anchor frames for all 30 VP9 locations. The two carriers use the same source groups, so the results compare delivery formats within six independent held groups. This study establishes a narrow capability for six-picture inserted face changes. Results for other resolutions, longer uploads, irregular timing, multiple faces and lip sync need separate evidence. [Frozen comparison and source commitments](evaluation-panels/video-face-short-carrier-transfer-2026-11-12.json)

## Local application verification

`VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW=true` enables this optional review for qualified clips. The application reports actual source times, the original and face crop before, during and after a range, and which windows were assessed. A time-limited window is marked unassessed. Some offline measurements took 17–20 seconds under concurrent work, so offline measurement coverage does not estimate production completion rates. Local paid-path replay showed the optional check leaves the overall decision unchanged and verified the existing HEVC evidence path. Signed source replay matched the original pixels and native timestamps, recomputed the selected boundaries, and rejected alternate carrier and altered source files. Chrome opened both carriers and sought the selected frame at its native time. [Paid worker, exports, signed replay and Chrome receipt](evaluation-panels/video-face-short-carrier-app-2026-11-12.json)

The following command checks the public comparison counts without model weights or media. With the original encoded panel available locally, append `--audit-source /path/to/carrier-panel` to independently recheck the native timestamps and RGB frame commitments.

```bash
PYTHONPATH=backend python3 -B -S -m scripts.review_video_face_short_carrier_transfer \
  docs/evaluation-panels/video-face-short-carrier-transfer-2026-11-12.json
```

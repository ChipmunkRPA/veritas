# Local color differences between earlier and analyzed video

The shipped earlier-copy check compares luminance after accounting for global
brightness and contrast. This trial checks whether color difference adds a
useful location when the pictures share the same source clock. A localized
lip redraw can retain its source-time speech event, and visual differences
may survive without a timing anomaly. Recent work examines how multiscale
residuals supplement visual context, and how separate visual and auditory cues
can inform time localization. The test uses an independently implemented
color calculation with no model weights. [Multiscale video residuals](https://arxiv.org/abs/2611.12508)
· [Temporal forgery localization](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_Structural-Semantic_Perception_for_Diffusion-Guided_Temporal_Forgery_Localization_CVPR_2026_paper.html)
· [Recent lip-redraw research](https://arxiv.org/abs/2610.23558)

## Frozen design

Both files must have matching verified picture timestamps and pass the shipped
shared-picture luminance gate. The new arm measures decoded Cb and Cr after
removing a whole-frame color adjustment. It uses robust fits from corresponding
locations, excluding the largest deviations. It scans 8, 12, 16 or 20 pictures
and compares the strongest tile with five pictures on both sides. The existing
connected extent rule expands that tile into a native-pixel rectangle. Neither
sound nor a marked facial area is required. Each selected range keeps the two
original images and exact source crops at before, during and after positions.
[Shipped luminance comparison](video-paired-local-area-2026-11-13.md)

Eight genuine GRID sources set the chroma point, with an identical copy, a
separate compression and a whole-picture tone and color adjustment for each.
The held bank reuses 17 neural source pairs from two families plus 34 ordinary
same-time resamples and earlier-pixel replacements. Eleven redraws kept an
independently measured local speech timing event. A separate source bank has
eight CREMA-D recordings, each with an authored local color change, an identical
copy, a recompression, a whole-picture color shift and a wrong-source delivery.
The source recordings were previously used in other tests. The exact local
color edits were newly authored. The two source banks are disjoint from the
GRID fit. [Earlier mouth pairs](video-paired-native-mouth-transfer-2026-11-13.md)
· [GRID source and terms](https://zenodo.org/records/3625687)
· [Pinned CREMA-D media](evaluation-panels/video-paired-local-area-sources-2026-11-13.json)

The fixed gate requires two added useful redraw locations in distinct source
groups, including a second-family case, and at least six useful exact-color
control locations. A useful region reaches at least half temporal overlap and
0.10 middle-picture intersection over union with the actual changed pixels.
The gate allows one new flag on 24 ordinary delivery controls and zero admitted
wrong-source pairs. Actual ordinary local changes count separately as factual
visual differences.

## Held results

| Outcome | Luminance | Chroma | Either |
| --- | ---: | ---: | ---: |
| Useful neural redraw locations | 14/17 | 6/17 | 15/17 |
| Useful locations on timing-qualified redraws | 10/11 | 1/11 | 10/11 |
| Useful locations on newly authored local color changes | 6/8 | 8/8 | 8/8 |
| Ranges on ordinary local mouth operations | 27/34 | 17/34 | 27/34 |
| Ranges on identical, recompressed or global-color controls | 0/24 | 0/24 | 0/24 |

All 83 same-source pairs passed the shared-picture gate. All eight wrong
sources were declined. Chroma localized all six redraws from the second
generator and zero of 11 from the first. The new useful case matched the
authored **14–30** picture interval exactly. The luminance arm selected
**23–31**. That case lacked an independently qualified local timing event.
The change remains experimental. The gain is confined to one source group
and provides no evidence about detecting lip redraws from a single upload.
Independent FFmpeg replay verified **486 exact crops and corresponding
original frames** across 81 flagged comparison ranges. The initial attempt
finished the neural bank, then stopped because the public cache used different
filenames. The rerun resolved the same eight files by their pinned hashes and
finished the unchanged public bank.
[Native time, geometry and crop ledger](evaluation-panels/video-reference-chroma-transfer-2026-11-13.json)

## Replay

The standard-library reader reproduces the frozen decision arithmetic. Run the
scorer with the retained native pairs and the CREMA-D files as `asset-*.flv`
in the archive directory. The reviewer can independently reopen both source
clocks and exact decoded crop commitments with FFmpeg.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze development held; do
  backend/.venv/bin/python -B -m scripts.video_reference_chroma_transfer \
    "$phase" "$previous" "$roster" "$run" --archive "$archive" || exit 1
done
backend/.venv/bin/python -B backend/scripts/review_video_reference_chroma_transfer.py \
  "$receipt" --assemble "$run" --previous "$previous"
python3 -B -IS backend/scripts/review_video_reference_chroma_transfer.py "$receipt"
```

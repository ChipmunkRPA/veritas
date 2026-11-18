# Color and brightness consistency in a single JPEG

This experiment tests whether a small generated removal changes the
relationship between color detail and brightness in the submitted JPEG.
The independent proposal uses only that file. Original images and known
edit masks are used to construct and score the evaluation. The design
draws on research that measures local pixel dependencies separately from
broader image context. It uses a project-authored measurement and no
weights from that research. [Pixel inconsistency study](https://arxiv.org/abs/2310.00234)

The method reads the native quantized coefficients from an upright 8-bit
4:2:0 JPEG. It reconstructs luminance, pools it onto the chroma grid, and
compares eight low-frequency AC coefficients across the matching color and
brightness blocks. Measurements use 3 × 3 neighborhoods of 16-pixel source
cells. For each neighborhood the comparison selects up to 24 distant
neighborhoods in the same image with similar brightness, luminance detail,
and base chroma. It measures unexpected loss of color detail, excess color
detail, and weaker than expected color-brightness agreement separately.
Each measurement ranks native rectangles. Rectangles require measured
cells over at least 85 percent of their area. A known edit is counted at
0.30 pixel intersection over union with at least half its changed pixels
covered. These are research criteria for judging the proposed location.

Fourteen earlier scene-text sources set the operating points. The point
for each reading and delivery quality allows one additional ordinary source
group during calibration among images the incumbent checks left clear. The
arms are selected on those calibration sources independently for JPEG75
and JPEG90. The held reserve specifies the next 24 entries in the pinned
publisher inventory. Eighteen previously unused source images cleared the
source checks. A 512-bit thumbnail-gradient comparison between the two
scene-text splits differed by at least 219 bits. Each held source has sharp
and feathered placements of the published
Nano Banana 2 removal inside the marked area. Ordinary controls include
unchanged files, matched blur, tone, copying and conventional fills. The
camera-image transfer uses 12 retained BG-REAL sources, LaMa and MI-GAN
removals, and separate conventional fill controls. These photos did not
contribute to the points. The existing aligned double-JPEG and regional
checks ran on the same exact JPEGs, including the app's repeat view at
quality 90. The publisher roster does not establish how every scene-text
source image was made. [Pinned scene-text dataset and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [Camera inputs, editor terms, and replay](image-masked-reconstruction-2026-11-17.md#reproduce)

## Held results

Calibration selected weaker color–brightness agreement at both delivery
qualities. The internal points were 8.53483 for JPEG75 and 9.06901 for
JPEG90, in within-image robust deviation units.

| Panel | JPEG quality | Useful locations, current → combined | Ordinary flags, current → combined | Misplaced candidate areas |
| --- | ---: | ---: | ---: | ---: |
| Fresh scene-text | 75 | 2 → 2 of 36 | 8 → 20 of 90 | 6 |
| Fresh scene-text | 90 | 0 → 0 of 36 | 13 → 24 of 90 | 3 |
| Retained camera | 75 | 0 → 0 of 48 | 16 → 32 of 120 | 8 |
| Retained camera | 90 | 4 → 4 of 48 | 10 → 26 of 120 | 8 |

All three readings assessed every file. None of their strongest rectangles
met the edit-overlap criterion on any of the 168 held edits before the
cutoff. An exhaustive geometry check used the actual edit masks to find
qualifying rectangles on 134 of 168. That identifies ranking as a limit even
where this grid could represent a useful location. The three readings stay
in evaluation. The current comparison here covers two app checks, and the
counts apply to this pair of editors and the scene-text staging. Camera
ordinary deliveries can repeat across the two editors. These are delivery
counts, with 12 distinct camera image sources.

[Calibration](evaluation-panels/image-native-chroma-consistency-freeze-2026-11-18.json.gz) ·
[Fresh intake](evaluation-panels/image-native-chroma-consistency-scene-intake-2026-11-18.json.gz) ·
[Held scene-text](evaluation-panels/image-native-chroma-consistency-scene-2026-11-18.json.gz) ·
[Held camera edits and ordinary versions](evaluation-panels/image-native-chroma-consistency-camera-2026-11-18.json.gz) ·
[Held conventional camera fills](evaluation-panels/image-native-chroma-consistency-camera-fill-2026-11-18.json.gz)

## Replay

Use the pinned scene-text inventory and the five prior intake reservations
through offset 96. Append each `cases` and `dispositions` list under their
common `revision`, as documented in the [previous JPEG trial](image-multiple-edit-replay-2026-11-18.md#replay-locally).
Use the prepared offset-96 source deliveries for calibration. Camera
replay uses the two prepared transfer banks named `fresh` and `fresh-ordinary`
in the [camera preparation instructions](image-masked-reconstruction-2026-11-17.md#reproduce).
Generate the exact comparator ledgers with `scripts.image_region_rank_reference`
and `--repeat-jpeg90` for each prepared manifest. The decoded media and
truth masks stay on the replay machine. Use unused output paths.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=backend/.venv/bin/python
module=scripts.image_native_chroma_consistency
"$runner" -B -m scripts.image_text_erase_localization_inputs prepare "$index" "$source" \
  --start 120 --previous-intake "$prior_intakes"
"$runner" -B -m "$module" freeze "$calibration" "$calibration_baseline" "$source/deliveries" \
  "$new_freeze" --transfer "$camera" --transfer "$camera_fills"
"$runner" -B -m "$module" held "$new_freeze" "$source/deliveries" "$scene_baseline" "$scene_output"
"$runner" -B -m "$module" held "$new_freeze" "$camera" "$camera_baseline" "$camera_output"
"$runner" -B -m "$module" held "$new_freeze" "$camera_fills" "$camera_fills_baseline" "$fills_output"
```

The public result recount needs only the Python standard library.

```sh
export PYTHONPATH=backend
base=docs/evaluation-panels/image-native-chroma-consistency
for bank in scene camera camera-fill; do
  python3 -B -S -m scripts.image_native_chroma_consistency review \
    "$base-freeze-2026-11-18.json.gz" "$base-$bank-2026-11-18.json.gz" || exit
done
```

# Native-resolution tiles for small JPEG edits

The earlier PSCC-Net evaluation passed the uploaded RGB image to its
backbone at native resolution. The published localization head then
interpolates its four feature maps to the fixed 256-pixel training
geometry. A small alteration can lose texture or spatial detail at that
stage. This test compares the same MIT-licensed checkpoint on native
256 × 256 crops and on the whole image. Crop context also changes, so
the comparison tests the effect of tiling as a whole. [PSCC-Net paper](https://arxiv.org/abs/2103.10596)
· [Pinned MIT code and checkpoints](https://github.com/proteus1991/PSCC-Net/tree/53e5ff77d8dc5feddda060cd085f9b765761f816)
· [Earlier PSCC comparison](image-jpeg-pscc-localization-2026-11-17.md)
· [Recent work on inference resolution](https://arxiv.org/html/2610.21780v1#S4.SS4)

The tile grid starts at the first pixel and ends at the last pixel on
each axis, with steps no larger than 192 pixels. Overlaps blend their
finest trained maps with a 24-pixel linear taper. A second tiled reading
uses only pixels at least 24 pixels inside each of two separate crops. At each
such pixel, it caps the blended score at the second-strongest agreeing
tile score. The whole-image and both tiling readings use the same native
8-pixel-cell proposal algorithm. A useful region requires at least half
the exact pre-compression changed pixels and a pixel intersection over
union of at least 0.30.

The probe uses quality 75. Calibration fixes the second-highest per-source
ordinary score for each reading, considering controls the existing JPEG
checks left clear. Eight earlier scene-text sources supply calibration.
The frozen choice between tiled readings favors more added useful regions
and uses fewer ordinary and misplaced highlights to break ties. The held
manifest hashes and exact source identities are bound at calibration.
Same-site audits compare the mean response at an edit's proposed location
with the greatest mean at that location in its matching unchanged, blur,
tone, and copy deliveries. Scene text also supplies a same-site conventional
fill. Camera conventional fills are measured as an additional control bank.

The fresh scene-text panel takes the next 24 source sets from the pinned
publisher inventory and retains ten after the earlier qualification checks.
Their source hashes and normalized pixel hashes are distinct from the prior
45 qualified sources. The closest cross-bank 512-bit thumbnail-gradient
fingerprints differed by 203 bits. The published Nano Banana 2 background
is staged inside its marked region and delivered with sharp and blended
boundaries. Source picture origins can be unknown. The separate camera
transfer uses six retained BG-REAL photos with LaMa and MI-GAN removals.
Each declared role uses the delivered JPEG and exact pre-compression edit
mask. [Scene-text publisher and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [BG-REAL photo and editor terms](image-jpeg-frequency-bands-2026-11-17.md#held-photographs-and-test-controls)

## Local results

The frozen points were 0.04774660 for the whole image, 0.64734735 for
overlap blending, and 0.56866508 for two-view agreement. The preselected
reading was overlap blending. The higher ordinary-control points for
both tiled readings show how much the change in input geometry also
raised responses on ordinary content. None of the 16 calibration edits
had a useful region even before applying these points.

| Held panel | Useful locations, existing → with any reading | Ordinary flags, existing → whole PSCC / overlap / two-view | Misplaced locations, whole PSCC / overlap / two-view |
| --- | ---: | ---: | ---: |
| Fresh scene-text edits | 0 → 0 of 20 | 7 → 8 / 11 / 19 of 50 | 0 / 2 / 6 |
| Retained camera edits | 2 → 2 of 24 | 10 → 12 / 10 / 33 of 60 | 4 / 0 / 8 |

The strongest region missed the overlap rule on all 44 held edits in
all three readings before any cutoff. Feeding the truth masks through
the same proposal geometry recovered 18 of 20 fresh scene-text regions
and all 24 camera regions. Tile blending therefore had enough region
resolution for almost all these edits. The two-view restriction gave
sufficient coverage for half the changed pixels on only 22 of 44 edits.
Where the preselected overlap reading flagged an edited file, both
locations had an equal or stronger response in an ordinary version at
the same position. The two-view reading showed this on 11 of its 14
flagged edits. The results leave both native tiled readings offline.

The app's existing aligned JPEG and regional checks were replayed on the
same exact files. Runtime verification checked that a known test map
returned from overlapping tiles preserves its native coordinates on
four raster sizes. The three runs processed 210 delivered JPEGs and
2,576 crops. A separate audit verified the JPEGs, truth masks, coordinate
bounds, and repeat-view coverage for all 210 deliveries. The ledgers
contain measurements and source digests. The individual media and masks
stay local.

[Frozen thresholds and calibration](evaluation-panels/image-pscc-native-tiles-freeze-2026-11-18.json.gz)
· [Fresh scene-text result](evaluation-panels/image-pscc-native-tiles-scene-2026-11-18.json.gz)
· [Fresh source qualification](evaluation-panels/image-pscc-native-tiles-scene-intake-2026-11-18.json.gz)
· [Retained camera result](evaluation-panels/image-pscc-native-tiles-camera-2026-11-18.json.gz)

## Reproduce

Use the pinned PSCC source and checkpoint files and the retained JPEG
banks from the earlier comparisons. The new scene-text source run uses
the same frozen publisher inventory with `--start 72`. Combine the
previous three intake files in chronological order, appending their
`cases` and `dispositions` arrays and keeping their shared `revision`.
[PSCC files and camera inputs](image-jpeg-pscc-localization-2026-11-17.md#reproduce)
· [Prior scene-text intakes and preparation](image-safe-dense-transfer-2026-11-18.md#reproduce)

From `backend`:

```sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=.venv/bin/python
module=scripts.image_pscc_native_tiles
python3 - "$first_intake" "$second_intake" "$third_intake" "$prior_intakes" <<'PY'
import json
from pathlib import Path
import sys
*sources, output = map(Path, sys.argv[1:])
files = [json.loads(path.read_bytes()) for path in sources]
assert len({file['revision'] for file in files}) == 1
combined = {'revision': files[0]['revision'],
            **{key: [row for file in files for row in file[key]] for key in ('cases', 'dispositions')}}
output.write_text(json.dumps(combined, sort_keys=True))
PY
"$runner" -B -m scripts.image_text_erase_localization_inputs prepare "$index" "$new_scene" \
  --start 72 --previous-intake "$prior_intakes"
"$runner" -B -m scripts.image_region_rank_reference "$new_scene/deliveries" "$scene_baseline" --repeat-jpeg90
"$runner" -B -m "$module" freeze "$assets" "$development" "$development_baseline" "$freeze" \
  --device mps --held "$new_scene/deliveries" --held "$camera" --held "$camera_fills"
"$runner" -B -m "$module" validate "$assets" "$freeze" "$scene_result" \
  --device mps --banks "$new_scene/deliveries" "$scene_baseline"
"$runner" -B -m "$module" validate "$assets" "$freeze" "$camera_result" \
  --device mps --camera --banks "$camera" "$camera_baseline" --banks "$camera_fills" "$camera_fills_baseline"
python3 -B -S -m "$module" review "$freeze" "$scene_result"
python3 -B -S -m "$module" review "$freeze" "$camera_result"
```

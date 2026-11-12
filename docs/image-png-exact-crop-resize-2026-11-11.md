# Exact PNG evidence after a crop and resize

Pixel-exact checks rejected all 188 incompatible inputs in the first run, including the fractional shift accepted by the earlier correlation check. They localized 12 of 24 feathered edits and 11 of 24 sharp edits in the two reused panels. The earlier panel yielded five exact untouched crops, one below its frozen coverage requirement. A separate prospective test confirmed the same method on 12 additional source records. [Frozen results and readback](evaluation-panels/image-png-exact-crop-resize-2026-11-11.json)

## Frozen comparison

The existing SIFT and bounded correlation search proposes up to eight integer crop rectangles. For each proposal, resize the actual RGB crop using Pillow bicubic, bilinear, or Lanczos, or OpenCV area interpolation. At least 65% of the submitted pixels must match the projection exactly in all three channels. A 20 × 20 pixel tile is textured when at least 12% of its horizontal and vertical neighbor pairs differ by six or more values in a channel. At least 12 of 25 fixed tiles must be textured and match exactly throughout, spanning at least four rows, four columns, and all quadrants. The bounding rectangle of all differences may cover at most 40% of the submitted grid. Multiple supported geometries or distinct supported projections leave the comparison unassessed.

The submitted RGB difference mask uses eight-connected components with a fixed minimum area. A region maps to the earlier native crop using the outward-rounded integer edges of the projection. The comparison identifies differences between the supplied files. The cause and provenance of either file require separate evidence.

The first run used all 320 exact PNG deliveries from the published crop-resize experiment. The fit-free candidate used the same 8 development and two 12-photo confirmation groups as the earlier experiment. These groups were exposed to the project before this run. The separate 12-photo Commons panel used the same source-size and scale limits, all four kernels, and a fixed local blur and contrast change per source. It excluded the 12 recorded Commons assets used to set the multidepth image check's points. The qualified Commons sources have recorded CC BY or CC BY-SA terms and prior use in a different experiment. An independent source list does not establish camera provenance or cross-site contributor identity. [Earlier crop and resize study](image-reference-crop-resize-transfer-2026-11-11.md) · [Commons assets and credits](image-affine-native-scale-2026-11-11.md)

## Exact outcomes

A useful location requires at least 0.30 native pixel intersection over union and 0.50 recall. The earlier run's ordinary local blur rows used no truth mask and report highlights only. The Commons rows use the exact authored change masks.

| Panel | Useful sharp | Useful feathered | Exact untouched | Local blur | Local contrast | Wrong or incompatible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exposed development | 3/8 | 3/8 | 4/8 | 3/8 highlights | — | 0/32 |
| Exposed earlier | 5/12 | 5/12 | 5/12 | 5/12 highlights | — | 0/48 |
| Exposed later | 6/12 | 7/12 | 8/12 | 7/12 highlights | — | 0/48 |
| Separate Commons | — | — | 10/12 | 9/12 useful | 9/12 useful | 0/60 |

The existing format comparison produced zero useful locations on those sharp and feathered deliveries. The experimental correlation-based PNG candidate localized 15 feathered edits and accepted one incompatible fractional shift. The exact method leaves that case unassessed. It highlights no untouched crops, no earlier JPEG processing controls, and no whole-image contrast controls. The Commons panel includes quarter-pixel and half-pixel shifts, rotation, stretch, and unrelated photos.

The seven missed earlier untouched crops expose two distinct coverage limits. Five lacked the exact proof's required distribution of textured tiles. The other two had correct rounded SIFT bounds but insufficient texture across both folds of the proposal's existing nine-tile correlation gate. The method kept both coverage guards at their frozen values.

The first run completed 440 views in 71.5 seconds on one CPU thread. The replay rebuilt every authored PNG and truth mask and independently verified all 89 accepted projections, selected difference masks and earlier native bounds. It checks the saved receipts without rerunning crop proposals.

## Prospective source confirmation

The second Commons test selected 12 further photo records by the same fixed digest ordering before measurement. All exact source digests were absent from the previous Commons test and the multidepth calibration roster. The photos had prior project exposure in the affine comparison. The candidate code, transforms, resize kernels, and per-role advancement requirements remained fixed.

The check reproduced all 12 untouched crops exactly and localized all 12 local blur edits and 12 local contrast edits. It matched the authored kernel on all 36 accepted pairs, including three groups for each of the four kernels. It declined all 60 incompatible and all 24 whole-image contrast and prior JPEG controls. Independent readback checked all 120 deliveries and 36 accepted projections. Across both runs, all 125 accepted crops matched the recorded native geometry. The prospective result preserves the original 5/12 coverage miss and measures authored edits on ordinary photos. [Exact results and readback](evaluation-panels/image-png-exact-crop-resize-2026-11-11.json)

The local app uses this evidence check as a fallback after integer alignment and same-framing resizing. An application replay reached the new path for all 23 useful sharp and feathered edits and two untouched crops from the two exposed panels. The fractional shift previously accepted by the correlation check remained unassessed. An authenticated API replay verified one actual edited example and the fractional control, including the edited pair's PDF crop and overlay pixels and offline JSON reconstruction. The comparison carries zero AI confidence weight.

## Reproduce

Use the exact development and held intakes, the completed earlier crop-and-resize run, the retained Commons intake from the affine-native-scale study, and the multidepth study's private freeze. Supply a new output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run; do
  backend/.venv/bin/python -B -m scripts.image_png_exact_crop_resize "$stage" \
    "$development" "$earlier" "$later" "$earlier_crop_run" "$commons" "$multidepth_freeze" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_png_exact_crop_resize \
  "$development" "$earlier" "$later" "$earlier_crop_run" "$commons" "$multidepth_freeze" "$run"
for stage in freeze run audit; do
  backend/.venv/bin/python -B -m scripts.image_png_exact_crop_resize_prospective "$stage" \
    "$run" "$commons" "$multidepth_freeze" "$prospective_run" || exit 1
done
```

# Testing whether intact JPEG texture predicts small removals

A compact predictor learns the fine luminance texture expected in an intact
8 × 8 JPEG cell. One input uses the cell's coarse detail, brightness and
quantization. A second input also sees fine detail in surrounding rings that
exclude the measured cell. The model fits 32 intact JPEGs from 16 source
photos. Eight separate photos provide 224 calibration deliveries. The four
readings compare missing detail and unexpected detail, with and without ring
information. Each setting uses one connected region and allows zero, one or
two additional ordinary calibration photo groups. Choices are frozen by
JPEG quality before the held measurements.

This tests a separate cue after the [boundary study](image-jpeg-boundary-2026-11-17.md).
Recent research reports that localized regions occupying a small share of the
picture are especially difficult. This test uses project code and independently
licensed editors to create controlled removals. [Localized deepfake study](https://arxiv.org/html/2512.16688v1)
· [LaMa terms](https://github.com/advimman/lama/blob/main/LICENSE)
· [MI-GAN weight terms](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS)

## Comparison

The retained panel contains 20 photos previously used in the boundary study.
Seven also appeared in earlier unrelated JPEG grid studies. The second panel
contains 12 newly acquired photos excluded from both preceding input banks and
the fitting and calibration splits. The receipts bind the attributed originals,
byte and pixel identities, and the held deliveries. Each photo supplies
removals from two editors, with sharp and inward-feathered versions. Paired
ordinary cases keep the original or use local blur, tone, copying, or traditional
inpainting. Counts are deliveries and some ordinary files recur across editors.
A correct location covers at least half the changed pixels and reaches an
intersection over union of 0.30. [Photo sources and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

| Panel | Quality | Correct areas, current → combined | Ordinary deliveries flagged, current → combined | Misplaced candidate regions on edited files |
| --- | --- | --- | --- | --- |
| Retained | 75 | 0 → 0 of 80 | 10 → 20 of 200 | 4 |
| Retained | 90 | 0 → 0 of 80 | 6 → 6 of 200 | 0 |
| New | 75 | 1 → 1 of 48 | 15 → 15 of 120 | 0 |
| New | 90 | 1 → 1 of 48 | 6 → 6 of 120 | 0 |

The candidate's top-ranked region overlapped none of the 256 edits at the
required extent, even before any cutoff, for all four readings. On the newly
acquired photos, the preferred measurement at the known edit position exceeded
both the intact and blurred versions on 6 of 48 quality-75 edits and 8 of 48
quality-90 edits. Native cell geometry could cover enough of every edited
area if arbitrary groups of cells could be selected. The predicted texture
rarely ranked the edit above its ordinary alternatives even with the correct
location supplied. The app continues using its current JPEG checks.
[Frozen choices](evaluation-panels/image-jpeg-texture-freeze-2026-11-17.json.gz)
· [Retained](evaluation-panels/image-jpeg-texture-retained-2026-11-17.json.gz)
· [Fresh](evaluation-panels/image-jpeg-texture-fresh-2026-11-17.json.gz)

## Reproduce

The standalone reader recounts the two comparisons without fitting or loading
the detector. Run from `backend`.

```sh
panels=../docs/evaluation-panels
stem=image-jpeg-texture
freeze="$panels/$stem-freeze-2026-11-17.json.gz"
for panel in retained fresh; do
  python3 -B -S -m scripts.image_jpeg_texture_prediction_panel review \
    "$freeze" "$panels/$stem-$panel-2026-11-17.json.gz" || exit
done
```

For exact pixel comparison, restore the partial removal banks using the
[frequency](image-jpeg-frequency-bands-2026-11-17.md#replay-and-independent-comparison)
and [boundary](image-jpeg-boundary-2026-11-17.md#replay) replays. In the commands
below `training` is the 24-photo partial bank. `training_fills` contains the
traditional controls from that bank for the eight source fingerprints in
`selection_sources` in the freeze. Keep their original delivery bytes. The
two training reference files come from `scripts.image_region_rank_reference`
with `--repeat-jpeg90`. The `images` manifest entries select the relevant
deliveries and the `groups` entries select their source groups.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
runner=.venv/bin/python
comparison=scripts.image_jpeg_texture_prediction_panel
"$runner" -B -m "$comparison" fit "$training" "$training_fills" \
  "$training_reference" "$training_fills_reference" "$run/fit"
"$runner" -B -m scripts.image_jpeg_band_sources acquire \
  "$panels/$stem-reservation-2026-11-17.json.gz" "$training/manifest.json" "$run/intake" \
  --previous-intake "$training_intake/intake.json" --previous-intake "$retained_intake/intake.json"
PYTHONPATH="$ort:." "$runner" -B -m scripts.image_jpeg_band_sources prepare \
  "$run/intake" "$editors" "$run/fresh" --partial
"$runner" -B -m scripts.image_jpeg_boundary_controls "$run/intake" "$run/fills" --partial
for bank in fresh fills; do
  "$runner" -B -m scripts.image_region_rank_reference \
    "$run/$bank" "$run/$bank-reference.jsonl" --repeat-jpeg90 || exit
done
"$runner" -B -m "$comparison" hold "$run/fresh" "$run/fills" \
  "$run/fresh-reference.jsonl" "$run/fit" "$run/fresh.json.gz" \
  --baseline-fills "$run/fills-reference.jsonl"
"$runner" -B -m scripts.export_image_jpeg_texture_replay "$run/fresh" "$run/fills" \
  "$run/intake" "$run/fit" "$run/fresh.json.gz" "$run/review"
```

The local exporter provides a case manifest, input identities, truth masks and
prediction areas so another detector can use exactly the same JPEGs. The
separate recount checked 896 truth cases, 1,792 file bindings and 14 reported
candidate regions. The public receipts contain numeric readouts and source
commitments. Photos, individual masks and the research checkpoint stay local.
[Reserved plan](evaluation-panels/image-jpeg-texture-reservation-2026-11-17.json.gz)
· [Fresh intake](evaluation-panels/image-jpeg-texture-intake-2026-11-17.json.gz)
· [Independent pixel arithmetic](evaluation-panels/image-jpeg-texture-audit-2026-11-17.json)

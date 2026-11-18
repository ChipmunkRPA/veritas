# Localizing small JPEG removals with a pretrained model

The study compares the official PSCC-Net pixel map, connected regions from
its finest map, and connected regions supported across all four scales. The
published backbone and localization head come from the pinned MIT repository.
The replay checks the source files and weights by SHA-256 and maps the output
back to the submitted JPEG pixel grid. It uses the pretrained weights without
fitting them on the study photos.
[Published method](https://ieeexplore.ieee.org/abstract/document/9819903/)
· [Pinned source and weights](https://github.com/proteus1991/PSCC-Net/tree/53e5ff77d8dc5feddda060cd085f9b765761f816)
· [MIT license](https://github.com/proteus1991/PSCC-Net/blob/53e5ff77d8dc5feddda060cd085f9b765761f816/LICENSE)

## Local comparison

Twenty-four BG-REAL photos provided 608 calibration deliveries. Both ranked
readings missed the required overlap on all 192 staged removals before
applying a score cutoff. Binarizing the finest map also missed all 192 at
each of nine probed cutoffs. The frozen choice at both JPEG qualities was
the agreement map at the point allowing no extra ordinary calibration source
groups. The held comparisons add its findings to the current JPEG checks.
Correct locations have at least 0.30 pixel intersection over union and at
least 50% edited-pixel recall.

Twenty retained photos had appeared in earlier JPEG studies. Twelve newly
acquired photos were screened for overlaps with the calibration, retained and
earlier JPEG texture panels. Each supplies LaMa and MI-GAN removals with hard
and blended edges. Controls preserve the same source and include unchanged
JPEGs, local blur, local tone, copies and traditional inpainting. Each
quality reuses the same source photos. Some controls repeat across editors.
[Photo sources and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [LaMa license](https://github.com/advimman/lama/blob/main/LICENSE)
· [MI-GAN weight terms](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS)

| Held panel | JPEG quality | Useful locations, current → combined | Ordinary deliveries flagged, current → combined | Misplaced candidate regions on edited files |
| --- | ---: | ---: | ---: | ---: |
| Retained photos | 75 | 0 → 0 of 80 | 10 → 10 of 200 | 0 |
| Retained photos | 90 | 0 → 0 of 80 | 6 → 27 of 200 | 8 |
| Newly acquired photos | 75 | 3 → 4 of 48 | 10 → 21 of 120 | 4 |
| Newly acquired photos | 90 | 0 → 3 of 48 | 0 → 13 of 120 | 1 |

One source photo accounts for all four additional correct locations. For
each of its three quality-90 gains, the same region also passed the overlap
rule when the model saw the unchanged JPEG and at least one locally blurred
control. The combined measurement added 45 ordinary flags across these
two panels.
[Calibration ledger](evaluation-panels/image-jpeg-pscc-calibration-2026-11-17.json.gz)
· [Retained ledger](evaluation-panels/image-jpeg-pscc-retained-2026-11-17.json.gz)
· [Newly acquired ledger](evaluation-panels/image-jpeg-pscc-fresh-2026-11-17.json.gz)
· [Same-position controls](evaluation-panels/image-jpeg-pscc-gain-context-2026-11-17.json)

## Transfer to scene-text removal

The same frozen measurements ran on 14 pairs from a publisher's scene-text
removal dataset. The authors used Nano Banana 2 for the replacement
backgrounds. From 24 pairs picked by source hash, ten had annotated areas
outside the predefined size range. Each retained case replays the replacement
background inside the annotated area, with sharp and feathered versions.
Unchanged and ordinary-edit deliveries use the same supplied reference. The
reference images have mixed or undetermined origin. The result measures
localization of the staged replacement and does not estimate false alarms
on verified camera originals. [Dataset and attribution](https://huggingface.co/datasets/QinmingZhou/TextEraseBench/blob/cf94f7eaffc654146cba02fe2f704cc1a6e4998d/README.md)
· [Dataset construction](https://arxiv.org/html/2606.28094v1)

Neither the existing JPEG checks nor any candidate measurement meets the
required overlap on the 56 staged edits. The candidate adds one ordinary
flag at quality 75 (18 → 19 of 70) and two at quality 90 (13 → 15 of 70).
Chipmunk Verify keeps the current JPEG checks. This result concerns small,
locally edited JPEGs in the three tested families.
[Transfer ledger](evaluation-panels/image-jpeg-pscc-provider-2026-11-17.json.gz)
· [Intake and exclusions](evaluation-panels/image-jpeg-pscc-provider-intake-2026-11-17.json.gz)

Supplying the matching earlier JPEG to the app's two-copy comparison
located 22 of 28 quality-75 edits and 23 of 28 quality-90 edits. The replay
used the same submitted bytes and pre-compression pixel masks as the
single-upload comparison. All 28 identical pairs had empty masks. The
comparison also displayed some differences from traditional retouching,
including every local blur. It provides pixel evidence that the two copies
differ and assigns zero weight to AI origin. It needs an earlier copy with
matching content. [Earlier-copy replay](evaluation-panels/image-jpeg-pscc-provider-paired-2026-11-17.json.gz)

## Reproduce

The numeric replay requires standard-library Python. From `backend`:

```sh
panels=../docs/evaluation-panels
stem=image-jpeg-pscc
freeze="$panels/$stem-calibration-2026-11-17.json.gz"
for bank in retained fresh provider; do
  python3 -B -S -m scripts.image_pscc_localization_panel review \
    "$panels/$stem-$bank-2026-11-17.json.gz" --frozen "$freeze" || exit
done
```

Download the nine files listed in `app.bench.image_pscc_localization.SOURCE_FILES`
from the pinned publisher revision, retaining their relative paths under
`$assets`. Restore the calibration and retained JPEGs from the
[partial removal and control recipes](image-jpeg-texture-prediction-2026-11-17.md#reproduce).
For the new photos, use the same source acquisition recipe with the new
[reservation](evaluation-panels/image-jpeg-pscc-reservation-2026-11-17.json.gz)
and include the prior JPEG texture intake as an additional `--previous-intake`.
The [new source intake](evaluation-panels/image-jpeg-pscc-intake-2026-11-17.json.gz)
records the acquired photos and their digests. Measure the current JPEG checks
with `scripts.image_region_rank_reference --repeat-jpeg90`, supplying the
delivery directory and the reference output path.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
runner=.venv/bin/python
comparison=scripts.image_pscc_localization_panel
"$runner" -B -m "$comparison" freeze "$assets" "$run/calibration" --device mps \
  --input "$training" "$training_reference" --input "$training_fills" "$training_fills_reference"
"$runner" -B -m "$comparison" hold "$assets" "$run/held-fresh" --device mps \
  --frozen "$run/calibration/result.json.gz" --input "$run/fresh" "$run/fresh-reference.jsonl" \
  --input "$run/fills" "$run/fills-reference.jsonl"
"$runner" -B -m scripts.export_image_pscc_review "$run/intake" \
  "$run/calibration/result.json.gz" "$run/held-fresh/result.json.gz" "$run/audit-fresh" \
  --bank "$run/fresh" --bank "$run/fills"
```

Repeat `hold` and the exporter with the retained deliveries and their intake.
For the scene-text transfer, decompress the
[publisher inventory](evaluation-panels/image-jpeg-pscc-provider-index-2026-11-17.json.gz)
and pass it to `scripts.image_text_erase_localization_inputs` with a new
output directory through its `prepare` command. Its `deliveries` folder is
the only `--input` bank for this `hold` and the only `--bank` for the exporter.
The `intake` folder under the same directory contains the parent input
bindings. Recreate the earlier-copy receipt with `compare-pairs`, supplying
the same `deliveries` folder and a new `.json.gz` output path.
The published freeze binds the MPS runtime. A CPU comparison can create a
separate calibration and matching held receipts with `--device cpu`. The
exporter verifies the original and delivery bytes and recounts each stored
candidate mask against the changed pixels. It provides exact inputs and a
common truth manifest for comparing a different detector. The local exports
verified 1,092 cases, 2,184 file bindings and 3,276 candidate mask count pairs.
Original media, individual masks and weights stay local.
[Recount receipts](evaluation-panels/image-jpeg-pscc-audit-2026-11-17.json)

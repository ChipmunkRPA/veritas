# Separate frequency checks for object removals in JPEGs

The frozen check found zero correctly placed regions for 384 newly staged
JPEG edits. It marked some ordinary images and selected misplaced regions
on some edited files. The frequency model remains a research comparison.

## Method

Separate three ranges of native luminance JPEG coefficients. For each
range, measure the dequantized magnitudes, signed agreement with nearby
blocks, and decoded color and texture at the same 8 × 8 positions. Fit
three small convolutional branches independently. Compare their individual
maps and an average of the three. The branch chosen for each quality uses
the native grid and retains a single connected region. The tested method
adapts the frequency separation and JPEG-quality augmentation described
in recent inpainting research. It uses independently implemented code
and does its own fitting. [Multi-frequency inpainting study](https://pubmed.ncbi.nlm.nih.gov/42335053/)

The development set divides 60 previously acquired, attributed BG-REAL
source photos into 43 fitting and 17 calibration groups. These cases use
LaMa object removals and Stable Diffusion 1.5 object insertions. They
include unchanged photos and matched local blur, tone, and copy controls.
Each condition has JPEG75 and JPEG90 saves. The calibration compares up
to two added ordinary source groups per JPEG quality and favors useful
feathered locations. Both qualities select the coarse branch at the
point with zero added calibration groups. The selected point added zero
correct locations on calibration. Two local fits reproduced the exact
weights, losses and calibration outcomes. [BG-REAL terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)
· [Fitting and calibration receipts](evaluation-panels/image-jpeg-bands-calibration-2026-11-17.json.gz)

## Held photographs and test controls

Use 24 additional BG-REAL originals with per-photo CC BY 2.0 attribution.
Source identifiers, photographers, file hashes, decoded pixels and close
duplicates are checked against the recorded prior source sets. The
published subject masks select the removal area. The full-subject panel
changes 11.9–18.5% of the pixels. The partial-subject panel changes
3.2–3.5%. It selects a compact portion on one side of each subject before
running either editor. Exact masks contain the pixels that changed before
the last JPEG save. Both editors preserve the original pixels outside the
planned support. [Reserved sources](evaluation-panels/image-jpeg-bands-reservation-2026-11-17.json.gz)
· [Validated source intake](evaluation-panels/image-jpeg-bands-source-intake-2026-11-17.json.gz)

Both panels use the same 24 photos, two checksum-pinned removal engines,
two deliveries of each edit, and four controls per engine and quality.
The two edit deliveries use the direct result and an inward feathered
result. Each panel contains 192 edited and 384 ordinary deliveries.
The masks score the selected edit region using a minimum intersection
over union of 0.30 and at least 50% recall of the changed pixels. The
1,152 deliveries represent 24 independent source groups shared by the
two editors and both sizes. [LaMa source and license](https://github.com/advimman/lama)
· [Pinned LaMa artifact](https://github.com/Sanster/models/releases/tag/add_big_lama)
· [MI-GAN weight license](https://github.com/Picsart-AI-Research/MI-GAN/blob/main/LICENSE-WEIGHTS)
· [Published MI-GAN artifact](https://huggingface.co/andraniksargsyan/migan/blob/1538c13/migan_pipeline_v2.onnx)

## Results

| Held panel | App checks locating edits | With frequency check | Ordinary flags, app → combined | Misplaced additions on edits |
| --- | ---: | ---: | ---: | ---: |
| Full removal, JPEG75 | 0/96 | 0/96 | 20 → 34 of 192 | 13 |
| Full removal, JPEG90 | 0/96 | 0/96 | 18 → 32 of 192 | 12 |
| Partial removal, JPEG75 | 0/96 | 0/96 | 41 → 49 of 192 | 4 |
| Partial removal, JPEG90 | 0/96 | 0/96 | 35 → 43 of 192 | 4 |

The comparator combines the existing JPEG compression readout with the
serving regional check. It uses the app's repeat-compression confirmation
for eligible JPEG90 inputs. The compression check assesses all 288
deliveries in each panel-quality combination. The regional check is in
scope for 276. The combined method keeps locations from these existing
checks before using the new response. [Full results](evaluation-panels/image-jpeg-bands-full-2026-11-17.json.gz)
· [Partial results](evaluation-panels/image-jpeg-bands-partial-2026-11-17.json.gz)

Before applying the frozen operating point, the leading region
meets the overlap requirement on six full-subject LaMa inputs at each
quality, eight partial-subject LaMa inputs at each quality, and two
partial-subject MI-GAN inputs at JPEG90. All of those scores fall below
the selected points. Lowering the point by itself would also expose
additional ordinary regions. The held rows preserve the alternatives and
exact mask overlap for each evaluated region.

A separate comparison reuses nine previously studied small LaMa removals
that were held out of fitting and calibration. It adds zero correct
locations and zero ordinary flags on 108 JPEG deliveries. The highest
region overlaps three edits at JPEG75 and four at JPEG90 before applying
the frozen point. This replay uses its saved baseline measurements.
[Retained small-removal results](evaluation-panels/image-jpeg-bands-retained-2026-11-17.json.gz)

## Replay and independent comparison

Use the credited source roster and exact publisher metadata. The earlier
local archives supply the development inputs and comparator journals.
[BG-REAL source plan](image-bg-real-roster-v218-method.md)
· [Region-ranking archive](image-region-ranking-2026-11-16.md#replay-and-compare-another-check)
· [Cross-quality archive](image-cross-quality-2026-11-16.md#exact-input-and-overlap-replay)
Place the two pinned editor files in the model folder as `big-lama.pt`
and `migan.onnx`. Reproduction used CPU, TorchScript, and the local
ONNX Runtime 1.22.1 wheel. The uncompressed photo, full rasters, and
fitted weights stay in the local evaluation directory.

```sh
export PYTHONPATH=backend HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=1
runner=backend/.venv/bin/python
panels=docs/evaluation-panels
source_script=scripts.image_jpeg_band_sources
comparison=scripts.image_jpeg_band_transfer
plan="$panels/image-jpeg-bands-reservation-2026-11-17.json.gz"
"$runner" -B -m "$source_script" acquire "$plan" "$retained/deliveries/manifest.json" "$run/intake"
PYTHONPATH="$ort:backend" "$runner" -B -m "$source_script" prepare "$run/intake" "$editors" "$run/holdout"
PYTHONPATH="$ort:backend" "$runner" -B -m "$source_script" prepare "$run/intake" "$editors" "$run/partial" --partial
"$runner" -B -m "$comparison" freeze "$run/fit" \
  --inputs "$retained/deliveries" "$retained/reference.jsonl" \
  --inputs "$cross/confirmation/deliveries" "$cross/fresh-reference.jsonl"
for panel in holdout partial; do
  "$runner" -B -m scripts.image_region_rank_reference "$run/$panel" "$run/$panel.jsonl" --repeat-jpeg90
  "$runner" -B -m "$comparison" assess "$run/fit" "$run/$panel" "$run/$panel.jsonl" "$run/$panel.json.gz"
done
"$runner" -B -m scripts.export_image_jpeg_band_review "$run" "$run/fit" "$run/review" \
  --panel holdout "$run/holdout.json.gz" --panel partial "$run/partial.json.gz"
```

The exporter builds the public `independent_review` manifest and exact
native PBM masks in the local evaluation directory. A different detector
can report regions on those same input bytes. Files held by the reviewer
include the original photo and its public attribution. The published
summary carries numeric measurements and digests rather than media.
[Independent evaluation format](independent-testing.md#count-exact-locations-on-your-own-files)
· [Verified input and overlap counts](evaluation-panels/image-jpeg-bands-independent-2026-11-17.json.gz)

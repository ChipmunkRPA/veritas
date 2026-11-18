# Masked reconstruction for small JPEG removals

Reconstruction-based forgery research uses predictions of missing pixels as
one input to a trained localization model. This trial tests that cue by
itself and in a small paired readout. It uses Microsoft's Apache-2.0
Swin reconstruction checkpoint. The released safetensors and three
accompanying files are pinned by SHA-256. [Reconstruction for image forensics](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Detecting_AI-Generated_Forgeries_via_Iterative_Manifold_Deviation_Amplification_CVPR_2026_paper.pdf)
· [SimMIM paper](https://arxiv.org/abs/2111.09886)
· [Pinned Microsoft files and license](https://huggingface.co/microsoft/swin-base-simmim-window6-192/tree/6c7c8dea55b6bc1fb8e20c21dae660cc2a45cada)

## Method

Each submitted JPEG is proportionally reduced to a 192-pixel long side
and padded with neutral values. Four fixed 32-pixel masking patterns hide
each location twice in different surroundings. The network predicts the
hidden pixels. The readout compares the observed error, disagreement
between the two predictions, and visible detail, also relative to the
surrounding area. Measurements map back to 8-pixel source cells and the
selected region uses the submitted file's exact pixel coordinates. The
input and output use the normalization from the original SimMIM training
code. A local replay changed all values in one hidden block and got
bit-identical predictions for that mask.
[Original normalization](https://github.com/microsoft/SimMIM/blob/main/data/data_simmim.py)

Two fixed readings rank stronger and weaker reconstruction errors. The
two paired linear readouts use local intensity, color and detail, with
the second also seeing the reconstruction measurements. They learn from
12,288 paired examples each, with positives inside the known changed
area and negatives from the same coordinates in ordinary deliveries or
from unchanged regions of the edited file. Source hashes assign 16 photos
to fitting and eight other photos to selection. Each JPEG quality uses
its own frozen point. All four readouts retain their comparisons at
points allowing zero, one or two extra ordinary calibration photo groups.

The first run lost its input volume midway through calibration. It
produced no operating point. The completed run uses 24 local BG-REAL
photos from the prior texture and multiscale tests. [Texture input roster](image-jpeg-texture-prediction-2026-11-17.md#reproduce)
· [Multiscale input roster](image-jpeg-pscc-localization-2026-11-17.md#reproduce)

## Held inputs

Held rows use the exact same JPEGs and masks for both the current JPEG
comparison and the candidate. A correct region covers at least half of the
changed pixels with intersection over union of at least 0.30. The two
qualities and soft and hard edges share source photos.

- Twelve new BG-REAL photos supply staged LaMa and MI-GAN local removals.
  Controls include unchanged JPEGs, blur, tone, copies and conventional
  inpainting. The roster excludes known photo and photographer overlap
  with both development banks.
- Fourteen already measured pairs and 16 newly acquired pairs from
  TextEraseBench provide staged local scene-text removals using the
  published Nano Banana 2 backgrounds. The second intake checks the next
  24 source pairs in the preselected ordering. Eight were outside the
  fixed annotated-area range. Untouched references can have an unknown
  origin. The comparison checks the added region at its exact staged
  coordinates. [Dataset and terms](https://huggingface.co/datasets/QinmingZhou/TextEraseBench)
  · [Dataset construction](https://arxiv.org/html/2606.28094v1#A3.SS1)

## Results

| Held inputs | JPEG quality | Correct locations, current → combined | Ordinary alerts, current → combined |
| --- | ---: | ---: | ---: |
| New BG-REAL removals | 75 | 0 → 0 of 48 | 16 → 16 of 120 |
| New BG-REAL removals | 90 | 4 → 4 of 48 | 10 → 10 of 120 |
| Earlier scene-text removals | 75 | 0 → 0 of 28 | 18 → 18 of 70 |
| Earlier scene-text removals | 90 | 0 → 0 of 28 | 13 → 13 of 70 |
| New scene-text removals | 75 | 0 → 0 of 32 | 10 → 10 of 80 |
| New scene-text removals | 90 | 0 → 0 of 32 | 0 → 0 of 80 |

The frozen selection used the stronger reconstruction error at both
qualities. Every candidate ranked a region on every held file. The
top-ranked region missed the overlap target on all 216 edits for each
of the four readings, before any cutoff. More permissive points added
ordinary alerts without adding correct locations. A separate recount of
the masks found that the fitted reconstruction reading selected at least
half-overlapping regions in 190 of 216 edited and unchanged pairs. The
texture-only reading did so in 197 pairs. For these inputs, the readouts
mostly continued to highlight features already present in the unchanged
image. The candidate remains in evaluation.
[Frozen calibration](evaluation-panels/image-masked-reconstruction-freeze-2026-11-17.json.gz)
· [New photos](evaluation-panels/image-masked-reconstruction-fresh-2026-11-17.json.gz)
· [Earlier scene text](evaluation-panels/image-masked-reconstruction-earlier-text-2026-11-17.json.gz)
· [New scene text](evaluation-panels/image-masked-reconstruction-fresh-text-2026-11-17.json.gz)
· [Exact mask recounts](evaluation-panels/image-masked-reconstruction-audit-2026-11-17.json)

## Reproduce

The operating points and exact overlap counts can be replayed with
standard-library Python. Run from `backend`.

```sh
panels=../docs/evaluation-panels
stem=image-masked-reconstruction
frozen="$panels/$stem-freeze-2026-11-17.json.gz"
for split in fresh earlier-text fresh-text; do
  python3 -B -S -m scripts.image_masked_reconstruction_panel review \
    "$panels/$stem-$split-2026-11-17.json.gz" --frozen "$frozen" || exit
done
```

The three BG-REAL photo banks can be restored with the published intake
records. The command verifies the source bytes and subject masks against
the completed intake. It retains the original intake bytes. No private
archive or prior candidate selection is needed. We verified all 24
photo and mask files from the fresh intake against the local copies.
[Fresh photo reservation](evaluation-panels/image-masked-reconstruction-reserve-2026-11-17.json.gz)
· [Fresh photo intake](evaluation-panels/image-masked-reconstruction-intake-2026-11-17.json.gz)

```sh
runner=.venv/bin/python
mkdir -p "$run/texture" "$run/pscc" "$run/fresh"
"$runner" -B -m scripts.restore_jpeg_source_intake \
  "$panels/image-jpeg-texture-intake-2026-11-17.json.gz" "$run/texture/intake"
"$runner" -B -m scripts.restore_jpeg_source_intake \
  "$panels/image-jpeg-pscc-intake-2026-11-17.json.gz" "$run/pscc/intake"
"$runner" -B -m scripts.restore_jpeg_source_intake \
  "$panels/$stem-intake-2026-11-17.json.gz" "$run/fresh/intake"
```

For each photo bank, use `scripts.image_jpeg_band_sources prepare --partial`
with the two pinned editors, `scripts.image_jpeg_boundary_controls --partial`,
and `scripts.image_region_rank_reference --repeat-jpeg90` on each delivery
directory. The [earlier removal replay](image-jpeg-pscc-localization-2026-11-17.md#reproduce)
provides the command arguments. Download the four reconstruction assets in
`app.bench.image_masked_reconstruction.ASSETS` into `$assets`. The replay
checks the asset and parameter bindings before forwarding any JPEG.

Rebuild the scene-text banks with the same
[publisher inventory](evaluation-panels/image-jpeg-pscc-provider-index-2026-11-17.json.gz).
Decompress it to `index.json`. Run the earlier preparation command without
additional flags to reproduce the previous intake. Supply that intake to
the new preparation command, and measure the current JPEG checks in the
new `deliveries` directory. [New scene-text intake and exclusions](evaluation-panels/image-masked-reconstruction-text-intake-2026-11-17.json.gz)

```sh
"$runner" -B -m scripts.image_text_erase_localization_inputs prepare \
  "$run/index.json" "$run/fresh-text" --start 24 --previous-intake "$earlier_intake/intake.json"
"$runner" -B -m scripts.image_region_rank_reference "$run/fresh-text/deliveries" \
  "$run/fresh-text/reference.jsonl" --repeat-jpeg90
```

Use `freeze` with four `--input` pairs for the two development JPEG banks
and their two conventional-fill banks. Use `hold` with the saved freeze for
each held bank. The published run uses `--device mps`, and a CPU run can
fit a new freeze on that runtime. For example, to measure the newest
scene-text cases and export their exact locations:

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$runner" -B -m scripts.image_masked_reconstruction_panel hold "$assets" "$run/held-text" \
  --device mps --frozen "$frozen" \
  --input "$run/fresh-text/deliveries" "$run/fresh-text/reference.jsonl"
"$runner" -B -m scripts.export_image_pscc_review "$run/fresh-text/intake" "$frozen" \
  "$run/held-text/result.json.gz" "$run/audit-text" --reconstruction \
  --bank "$run/fresh-text/deliveries"
```

The exporter independently recounts native truth and prediction masks and
binds the parent and submitted files for comparison with another detector.
The local replays verified 756 deliveries, 1,512 file bindings and 3,024
candidate mask count pairs.
The public receipts keep file hashes and measurements. Images, masks and
the Microsoft checkpoint stay in the local testing directories.

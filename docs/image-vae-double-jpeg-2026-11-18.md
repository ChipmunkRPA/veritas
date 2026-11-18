# Local inpainting with a reconstructed background after JPEG recompression

The July 2026 DinoLizer revision separates pristine pixels, autoencoder
reconstruction and inpainting. Its reported model uses a ViT-B/14 backbone
refined on B-Free, a LoRA adaptation and overlapping 504-pixel crops. The
authors’ inference code averages three logits over overlapping crops.
The repository, the separately hosted checkpoint and the FOSSIL data
share supply no explicit commercial grant on their public pages. B-Free
also limits its released files to informational and nonprofit use. The
comparison below uses an independent lightweight head on the Apache-2.0
Meta DINOv2-small encoder. [Paper](https://arxiv.org/html/2511.20722v2) ·
[Author repository](https://github.com/anonyme610/dinolizer) ·
[Author checkpoint listing](https://nextcloud.univ-lille.fr/index.php/s/HS6CLT73DZpZEyq) ·
[FOSSIL listing](https://nextcloud.univ-lille.fr/index.php/s/FHTcLLwGggWHX6d) ·
[B-Free terms](https://github.com/grip-unina/B-Free/blob/main/README.md#license) ·
[Pinned Meta encoder](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)

## Frozen local comparison

Each BG-REAL group supplies the pristine photo, a publisher-matched
reconstruction and an edit derived from the generator image and exact
change mask. Both edited variants have identical pixels inside the
mask. The first keeps the original background outside the mask, and
the second uses the same photo’s reconstruction outside the mask.
A single JPEG60 encode and a JPEG90 decode followed by JPEG60 deliver
each variant on the same native geometry. Ordinary controls include
the untouched photo, full and local reconstruction, local blur and a
same-photo copy. The source intake used Kandinsky, SDXL and Stable
Diffusion 1.5 images. The intake checked CC BY 2.0 per photo and the
publisher’s CC BY 4.0 contributions. The publisher separately lists
the inpainting model terms. The frozen roster excludes FLUX-derived
images and controls. [BG-REAL license and attribution](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

Use the same 14-pixel feature grid, 224-pixel tiles with a 168-pixel
step, and rectangle search for each arm. Fit binary and three-class
heads once from single-encode data and once using both histories.
During fitting, sample matched coordinates inside and outside the edit.
The three-class head treats autoencoder reconstruction as a separate
label. Calibration uses the second-highest ordinary group maximum for
each head and JPEG history. A useful region requires at least 0.30 IoU
and 0.50 recovery of the exact pre-compression edit pixels.

The three roles contain 16 planned fit, eight calibration and 16 held
groups. One fit group lacks sufficient matched reconstruction support,
so 15 contribute to training. Exact source, subject photo and credited
photographer identifiers are disjoint across roles. The archived
photographs have prior exposure in other project experiments. The
held comparison tests this frozen recipe on separate groups from
training and calibration. It measures known, paired edits from one
publisher and does not establish accuracy on customer uploads.
[Bounded source crops](image-background-patch-transfer-2026-11-11.md) ·
[Stable Diffusion insertion bank](image-bg-real-insertions-2026-11-11.md) ·
[Additional held source crops](image-wavelet-region-transfer-2026-11-11.md)

## Held results

| Check | Correct edit ranges, JPEG60 | Ordinary flags, JPEG60 | Correct edit ranges, JPEG90→60 | Ordinary flags, JPEG90→60 | Misplaced edited files, JPEG90→60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing JPEG luma | 0/32 | 0/80 | 0/32 | 0/80 | 0/32 |
| Single-history binary | 0/32 | 26/80 | 0/32 | 25/80 | 9/32 |
| Single-history three-class | 0/32 | 26/80 | 0/32 | 26/80 | 9/32 |
| Paired-history binary | 0/32 | 22/80 | 0/32 | 23/80 | 8/32 |
| Paired-history three-class | 0/32 | 25/80 | 0/32 | 25/80 | 9/32 |

Each JPEG history contains 16 edited images with the original outside
the edit and 16 with reconstruction outside it. The 80 ordinary
deliveries represent five controls per source photo. On JPEG90→60,
paired-history binary flags ordinary controls in five groups and the
three-class head flags six. Before applying the calibrated point, the
three-class paired-history head places one correct box on the same
Kandinsky edit in both outside contexts. Both boxes have 0.809 IoU
and 87.5% recall. Both scores fall below the ordinary point. The
remaining top regions miss the required geometry. This evaluation
keeps the new heads out of serving.

The portable runner measured 120 fitting, 80 calibration and 224 held
JPEG deliveries on one CPU thread in 276 seconds. Peak resident memory
was about 1.15 GiB. This rerun reproduced the earlier case hashes, fitted
heads, operating points and all recorded scores and regions exactly.

## Replay

The [numeric capsule](evaluation-panels/image-vae-double-jpeg-2026-11-18.json.gz)
contains source and delivery commitments, group aliases, scalar scores,
coordinates, overlap counts and calibration points. It contains no
images, personal names, raw masks or model coefficients.
The standard-library reviewer reconstructs group counts, calibration
points, decisions and overlap arithmetic.

```sh
export PYTHONPATH=backend
receipt=docs/evaluation-panels/image-vae-double-jpeg-2026-11-18.json.gz
python3 -B -S -m scripts.review_image_vae_double_panel verify "$receipt"
```

Native replay additionally reads the exact attributed source files
from the archived BG-REAL intake packs. The portable runner expects
these four directories under `$source_archive`, each including its
`plan.json`, `attempts.jsonl`, `result.json` and `assets/` from the
earlier evaluations.

- `image-background-patch-transfer-2026-11-11/intake`
- `image-bg-real-small-insertions-2026-11-11/trial`
- `image-bg-real-remote-context-2026-11-11/trial`
- `image-wavelet-region-transfer-2026-11-11/holdout`

Use the pinned Meta encoder from the project model cache and the
library versions captured in the capsule. The runner recomputes native
edit/reconstruction geometry and checks the source hashes and exact JPEG
delivery bytes. Set `$out` to an empty local working directory. The source
archives and newly fitted local heads stay outside the numeric capsule.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
runner=backend/.venv/bin/python
module=scripts.image_vae_double_panel
"$runner" -B -m "$module" freeze "$source_archive" "$out/freeze.json"
"$runner" -B -m "$module" fit "$source_archive" "$out/freeze.json" --out "$out/fit.json"
"$runner" -B -m "$module" calibration "$source_archive" "$out/freeze.json" \
  --prior "$out/fit.json" --out "$out/calibration.json"
"$runner" -B -m "$module" held "$source_archive" "$out/freeze.json" \
  --prior "$out/calibration.json" --out "$out/held.json"
python3 -B -S -m scripts.review_image_vae_double_panel verify "$receipt" --native "$out"
```

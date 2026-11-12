# Small learned image residuals with matched processing controls

Two compact single-image patch checks missed all 124 sharp or feathered edits across PNG and JPEG75. Both also highlighted ordinary controls. This comparison gives no basis to use either learned check in customer analysis. [Audited counts and method](evaluation-panels/image-residual-patch-transfer-2026-11-11.json)

## Source separation and model fit

DiffusionPrint motivates learning localized generation traces while accounting for the effect of an autoencoder on unedited pixels. BG-REAL supplies separate matched autoencoder controls and records the original photo, generated image and edit mask. This test trains two small convolutional heads from scratch. It uses no upstream detector code or checkpoint. [DiffusionPrint](https://arxiv.org/abs/2604.12443) · [BG-REAL paper](https://arxiv.org/abs/2607.26232v3) · [Pinned dataset and source terms](https://huggingface.co/datasets/anush-123/bg-real/tree/aaba6cd1ced4ce3c846be9641fed092adb4919d1)

The fit uses 16 native 24-pixel patches per source on eight Stable Diffusion 1.5 photos. At least 80% of a sampled patch lies inside the exact sharp edit. The original, whole-image autoencoder, local autoencoder, blur, contrast, sharpness and both generated-edit variants share those locations. Each produces aligned PNG and JPEG75 training patches. Both heads learn filters over high-pass RGB at two spatial scales. The binary head combines every ordinary control into one class. The second head learns a separate autoencoder class and penalizes changes in its raw edit response across the two encodings. The fit uses 1,024 aligned patch pairs per head. [Paired fit sources](image-bg-real-insertions-2026-11-11.md)

At inference, only the submitted image enters either head. Each scans a fixed 24 × 24 grid and ranks regions by within-image robust-normalized contrast. Eight other source groups provide ordinary originals and processing variants to fix the operating point independently for each head and delivery. The measured scores are uncalibrated. The comparison evaluates the extra autoencoder class and cross-format consistency together.

## Held comparison

Two previously exposed panels each use 12 source groups, evenly split between SDXL and Kandinsky. A separate frozen publisher intake found seven additional source groups from the unused eligible pool, four SDXL and three Kandinsky. It verified all 28 acquired files, screened the photos against the existing exposure graph, and checked that each constructed local edit stays inside the publisher mask. Recorded photographer accounts and photo IDs are disjoint across fitting, point selection, and the three held panels. Unrecorded contributor aliases and upstream editor training overlap remain unknown. The publisher records CC BY attribution for its source photos and derived test assets. [First held panel](image-background-patch-transfer-2026-11-11.md) · [Second held panel](image-wavelet-region-transfer-2026-11-11.md)

Each source contributes one sharp edit, one feathered edit, and six ordinary controls per format. The same precompression edited-pixel truth is used for both deliveries. Useful localization requires at least 0.30 intersection over union and at least 0.50 recall. The comparison baseline uses the existing PNG sharp-boundary or JPEG double-quantization region on the exact submitted view.

| Format and check | Sharp /31 | Feathered /31 | Ordinary highlights /186 |
| --- | ---: | ---: | ---: |
| PNG existing boundary check | 17 | 1 | 21 |
| PNG binary residual | 0 | 0 | 31 |
| PNG matched-processing residual | 0 | 0 | 25 |
| JPEG75 existing regional quantization check | 0 | 0 | 0 |
| JPEG75 binary residual | 0 | 0 | 34 |
| JPEG75 matched-processing residual | 0 | 0 | 37 |

On the seven fresh PNG groups, the existing boundary check located five sharp edits and one feathered edit, with seven ordinary highlights. Each learned head highlighted 12 of the 42 fresh ordinary controls in each format. The best unfiltered rectangle from either learned head had useful overlap in only four edited deliveries. These were the sharp and feathered PNG and JPEG75 variants of the same exposed Kandinsky source. Changing the point alone would leave most edits unlocalized.

A separate offline replay rebuilt 496 encoded deliveries and exact truth masks, checked 1,488 native overlap records, and reproduced both advancement decisions from saved responses, calibration maxima and incumbent regions. The fit and confirmation commands run inference again. The public receipt contains counts and hashes. Photos, learned weights and per-image responses stay local.

## Replay

Use the pinned BG-REAL metadata and retained source intakes. Keep a separate folder for the seven-group source qualification and the frozen detector run.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze acquire; do
  backend/.venv/bin/python -B -m scripts.image_residual_patch_intake "$phase" \
    "$metadata" "$fit" "$calibration" "$earlier" "$later" "$fresh" || exit 1
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for phase in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_residual_patch_transfer "$phase" \
    "$fit" "$calibration" "$earlier" "$later" "$fresh" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_residual_patch_transfer "$run" "$earlier" "$later" "$fresh"
```

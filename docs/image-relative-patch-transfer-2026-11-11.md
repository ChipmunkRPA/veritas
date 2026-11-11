# Regional response to generated image edits

Recent work on relative patch responses observes that replacing one area can change the representation of distant, unchanged pixels. We tested a compact version of that idea on existing source-qualified images. It uses a frozen 21.8-million-parameter visual classifier, one fitted direction and a native-coordinate region readout. We used the published paper to choose the comparison and the existing MIT checkpoint for features. The authors' model, training data and code were outside this experiment. [Relative Patch Response Learning](https://arxiv.org/abs/2610.11876) · [Checkpoint and license](https://huggingface.co/OwensLab/commfor-model-384)

## Fixed panel and method

Twelve publisher LaMa object-removal source groups provide eight fitting groups and four point-selection groups. Each has the original, generated edit, and three conventional edits using tone, blur, and copy-move. The formats are the source delivery and a matched JPEG75 encoding. For each edit, the fit computes the mean change in the aligned patch features inside its exact footprint, after subtracting the mean change outside the affected cells and their immediate neighbors. It then subtracts the average response of the three ordinary edits from that of the generated edit. Equal-weight normalized contrasts fit a regularized linear direction. [LaMa source and rights](image-bg-real-backup-intake-v262-method.md)

The comparison tests that fitted direction against the frozen classifier's own linear projection. Both project one full-image forward onto a 24×24 grid, compare local responses with the within-image median and dispersion, and select the strongest 4-, 6-, or 8-cell square. Each internal flag point is the second-highest control-group maximum on the four development groups. The rule uses a strict comparison and the points are written before the transfer images are scored. The actual mask is available for fit and evaluation. Prediction on a transfer image uses its pixels alone.

The transfer contains six different source groups with locally composed FLUX Klein edits, none of which were used to choose the direction or points. These are the same source-qualified 256×256 groups used in the v288 production comparison, with one original, three ordinary controls, and one generated edit per group and format. Localization requires at least 0.30 IoU and 0.50 recall of the exact pre-encoding changed pixels. These are study criteria. The transfer uses a separate generator and source group from fitting, and the photos have prior project exposure. [FLUX inputs and exact masks](image-flux-photo-expansion-v285-result.md) · [Earlier product comparison](image-flux-product-comparison-v288-result.md)

## Results

| Method | Source edits localized /6 | Source controls highlighted /24 | JPEG75 edits localized /6 | JPEG75 controls highlighted /24 |
| --- | ---: | ---: | ---: | ---: |
| Current regional routes, frozen v288 | 0 | 6 | 0 | 13 |
| Frozen classifier projection | 0 | 4 | 0 | 0 |
| Fitted relative response | 0 | 13 | 0 | 17 |

The fitted direction produced four flagged generated edits per format, all with zero overlap. The frozen projection produced one highlight on a generated source delivery with 0.451 IoU and 0.451 recall, below the required recall. Missed positives count as zero in the means. The fitted method remains out of the app.

The CPU worker made 180 actual forwards in each of two runs, with complete source and format coverage. The first worker wrote all outputs before the BSD timing wrapper exited on an unavailable system statistics call. The second run exited zero and reproduced the exact point files, directions, and counts. [Aggregate and freeze commitments](evaluation-panels/image-relative-patch-transfer-2026-11-11.json)

## Independent crop inputs

A second experiment isolates the classifier's ordinary image response from full-image patch attention. It scores 25 overlapping source crops spanning one third of each image side, each resized to 384×384. Both arms choose the same strongest crop. One uses the strongest raw logit and the other subtracts the median logit of all 25 crops from that logit. Separate points were fixed using the same four development groups and their 32 controls before any transfer crop scoring.

Both arms missed all six transfer edits in each format and flagged zero of the 24 controls per format. Ranking AUC was 0.476 source and 0.424 JPEG75 for the absolute crop. It was 0.431 source and 0.531 JPEG75 for the relative crop. All 92 planned views reached the model, with 2,300 crop inputs in 460 CPU batches. The run exited zero in 124.9 seconds at approximately 658 MB process RSS. Its localization counts passed the portable receipt checks using native source-pixel extents. Both crop rules remain experimental. [Crop comparison and exact aggregates](evaluation-panels/image-crop-context-transfer-2026-11-11.json)

The next wider source comparison needs explicit photo permissions and exact matched edit footprints. DailyBench provides more recent generators and states Apache-2.0 for its dataset release. Its paper sources the original images from LAION-Aesthetics, so the release license alone does not establish photo-level rights. We did not acquire or score that media for this comparison. [DailyBench source method](https://arxiv.org/html/2607.24016v6#S2.SS1) · [Publisher dataset card](https://huggingface.co/datasets/WhiteJiangzz/DailyBench)

## Local reproduction

With the private source-qualified v262/v264 and v285/v288 artifacts already installed, use fresh output directories and the pinned MIT checkpoint file.

```sh
export PYTHONPATH=backend VERITAS_OFFLINE=true HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -m scripts.image_relative_patch_transfer freeze "$CHECKPOINT" "$PATCH_OUTPUT"
backend/.venv/bin/python -m scripts.image_relative_patch_transfer run "$CHECKPOINT" "$PATCH_OUTPUT"
backend/.venv/bin/python -m scripts.image_crop_context_transfer freeze "$CHECKPOINT" "$CROP_OUTPUT"
backend/.venv/bin/python -m scripts.image_crop_context_transfer run "$CHECKPOINT" "$CROP_OUTPUT"
```

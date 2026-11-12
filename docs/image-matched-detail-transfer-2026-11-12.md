# Fine-detail changes relative to similar parts of one image

The two fine-detail checks localized zero of 48 known edits at their fixed points. The available rectangle shapes could have met the overlap rule for 45 edits. Even before applying a point, the highest-ranked rectangle was useful on two images with the spatial reference and one with the distant matched reference. The failed advancement rule keeps the new check experimental. [Audited results](evaluation-panels/image-matched-detail-transfer-2026-11-12.json)

## Frozen comparison

Earlier fine-detail checks compared a proposed image area with nearby strips. Neighboring objects can differ in brightness and texture even in ordinary photographs. This study tests whether comparing similar regions farther away makes small edits easier to locate. Research on signal-dependent noise and color residuals motivates the measurement. The implementation uses fixed image statistics and runs locally without trained weights. [Signal-dependent noise study](https://arxiv.org/abs/2010.16211) · [Color residual study](https://arxiv.org/abs/2608.20558) · [Earlier local comparison](image-residual-color-transfer-2026-11-11.md)

Use complete 16 × 16 source cells. A cell supplies coarse brightness, contrast, gradient strength, and gradient direction balance. Its five fine-detail measurements are luma and chroma power ratios across two Gaussian residual bands, the fine-band chroma-to-luma ratio, one-pixel luma autocorrelation, and absolute luma–chroma correlation. Exclude cells with at least 15% nearly clipped luma pixels. The spatial comparison selects up to 18 cells more than two grid cells away. The matched comparison selects up to 18 cells more than six grid cells away within the fixed coarse-context distance. Either requires eight reference cells. Each feature is compared with the reference median and scaled by its median absolute deviation with a fixed floor.

Both arms examine the same rectangles with sides between three and 16 grid cells and an aspect ratio of at most two. At least 80% of the region must have a reference, and at least 60% of those cells must agree in direction for the same feature. The score uses their region average, a penalty for agreement extending into the immediately surrounding cells, and a modest area adjustment. Eight separate ordinary source groups set the two points using 32 original, contrast, blur, and resampling deliveries. Each point is the second-highest source-level maximum and must be strictly exceeded. A region is useful at 0.30 changed-pixel intersection over union and 0.50 recall. The frozen rule required gains on two LaMa feathered groups and another feathered delivery with limited additional ordinary flags. Responses carry zero AI-confidence weight.

The 138 held RGB PNG views were reused exactly from the earlier tile study. They include 12 LaMa source groups, six SDXL and Kandinsky source groups, and 12 screenshot sets paired with the LaMa sources. Each source has unchanged and ordinary processing controls. The screenshot sets have a text layout, the same layout displaying the unchanged photograph, and the display with the feathered edit. The original pixels, native edit masks, delivered RGB, and prior response ledgers are bound to the earlier source freeze. All source groups had earlier project exposure under other methods. This comparison is a fixed-method transfer on those files. The shipped sharp rectangular boundary check is measured separately at its existing point and file scope. [Source licenses, ancestry checks, and prior comparisons](image-native-tile-agreement-transfer-2026-11-12.md)

## Held outcomes

| Outcome | Sharp boundary | Earlier crop comparison | Earlier tile agreement | Spatial detail | Matched detail |
| --- | ---: | ---: | ---: | ---: | ---: |
| Useful native edits, 18 files | 4 | 1 | 0 | 0 | 0 |
| Useful feathered edits, 30 files including screenshots | 0 | 0 | 0 | 0 | 0 |
| Flagged ordinary photo views, 66 files | 4 | 8 | 8 | 0 | 6 |
| Flagged ordinary screenshots, 24 files | 13 | 3 | 16 | 7 | 9 |
| Useful highest-ranked regions before points, 48 edited files | 5 | 3 | 0 | 2 | 1 |

The exact edit occupied a median of 2.96% of the submitted pixels. Thirty-two of 48 edits occupied less than 5%. The four useful sharp-boundary results were native SDXL or Kandinsky edits. The matched detail check exceeded its point on seven of the 12 edited screenshots, yet its rectangles missed useful overlap in every one. It flagged nine ordinary screenshot groups and added seven such groups relative to the earlier crop rule. The retrospective comparison still found zero useful locations in both arms at equal budgets of zero, one, two, four, or eight false groups. [Related evidence on small inpainted regions](https://arxiv.org/abs/2512.16688)

A deterministic pixel-level smoke added a 96 × 96 alternating red-channel change to a smooth textured image. The matched arm reported its exact bounds with a response of 6.37 against the sealed point of 4.44. The unchanged view scored 0.81. This verifies the intended response to that authored perturbation.

The initial held launch stopped on a historical comparator record that stores a score list. Replaying its prior score reduction resolved the reader error. The complete rerun reproduced the original source freeze, method parameters, and all 32 calibration rows. It checked the historical crop rule on the same decoded RGB views and completed held scoring in about 49 seconds with zero model forwards.

## Reproduce

The public ledger contains scalar decisions and geometry. The standard-library replay independently checks 690 held decisions, the operating points, location arithmetic, and matched false-group counts. It rejected an altered held flag. Native replay checked 170 exact changed-pixel masks, 748 recorded rectangle overlaps, and 340 fine-detail map commitments. No source pixels, masks, or model weights accompany the ledger.

```sh
python3 -B -I -S backend/scripts/review_image_matched_detail_transfer.py \
  docs/evaluation-panels/image-matched-detail-transfer-2026-11-12.json
```

For native replay, retain the exact earlier local image-study inputs, screenshot captures, and score ledger. Set `alternate` to the qualified later SDXL and Kandinsky holdout and `run` to a new private output directory. The native reviewer reconstructs and checks the exact changed-pixel masks, independently measures overlaps at the recorded regions, and replays the residual maps.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze calibrate held; do
  backend/.venv/bin/python -B -m scripts.image_matched_detail_transfer "$phase" "$run" --alternate "$alternate" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_matched_detail_transfer "$receipt" \
  --native "$run" --alternate "$alternate"
```

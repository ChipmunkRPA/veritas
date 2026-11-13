# Signed residual phase on localized generated edits

Neither phase check located a known edit at its frozen point. Both returned zero flags on 90 ordinary and interface controls and zero useful regions on 48 generated edits. The existing sharp-boundary check located four native edits and missed all 30 feathered deliveries. The two phase checks remain offline. [Audited results](evaluation-panels/image-phase-locked-residual-2026-11-12.json)

## Frozen method

Prior research measures periodic peaks in the noise residuals of generated images. This CPU test asks whether a localized region preserves a common pixel-grid phase across its four quadrants while the surrounding ring has weaker periodicity. Signed residual coefficients distinguish this measurement from an earlier project experiment that checked periodic variations in the magnitude of interpolation residuals. The detector sees only the submitted RGB pixels. It uses no trained model or outside code and contributes zero authorship weight. [Study of periodic generator artifacts](https://openaccess.thecvf.com/content/CVPR2023W/WMF/papers/Corvi_Intriguing_Properties_of_Synthetic_Images_From_Generative_Adversarial_Networks_to_2023_paper.pdf) · [Earlier resampling check](image-local-resampling-transfer-2026-11-11.md)

Subtract a fixed Gaussian smooth from luma, red minus green and blue minus green. Normalize the signed residual by local root-mean-square energy with a fixed floor. Each complete 16 × 16 native cell contributes Fourier coefficients at four orientations for 4- and 8-pixel periods. Exclude heavily clipped or near-flat cells. For each rectangle, project its quadrant coefficients onto the phase of its interior mean and use the weakest aligned quadrant. Subtract 65% of the surrounding-ring amplitude. A fixed area term favors distributed support. An authored period-four chroma perturbation verified that the implementation responds to a coherent local grid. The test does not estimate how commonly actual edits preserve that grid.

Eight separate licensed photos supply 32 ordinary calibration views, with the cutoff at the second-highest source maximum per arm. The held comparison reuses 12 source-preserving LaMa photo groups, six separate SDXL and Kandinsky groups, and the exact Chromium captures from the prior native-tile study. Every held delivered PNG and changed-pixel mask was compared against the pinned source digest. The photos have earlier project exposure. Their roles are separated from the calibration photographs. The 12 screenshots containing edits share underlying sources with the LaMa panel. A useful location recovers at least half the changed pixels and reaches 0.30 intersection over union. [Original source qualification and rights](image-native-tile-agreement-transfer-2026-11-12.md)

| Held outcome | Existing sharp boundary | Four-pixel phase | Eight-pixel phase |
| --- | ---: | ---: | ---: |
| Native edit locations | 4/18 | 0/18 | 0/18 |
| Feathered locations, including screenshots | 0/30 | 0/30 | 0/30 |
| Flags on ordinary photo views | 4/66 | 0/66 | 0/66 |
| Flags on ordinary screenshots | 13/24 | 0/24 | 0/24 |
| Top-ranked useful locations before the cutoff | 5/48 | 3/48 | 0/48 |

Both phase arms assessed all 138 held images. The incumbent assessed 132. The frozen advancement gate required two primary feathered gains and one alternate-editor or screenshot gain, with limits on incorrect highlights. The candidates missed the gain requirements. Lower cutoffs cannot recover useful top-ranked locations beyond the counts shown here. A first launch stopped before saving a held result after the box filter returned tiny negative values for some local means of squares. The corrected implementation clamps floating-point roundoff at zero and asserts finite coefficients. The input roster and gate stayed fixed for the new freeze.

## Replay

Run from the repository root. Set `alternate` to the retained later SDXL and Kandinsky holdout from the earlier native-tile comparison. The native replay reconstructs the exact pinned RGB and truth masks, checks calibration and reproduces every response. The standard-library audit independently recomputes the cutoffs, overlap decisions and advancement arithmetic from the frozen ledger.

```sh
export PYTHONPATH=backend OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for phase in freeze calibrate held review; do
  backend/.venv/bin/python -B -m scripts.image_phase_locked_residual "$phase" "$run" --alternate "$alternate" || exit 1
done
python3 -B -I -S backend/scripts/review_image_phase_locked_residual.py "$run"
```

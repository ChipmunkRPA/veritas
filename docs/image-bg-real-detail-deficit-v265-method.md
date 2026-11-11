# Local cross-band detail on matched object-removal and ordinary edits

Measure whether an edited region has less fine detail than nearby regions with similar coarse detail. The exploratory cue uses only the current image. A clean reference and known masks enter the evaluation, where they allow comparisons at exactly the same locations across source-related views. The prior fitted noise and texture localizer measured single-residual power, spatial correlation, and ratios across spatial windows. This cue measures variance shared between two separate spatial-frequency bands. [Prior residual experiment](image-residual-v13-result.md)

Convert RGB to luminance with weights 0.299, 0.587 and 0.114. The fine band is luminance minus Gaussian smoothing at 0.8 pixels. The coarse band is the difference between smoothing at 0.8 and 2 pixels. Use reflective boundaries. For each full 16-pixel tile, subtract its band means and measure variance and covariance. Estimate unexplained fine-band variance by subtracting squared cross-band covariance divided by coarse variance plus 0.25, with a floor of zero. Take the natural log of coarse variance plus 0.25 divided by unexplained fine-band variance plus 0.25. Subtract the median over a 5 × 5 tile neighborhood. Higher values indicate a local fine-detail deficit. This directional hypothesis and every parameter are fixed before measuring the project images.

Use the 12 exact groups already evaluated in v264. Reconstruct the five views from the qualified sources and verify the retained footprints and JPEG bindings. Evaluate source RGB, JPEG quality 75, and JPEG quality 90 separately. The groups repeat across all three formats. Keep the exact AI edit footprint fixed across the original, LaMa, tone, blur and copy-move views. Also evaluate each ordinary edit using its own changed-pixel footprint. Trim incomplete border tiles and report how much exact affected area remains. [Qualified comparison sources](image-bg-real-backup-jpeg-v264-result.md)

For every affected tile, calculate its concordance with all tiles that do not touch the footprint or an affected tile, giving ties half credit. Weight affected tiles by their exact changed-pixel count. Also calculate the area-weighted mean signed response at the same fixed site. Compare the LaMa response to each matched control and the highest of the three matched ordinary-edit responses. Give each source group equal weight in aggregate summaries. Report counts whenever a comparison lacks usable target or background tiles. These are retrospective descriptive comparisons on one editing family, with no model calls, thresholds, fitted parameters or customer-facing decisions.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_bg_real_detail_deficit_v265 freeze
backend/.venv/bin/python -B -m scripts.image_bg_real_detail_deficit_v265 run
```

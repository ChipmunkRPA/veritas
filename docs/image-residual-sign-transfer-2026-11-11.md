# Residual sign patterns on small generated image inserts

Three residual sign checks found zero useful locations on 96 sharp or feathered positive deliveries. The two-scale check flagged 3 of 144 ordinary PNG views and 6 of 144 ordinary JPEG75 views. The fixed comparison leaves the candidate experimental. [Exact counts and method commitments](evaluation-panels/image-residual-sign-transfer-2026-11-11.json)

## Method

Prior image-forensics work describes co-occurrence statistics on quantized high-pass residuals. Separate research finds that detectors trained on fully generated images struggle on small inpainted regions. This CPU experiment measures sign-transition distributions within the submitted image. It uses no external weights or detector training code. [Residual descriptor research](https://arxiv.org/abs/1703.04615) · [Localized inpainting evaluation](https://arxiv.org/abs/2512.16688)

Use native luma on a 16-pixel grid. The two bands use second differences at strides one and two. For each orientation, count the four ways the sign changes across three adjacent residuals. Discard triplets with any absolute residual below one gray level. Split the remaining samples by whether their local mean absolute residual exceeds four. On the known JPEG75 grid, retain only triplets whose residual footprint stays inside one block along the measurement axis. The local activity estimate can include neighboring pixels.

A candidate compares its inset core with four nearby strips using squared Hellinger distance between the conditional distributions. Each comparison needs at least one stratum with 32 samples in both regions for each orientation. The score takes the third-largest core-to-strip distance after subtracting the median distance between strips. The two-scale arm uses the smaller score at the same rectangle. Each arm selects its highest-ranked region. The selection rule considers a match at two scales as one local observation with zero authorship confidence weight.

The strict point is the second-highest of eight control-source maxima, with a minimum of 0.002, fitted separately by delivery and arm. The development cohort uses the retained SD1.5 panel. Two groups of 12 other photos supply sharp and feathered SDXL or Kandinsky inserts, unchanged sources, blur, contrast, and ordinary local or global autoencoder processing controls. The two held cohorts have earlier project exposure on other methods. Photo and recorded contributor groups remain disjoint from fitting and from each other. The PNG and JPEG75 variants share exact precompression changed-pixel truth. [Fit source qualification](image-bg-real-insertions-2026-11-11.md) · [First held qualification](image-background-patch-transfer-2026-11-11.md) · [Second held qualification](image-wavelet-region-transfer-2026-11-11.md)

The reference is the existing PNG sharp-boundary comparison and the JPEG aligned double-quantization region. A useful region needs at least 0.30 intersection over union and 0.50 recall. The two-scale arm needed two newly localized feathered edits in each panel per delivery and at most four additional ordinary flags on at most three source groups. The comparison covers these regional checks and these exact deliveries.

## Held results

| Format | Check | Useful sharp /24 | Useful feathered /24 | Ordinary flags /144 | Untouched flags /24 |
| --- | --- | ---: | ---: | ---: | ---: |
| PNG | Format reference | 12 | 0 | 14 | 2 |
| PNG | Fine triplets | 0 | 0 | 1 | 0 |
| PNG | Wide triplets | 0 | 0 | 3 | 0 |
| PNG | Two scales | 0 | 0 | 3 | 0 |
| JPEG75 | Format reference | 0 | 0 | 0 | 0 |
| JPEG75 | Fine triplets | 0 | 0 | 24 | 4 |
| JPEG75 | Wide triplets | 0 | 0 | 0 | 0 |
| JPEG75 | Two scales | 0 | 0 | 6 | 1 |

All candidate arms returned a measurable proposal on every view. Removing the point after evaluation would leave useful highest-ranked two-scale proposals on six PNG positives and zero JPEG75 positives. The fine arm could localize one of 48 JPEG75 positives when ignoring its point. The test therefore gives little support for recalibration of the JPEG ranking on these inputs.

The 384-view held run took 48.7 seconds. A separate readback rebuilt all exact deliveries and truth masks, verified 1,536 mask hashes and overlap counts, and reproduced all 16 aggregate cells. Five focused tests check the native sampling coordinates and JPEG block rule against direct arithmetic and recover a deliberately authored periodic region. The tests also verify that a flat image yields no assessed proposal.

## Replay

Supply the retained development and two confirmation intake directories and the earlier blended-boundary `rows.json`. Use a fresh output directory. Keep the original licensed sources and per-case maps in the local workspace.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_residual_sign_transfer "$stage" \
    "$development" "$earlier" "$later" "$boundary_rows" "$output" || break
done
backend/.venv/bin/python -B -m scripts.review_image_residual_sign_transfer \
  "$earlier" "$later" "$output" "$output/audit.json"
```

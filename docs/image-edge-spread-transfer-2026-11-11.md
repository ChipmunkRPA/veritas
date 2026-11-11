# Matched edge widths around local image edits

Measuring the width of individual edges found a small number of complementary edit locations, including feathered changes. The opposing-flank method found one new useful feathered location on the earlier PNG panel and none on the later PNG panel. At JPEG75 it found one on each panel and flagged 63 of 144 ordinary controls. Both formats fell short of the fixed advancement rule. [Full results](evaluation-panels/image-edge-spread-transfer-2026-11-11.json)

## Method and source panels

For each supported edge in native luminance pixels, sample across the direction of strongest brightness change. Use the median of three parallel profiles to estimate the distance between 20% and 80% of the contrast across the edge. The check declines weak or complex transitions and selects at most one measured edge per 4 × 4 pixel cell. The candidate compares average log widths for edges matched by contrast and orientation inside the region and in a separated surrounding ring. A second arm requires the direction of the difference to agree on two opposite sides and uses the smaller corroborating difference. The proposed mask is the enclosing rectangle in source coordinates. [Primary edge-spread method](https://infoscience.epfl.ch/entities/publication/4693a9fb-ebf2-495f-89d4-2688cf4af9f7/conferencedetails) · [Blur consistency in image forensics](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Image_Splicing_Detection_CVPR_2017_paper.pdf)

An authored image with a local Gaussian blur produced a region with edges 2.20 times as wide as the matched surroundings. The same image with whole-image blur stayed below the fixed minimum response. On eight rights-qualified development photos, 48 controls per delivery set the review point for each arm at the second-highest per-source maximum, with a fixed minimum. Development controls include whole-image processing, local blur, sharpening, and contrast. The runner froze the methods and source commitments before scoring the confirmation panels. [Exact profile](evaluation-panels/image-edge-spread-transfer-freeze-2026-11-11.json)

The two confirmation panels contain 12 photos each, with distinct source and creator groups across both panels and development. Both were used in prior project studies. The replay binds all PNG and JPEG75 views to the earlier sealed results, using original, blur, contrast, local and global processing, and feathered processing controls. Sharp and feathered positive masks record exact RGB changes before encoding. BG-REAL lists CC BY 2.0 for these photos and CC BY 4.0 for its contributions. The photo bytes, identities, and individual region coordinates remain with the evaluator. [Earlier comparison and rights](image-background-patch-transfer-2026-11-11.md) · [Later qualification and wavelet comparator](image-wavelet-region-transfer-2026-11-11.md)

## Results

A useful location has at least 0.30 intersection over union and 0.50 recall against the exact changed pixels. Any reported rectangle on a control counts as a flag. Each row covers 12 source groups. The controls and formats reuse source photos.

| Panel | Delivery | Check | Sharp /12 | Feathered /12 | Controls /72 | Control groups /12 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Earlier | PNG | Matched ring | 0 | 1 | 13 | 3 |
| Earlier | PNG | Opposed flanks | 0 | 1 | 7 | 2 |
| Earlier | PNG | Wavelet detail | 0 | 1 | 1 | 1 |
| Later | PNG | Matched ring | 0 | 0 | 20 | 4 |
| Later | PNG | Opposed flanks | 1 | 0 | 25 | 5 |
| Later | PNG | Wavelet detail | 0 | 0 | 8 | 3 |
| Earlier | JPEG75 | Matched ring | 0 | 1 | 7 | 3 |
| Earlier | JPEG75 | Opposed flanks | 0 | 1 | 27 | 5 |
| Earlier | JPEG75 | Wavelet detail | 1 | 1 | 2 | 2 |
| Later | JPEG75 | Matched ring | 1 | 0 | 7 | 2 |
| Later | JPEG75 | Opposed flanks | 1 | 1 | 36 | 6 |
| Later | JPEG75 | Wavelet detail | 0 | 0 | 2 | 2 |

The earlier feathered location was different from the wavelet result in both deliveries. All useful edge-spread locations had wider edges inside the proposed region. Requiring agreement on opposite flanks changes the score scale. Its separately fitted cutoff yielded more JPEG75 flags than the ring-only arm on both panels. The frozen rule required at least two complementary feathered locations, at most ten control flags, and at most six affected control groups in each panel. The result remains experimental. Natural depth of field, ordinary editing, and compression can all change measured edge widths. The cue carries zero weight in authorship confidence.

## Reproduce locally

Use the retained development and confirmation intakes, the prior boundary `rows.json`, the earlier wavelet `rows.json`, and a fresh output folder. The later wavelet journal is read from the later intake. The CPU runner processed all 384 confirmation views in 42.9 seconds. A separate Pillow implementation reconstructed all 384 inputs and truth masks, verified 768 candidate mask hashes, and checked 1,152 measurements and 12 summary rows. [Runner](../backend/scripts/image_edge_spread_transfer.py)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_edge_spread_transfer "$stage" \
    "$development" "$earlier" "$later" "$boundary_rows" "$earlier_wavelet_rows" "$output" || break
done
```

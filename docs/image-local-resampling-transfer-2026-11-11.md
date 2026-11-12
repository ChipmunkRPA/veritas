# Localizing resize traces in native images

The quadrant-agreement check found useful locations on 11 of 72 deliberately resized PNG patches, including four feathered patches. It flagged three of 216 ordinary PNG controls. It located none of the same resized patches after JPEG75 delivery and none of the unmodified generated patches in either format. Neither delivery reached the frozen research criteria. [Aggregate and fixed points](evaluation-panels/image-local-resampling-transfer-2026-11-11.json)

## Signal and protocol

The check measures periodic changes in the magnitude of horizontal and vertical second differences. It divides each residual by its nearby average magnitude, bounds extreme values, and removes slower changes. Each native 16-pixel cell is projected onto fixed spatial frequencies outside the main JPEG grid harmonics. A candidate rectangle needs an excess at the same frequency in both axes. The measurement compares the inside with its surrounding ring and with each of four neighboring strips. Checking the strips separately reduces phase cancellation when a periodic pattern covers the entire file. The second arm also requires two-axis phase agreement in three of four interior quadrants. The mask is the selected rectangle in native source coordinates. [Primary work on interpolation periodicity](https://library.utia.cas.cz/separaty/2008/ZOI/saic-blind%20authentication%20using%20periodic%20properties%20ofinterpolation.pdf) · [Resampling forensics study](https://hfarid.org/downloads/publications/sp05.pdf)

The authored qualification used a deterministic textured RGB field. Both PNG and JPEG75 recovered the expected frequencies of 1/6 and 2/7 cycles per pixel in local bilinear, bicubic, and feathered views. Applying the same scale to the entire authored image or copying a patch at its existing size stayed below the fixed minimum. Eight licensed development photos then supplied 144 ordinary control views across the two deliveries. The strict review point is the second-highest development source maximum or the fixed minimum, whichever is higher. The code, source commitments, transforms and advancement rules were frozen before scoring the confirmation panels. [Exact freeze](evaluation-panels/image-local-resampling-transfer-freeze-2026-11-11.json)

Each of two 12-photo confirmation panels contains disjoint credited photographers and source images. They had prior project evaluation exposure. Each photo supplies the exact eight previously sealed images, including sharp and feathered generated patches at their existing sizes. Six new versions add localized bilinear and bicubic enlargements, a feathered bilinear enlargement, whole-image bilinear and bicubic controls, and an unresized native copy. Each photograph uses either a 6/5 or 7/5 scale along both axes. The PNG and JPEG75 truth masks share the same exact changed RGB pixels measured before delivery encoding. BG-REAL declares CC BY 2.0 for the selected photographs and CC BY 4.0 for its contributions. Source media and individual location records stay with the evaluator. [Earlier source rights](image-background-patch-transfer-2026-11-11.md) · [Later source qualification](image-wavelet-region-transfer-2026-11-11.md)

## Results

Each panel and delivery has 36 deliberate resize attempts, 24 unmodified generated patches, and 108 ordinary controls. On one earlier photo, all three localized resize attempts left the source pixels unchanged. Those three attempts remain in the frozen denominator. A useful location needs at least 0.30 intersection over union and 0.50 recall of exact changed pixels. A control flag is any selected rectangle on an ordinary view.

| Panel | Delivery | Check | Resized useful /36 | Feathered resize /12 | Generated useful /24 | Controls /108 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Earlier | PNG | Both axes | 5 | 2 | 0 | 1 |
| Earlier | PNG | Quadrant agreement | 5 | 2 | 0 | 1 |
| Later | PNG | Both axes | 6 | 2 | 0 | 2 |
| Later | PNG | Quadrant agreement | 6 | 2 | 0 | 2 |
| Earlier | JPEG75 | Both axes | 0 | 0 | 0 | 6 |
| Earlier | JPEG75 | Quadrant agreement | 0 | 0 | 0 | 6 |
| Later | JPEG75 | Both axes | 0 | 0 | 1 | 3 |
| Later | JPEG75 | Quadrant agreement | 0 | 0 | 0 | 2 |

The PNG geometry results covered both tested factors in the earlier panel and only the 6/5 factor in the later panel. The three PNG control flags came from three photos. They occurred on one native copy, one whole-image resize and one local contrast edit. The frozen research rule required 18 useful resized regions, including four feathered regions, with at most 14 ordinary flags from six source groups on each panel. The cue remains experimental. Natural patterns and ordinary resizes can produce periodicity. The check covers axis-aligned interpolation at the studied factors, with zero contribution to AI confidence.

In a separate readback, the best positive-response quadrant rectangle before applying the point could localize 20 of 72 resize attempts in PNG and four of 72 in JPEG75. All 20 PNG proposals and three of the four JPEG75 proposals recovered a frequency within 0.01 cycles per pixel of the expected interpolation frequencies. The result gives a descriptive upper count for this ranking on these images. The frozen comparison points were retained.

## Local reproduction

Use the retained development and confirmation intakes, the earlier boundary `rows.json`, and a fresh output directory. The later comparison journal is read from the later intake. The CPU run scored 672 confirmation views in 40.2 seconds. A separate Pillow replay regenerated 672 exact deliveries and truth masks, verified 1,344 candidate mask hashes and overlap measurements, and checked all eight summaries. [Local runner](../backend/scripts/image_local_resampling_transfer.py)

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze fit confirm; do
  backend/.venv/bin/python -B -m scripts.image_local_resampling_transfer "$stage" \
    "$development" "$earlier" "$later" "$boundary_rows" "$output" || break
done
```

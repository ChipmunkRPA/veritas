# Predicting fine colors from nearby color variation in JPEG edits

A small photographic neighborhood can have one dominant color direction when brightness varies across a material. A localized synthetic replacement might change that relationship at fine scales. This CPU trial fits the dominant linear RGB direction on one half of a local block, measures errors on the other half, and swaps the halves. The color-line hypothesis comes from image matting. The experiment uses its own measurements, no published detector implementation, and no inference weights. It differs from earlier tests of correlation in fixed residual bands and relative luma-gradient orientation. [Primary color-line research](https://people.csail.mit.edu/alevin/papers/Matting-Levin-Lischinski-Weiss-PAMI.pdf) · [Earlier color residual comparison](image-color-residual-transfer-2026-11-11.md) · [Earlier direction comparison](image-directional-structure-transfer-2026-11-13.md)

## Frozen comparison

Decode the final JPEG and convert its RGB values to linear sRGB. On each complete native 16 × 16 cell, average the same aligned 2 × 2 samples in the original and a version blurred with a 1.8-pixel Gaussian. Split the 64 pooled colors into two checkerboard halves. On each training half, fit a center and the axis of maximum color variance. On the other half, measure the share of color variation perpendicular to that axis. The first arm compares fine and coarse errors measured about their own axes. The second tests the fine samples against both axes. Each feature uses the weaker of the two held-out comparisons. Flat, strongly clipped or poorly explained coarse cells contribute no measurement.

Rank native rectangles of three to twelve cells per side with adequate compared cells and similar coarse brightness and color variation in the core and two opposed surrounding strips. Each arm returns at most one rectangle. The fixed point is the second-highest source maximum on 40 ordinary-only JPEG75 deliveries from eight separate photo groups, with a strict exceedance rule. Useful localization requires 30% changed-pixel intersection over union and 50% recall. The advancement rule requires three new useful feathered groups across two families, with at most two new flagged ordinary groups and at most four misplaced edit highlights.

The first held panel contains 12 separate attributed photos and six edits per family from Kandinsky and Stable Diffusion 1.5. A second panel contains six LaMa removal sources and one source photograph with two SDXL and Kandinsky deliveries. The latter variants share one photo and count as a single source group. All held inputs had earlier project exposure under different methods. Cases use exact pre-final-encode changed-pixel masks and common JPEG75 4:2:0 delivery. Matched ordinary views include unaltered photos, blur, tone, same-image copies, and where available resize, prior compression or autoencoder processing. The existing aligned JPEG luma check receives the same bytes. [First panel, generator terms and credited source qualification](image-three-class-patch-fresh-2026-11-12.md) · [Second panel and source rights](image-masked-context-new-editors-2026-11-13.md)

Both arms located a deterministic 96 × 96 region containing a second fine color direction in a controlled decoded RGB field. The dominant-color baseline had scores below 0.00004 and a flat image produced no region. This tests the measurement on its intended contrast. It supplies no accuracy evidence for JPEG delivery or an AI editor.

## Results

| Held panel | Check | Candidate coverage | Sharp useful | Feathered useful | Ordinary flags | Wrong edit regions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Main | Current JPEG luma | 0/96 selected | 0/12 | 0/12 | 0/72 | 0/24 |
| Main | Fine versus coarse errors | 96/96 | 0/12 | 0/12 | 6/72 | 2/24 |
| Main | Transferring the coarse axis | 96/96 | 0/12 | 0/12 | 6/72 | 3/24 |
| Second | Current JPEG luma | 3/64 selected | 0/8 | 0/8 | 3/48 | 0/16 |
| Second | Fine versus coarse errors | 56/64 | 0/8 | 0/8 | 0/48 | 0/16 |
| Second | Transferring the coarse axis | 56/64 | 0/8 | 0/8 | 19/48 | 8/16 |

The main panel's ordinary flags came from one photo group for the first arm and two groups for the second arm. Neither arm put a useful rectangle first on any of the 40 held edits even before applying the points. In the main panel, each arm selected the same rectangle in the original and the feathered file on ten of twelve photo groups. This suggests that the strongest color-line response mostly followed existing scene content. The cue stays offline and contributes zero AI confidence weight.

## Replay

The first complete one-thread run processed all 200 deliveries in 30.27 seconds, and a separate execution rebuilt and compared all native JPEG and exact changed-mask commitments and each reported measurement. An initial attempt stopped while writing the second bank's output because the reused source iterator temporarily changes its scratch path. The completed run fixed the destination binding without changing the signal, source split, or points. Raw images, individual source rows and masks remain local. [Frozen result and replay receipt](evaluation-panels/image-color-line-transfer-2026-11-13.json)

Use the same retained sources and prior ledgers as the [gradient direction transfer](image-directional-structure-transfer-2026-11-13.md#replay). The pinned historical checkpoint is checked for provenance only.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_color_line_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --output "$run" --publication "$receipt" || exit 1
done
```

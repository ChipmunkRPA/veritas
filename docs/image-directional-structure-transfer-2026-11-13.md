# Fine and broad gradient directions around small JPEG edits

A local edit can change how fine image detail follows the broader structure around it. This CPU experiment tests for that change in a single uploaded JPEG. It measures the fine gradient angle relative to the broader gradient at the same native pixels. Earlier work tested amplitude and polarity coupling separately for fixed horizontal and vertical derivatives. The new measure treats the fine and broad derivatives as two vectors, controls for broad edge orientation and strength, and requires excess over both opposite surrounding strips. Gradient-domain editing motivates the hypothesis. [Primary gradient-domain editing study](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf) · [Earlier interband dependence study](image-jpeg-interband-dependence-2026-11-12.md)

Measure gradients on Gaussian-smoothed luminance at scales 0.8 and 4 pixels. Sample every third pixel on the native grid. Ignore weak or clipped pixels. For each supported sample, measure the squared perpendicular component of the normalized fine gradient relative to the broad gradient and whether the two directions oppose. Divide samples into four broad edge orientation bins and three strength bins. Only bins with enough samples in the center and both opposed surrounding strips contribute to a score. Compare the center and both strips within those bins. The joint rule retains the weaker of the two scaled directional responses on the same axis. All candidate rectangles and overlap checks use native source coordinates.

Eight separate photo groups supply 40 ordinary calibration deliveries. The score point per arm is the second-highest ordinary source maximum with a floor of zero. It is applied unchanged to the later comparisons. Confirmation has six source groups each for Kandinsky and Stable Diffusion 1.5. A second bank contains six LaMa neural removals and one separate photo with two diffusion editor variants. All had previous project exposure under other checks. Their exact original and edit pixels, masks, processing controls and source rights come from the retained prior intakes. No model weights or publisher detector code ran. [First source bank and rights](image-three-class-patch-fresh-2026-11-12.md) · [Additional editor bank](image-masked-context-new-editors-2026-11-13.md)

A useful region requires at least 30% intersection over union and 50% edited-pixel recovery. The frozen rule calls for three useful new feathered groups across two editor families, at most two additional ordinary groups, and at most four wrongly located edit flags. The sole synthetic positive confirms the directional statistic responds to controlled perpendicular and reversal counts, and the flat image returns no proposal. These controls establish implementation behavior.

## Local results

| Main confirmation | Useful sharp /12 | Useful feathered /12 | Ordinary flags /72 | Ordinary source groups flagged /12 | Wrong locations /24 |
|---|---:|---:|---:|---:|---:|
| Current aligned JPEG luma | 0 | 0 | 0 | 0 | 0 |
| Perpendicular component | 0 | 0 | 6 | 1 | 2 |
| Perpendicular and reversal together | 0 | 1 | 7 | 2 | 3 |

The joint rule found one Stable Diffusion 1.5 insertion. It found no Kandinsky edit and fell below the predeclared coverage rule. Even before applying the fixed points, the joint arm’s top box was useful for just one sharp and one feathered file. The perpendicular arm put a useful box first for two sharp and one feathered file. On the other editor bank, no arm found a useful region on any of sixteen edited deliveries, even before the fixed points. The joint arm gave no flags on the 48 external ordinary controls, versus three for luma and four for the perpendicular arm. The candidate remains offline.

## Replay

The one-thread run rebuilt and assessed all 200 exact JPEG75 inputs and native truth masks in 28.56 seconds. The separate replay repeated every calculation and compared the complete private records and aggregate. Raw photos and per-file measurements remain local. [Frozen receipt](evaluation-panels/image-directional-structure-transfer-2026-11-13.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_directional_structure_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --output "$run" --publication "$receipt" || exit 1
done
```

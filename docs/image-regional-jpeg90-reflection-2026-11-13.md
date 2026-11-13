# Local JPEG90 evidence after a reversible reflection

The prior JPEG90-to-JPEG75 comparison lost useful edit locations because a
second save removed all overlapping selected pixels in those cases. This
study uses the same JPEG90 decoded pixels, reversed left to right without
compression or resampling. It keeps the original proposed pixel sets and
measures their support in the aligned repeat. The measurement concerns how
a localizer responds to a reversible transform. Both passes use the same
weights and carry zero AI-authorship confidence. The document-forensics
study below motivates checking the stability of regional proposals under
alternative views. It does not validate the thresholds or JPEG90 scope
used here. [Prior JPEG quality test](image-regional-jpeg90-quality-agreement-2026-11-13.md)
· [Test-time augmentation research](https://arxiv.org/html/2609.38391v1)
· [Existing checkpoint and model terms](native-regional-serving-v47-result.md)

## Frozen comparison

Run the repeat only where the direct JPEG90 pass selected a component. Keep
the uploaded JPEG90 reliability mask in source coordinates for both passes.
The first arm retains a direct component when at least half of its original
pixels also belong to a selected component of the reflected response. It
uses the existing whole-image area gate and regional score threshold.

The second arm uses the repeated response without first thresholding it.
Compute the median inside the original component and the median in its
outside ring. Exclude unreliable pixels, the other selected native component,
and the first two pixels outside the component. The ring extends eight
pixels from the component and must have at least 32 eligible pixels and one
quarter as many eligible pixels as the component. Divide the median
difference by the larger of .02 or the outside interdecile range. Quantize
the repeated response to 12-bit steps for reproducible point comparisons.
The operating point is the third-highest ordinary source-group maximum
in development. A retained region has to strictly exceed it. Both arms retain
exact native components without expanding or filling their boundaries.

Development has 200 deliveries across the previous two source banks. The
11-photo confirmation has 88 deliveries with no overlap in source photo,
credited photographer or original bytes with either development bank. It
uses the same independently built qualification archive as the JPEG quality
confirmation, which had earlier project exposure. The current method and
point were sealed before evaluating the new reflected responses. Positive
variants include published LaMa removals and Stable Diffusion 1.5 insertions
in sharp and feathered versions. Ordinary variants include untouched,
local blur, tone and copy, a resize round trip and an earlier JPEG save.
The previous JPEG75, JPEG90 and JPEG90-to-JPEG75 baselines are reused by
exact delivery and mask commitments. [Prior confirmation and rights](image-regional-jpeg90-quality-agreement-2026-11-13.md)

A useful location covers at least half the exact changed pixels with at
least 0.30 intersection over union. On confirmation, the new arm must
preserve all four useful direct locations across both previously localized
source photos, add at most two ordinary highlights in at most one photo
relative to JPEG75, and add at most two misplaced edited highlights.

## Results

| Panel | Check | Useful edit locations | Ordinary highlights | Ordinary groups | Misplaced edit highlights |
| --- | --- | ---: | ---: | ---: | ---: |
| Two development banks | JPEG75 native | 11/46 | 13/154 | 5/22 | 4/46 |
| Two development banks | JPEG90 native | 11/46 | 30/154 | 8/22 | 9/46 |
| Two development banks | Reflection with hard overlap | 10/46 | 14/154 | 5/22 | 5/46 |
| Two development banks | Reflection with local contrast | 2/46 | 3/154 | 2/22 | 0/46 |
| Source-separated confirmation | JPEG75 native | 4/22 | 0/66 | 0/11 | 2/22 |
| Source-separated confirmation | JPEG90 native | 4/22 | 15/66 | 3/11 | 6/22 |
| Source-separated confirmation | JPEG90-to-JPEG75 agreement | 2/22 | 0/66 | 0/11 | 0/22 |
| Source-separated confirmation | Reflection with hard overlap | 4/22 | 10/66 | 2/11 | 4/22 |
| Source-separated confirmation | Reflection with local contrast | 2/22 | 0/66 | 0/11 | 0/22 |

The reflection with hard overlap preserved both localized source photos in
the separate bank and removed five ordinary highlights. It still showed
ten ordinary highlights across two photos, versus zero for JPEG75. The
local contrast arm repeated the earlier quality-agreement tradeoff: it
cleared the ordinary controls but lost both useful variants of one photo.
Both lacked enough reliable outside-ring pixels to assess contrast. The
same components had 96% and 99% exact hard overlap, so changing the contrast
point alone cannot recover them. Neither passed confirmation, and neither found a useful LaMa removal.
A descriptive sweep over the observed hard overlap values retained at least
ten ordinary highlights whenever it preserved all four useful locations.
That sweep did not change the frozen point or qualify a new product check.

A second native run reproduced all 288 delivery rows, exact component
hashes, response quantiles and operating-point decisions. The portable
reviewer matched on Python 3.11 and 3.12.
Of the 288 evaluated deliveries, 75 had native components and ran the
reflected view. The 99 measured native components included 63 with a
sufficient reliable ring to measure contrast. Six development source
groups supplied an ordinary maximum. The third-highest maximum was
422/705 in normalized response units. Public records contain anonymous
source aliases, counts, quantiles, bounds and hashes, without source
photos or model weights. [Outcome ledger](evaluation-panels/image-regional-jpeg90-reflection-2026-11-13.json)

## Replay

From the repository root, use the original source archives and checkpoint
from the earlier [JPEG quality replay](image-regional-jpeg90-quality-agreement-2026-11-13.md#reproduce).
Pass its direct-study reconstruction inputs to the new runner with
`--main-roster`, `--external-roster`, and `--confirmation-roster` from
its confirmation command. `--output` points to a new private directory.
Run the four phases in order. The development cutoff is written before
confirmation inference. Source images remain private.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze development confirmation replay; do
  backend/.venv/bin/python -B -m scripts.image_regional_jpeg90_reflection "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" --tile-run "$tile_run" \
    --main-roster "$later_f5/plan.json" --external-roster "$new_editor_transfer/plan.json" \
    --confirmation-roster "$early_f5" --output "$reflection_run" || exit 1
done
PYTHONPATH=backend python3 -B -S -m scripts.review_image_regional_jpeg90_reflection \
  "$reflection_run/receipt.json" \
  docs/evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json \
  docs/evaluation-panels/image-regional-jpeg90-quality-confirmation-2026-11-13.json
```

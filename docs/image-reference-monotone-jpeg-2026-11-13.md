# Earlier-copy comparison after JPEG tone changes

A small object removal can be difficult to locate from one compressed image.
An earlier copy supplies direct pixel evidence. The current same-framing
comparison models an affine whole-image tone adjustment. This study asks
whether a nonlinear tone adjustment fitted and checked in separate spatial
areas can recover more useful edit regions. The result describes differences
between the supplied copies and carries zero AI authorship-confidence weight.
[Single-upload removal results](image-flame-jpeg-transfer-2026-11-13.md)

## Frozen method

The first comparison supports upright RGB JPEGs with standard 4:2:0
baseline encoding, equal dimensions and sides of 512–768 pixels. It tests
earlier quality 75, 90 or 95 and submitted quality 75 or 90. Whole-image
phase correlation proposes the same framing. Detailed local 64-pixel
tiles must corroborate the correspondence across at least six of nine
image zones. Two disjoint checkerboard tile sets each fit a monotone
brightness curve per channel. The fit uses brightness-bin medians after
Gaussian smoothing. Each curve must explain at least 80% of the held
spatial fold within six RGB levels after simulating the submitted JPEG
codec. The retained residual takes the smaller difference under both
fits. Connected regions require high-residual core pixels and preserve
the native source coordinates.

Two retained, source-disjoint banks contain seven and six separate LaMa
object-removal photos. Each yields sharp and feathered versions, matched
local tone, blur and copy controls, and ordinary unchanged, globally
graded, resized and recompressed copies. Fractionally shifted, translated
and unrelated copies test rejection. The exact truth is calculated before
the final JPEG encode from the changed pixels after applying the same
global grade to the original and the locally altered source. A useful
location needs 0.30 intersection over union and 0.50 changed-pixel recall.
Conventional local modifications also count as real differences. These
photos had earlier project exposure under separate methods. [Source and rights records](image-jpeg-texture-rank-transfer-2026-11-12.md)
· [Separate LaMa panel](image-masked-context-new-editors-2026-11-13.md)

## Initial comparison

| Panel | Check | Useful sharp edits | Useful feathered edits | Ordinary delivery highlights | Ordinary groups highlighted |
| --- | --- | ---: | ---: | ---: | ---: |
| Main | Current paired route | 1/7 | 1/7 | 11/28 | 6/7 |
| Main | Cross-checked global tone | 7/7 | 7/7 | 10/28 | 6/7 |
| Held | Current paired route | 1/6 | 1/6 | 9/24 | 4/6 |
| Held | Cross-checked global tone | 5/6 | 5/6 | 14/24 | 5/6 |
| Held | Use the new check only when current is unassessed | 3/6 | 3/6 | 13/24 | 5/6 |

Both methods rejected all 39 incompatible-framing pairs. The new method
also localized five of six matched ordinary tone and copy edits on the
held panel. Graded resize round trips accounted for three of the four
additional nuisance highlights in the fallback-only readout. Because
those resized controls can produce extensive differences, the initial
comparison supports an additional delivery scope check before app
enablement. The exact run and replay each reconstructed all 156 delivered
pairs and the selected masks. [Initial replay receipt](evaluation-panels/image-reference-monotone-jpeg-transfer-2026-11-13.json)

## Separate sharpness confirmation

The extra scope check compares fine-to-coarse luma energy over
corresponding detailed 64-pixel tiles after simulating the accepted
whole-image tone and codec. It withholds the fallback if both modeled
deliveries show widespread fine-detail loss in the submitted copy, or
there are too few detailed tiles to check. The ordinary controls include
two resize kernels and whole-image blur. The point was chosen using the
initial main bank and fixed before scoring eight separate v225 source
photographs. The retained publisher originals, edits and exact masks
remain local. [Source qualification and photo rights](image-bg-real-transport-v225-result.md)

| Check | Useful sharp edits | Useful feathered edits | Ordinary delivery highlights | New ordinary files versus current | Incompatible pairs compared |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current paired route | 1/8 | 1/8 | 20/48 | 0 | 0/24 |
| Unguarded fallback | 4/8 | 4/8 | 36/48 | 16 | 0/24 |
| Sharpness-checked fallback | 4/8 | 4/8 | 26/48 | 6 | 0/24 |

The sharpness check retained all six useful additions and rejected ten of
the new ordinary highlights. It added no newly flagged source group.
Four ordinary graded images and two Lanczos resize round trips still
produced new regions, exceeding the frozen limit of two files. Two
of those highlights covered more than 115,000 pixels each. The fallback
stays experimental. The paired improvement on known removals justifies
more precise handling of residuals from ordinary delivery changes.
[Confirmation and exact replay receipt](evaluation-panels/image-reference-monotone-jpeg-confirmation-2026-11-13.json)

## Reproduce

Both runs use the installed Python environment without new detector
weights. Use the [earlier tile trial](image-regional-tile-rescue-2026-11-13.md)
inputs for the initial evaluation and the retained v225 qualification and
its original assets for the second bank. Use empty private output
directories. Each review independently reconstructs the JPEG bytes,
changed-pixel masks, comparison states and every selected pixel mask.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_monotone_jpeg_transfer "$phase" \
    --tile-run "$tile_run" --archive "$archive" --prior-probe "$probe" \
    --residual "$residual" --early-f5 "$early_f5" --later-f5 "$later_f5" \
    --checkpoint "$checkpoint" --output-fresh "$fresh_transfer" \
    --external "$new_editor_transfer" --output "$run" --publication "$receipt" || exit 1
done
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_reference_monotone_jpeg_confirmation "$phase" \
    --source-roster "$v225" --later-f5 "$later_f5" --external "$new_editor_transfer" \
    --output "$confirmation_run" --publication "$confirmation_receipt" || exit 1
done
```

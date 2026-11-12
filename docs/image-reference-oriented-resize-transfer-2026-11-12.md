# Earlier-copy image evidence after rotation, cropping and resizing

The combined check recovered 42 earlier-file matches that the current route left unassessed, including 20 useful local edit locations. It made zero claims on 173 control pairs. The check takes a separately supplied earlier file and a single-frame RGB PNG that has been oriented, cropped and resized. Its evidence carries zero weight in AI confidence. [Existing earlier-copy inspection](image-reference-comparison-2026-11-11.md) · [Frozen comparison](evaluation-panels/image-reference-oriented-resize-transfer-2026-11-12.json)

## Method

Consider the seven right-angle rotations or reflections and the identity. In each orientation, the existing feature matcher and bounded search propose an earlier integer rectangle. The strongest eight proposals are tested with the existing bicubic, bilinear, Lanczos and area interpolation methods. The crop must cover at least 55% of the earlier area. The supported earlier and submitted dimensions stay within the existing exact PNG resize scope, including the longest-side limits of 768 and 845 pixels respectively and a nominal proportional scale from 0.72× to 1.17× after rounding.

A match needs exact equality in at least 65% of submitted RGB pixels and at least 12 complete textured tiles, covering four quadrants and four rows and columns. The bounding rectangle of all mismatches can cover at most 40% of the submitted grid. The method declines the new proof if the identity passes or if more than one distinct orientation, rectangle or RGB projection passes among the searched proposals. Two resize methods may share a proof only when they reproduce identical RGB bytes. The submitted crop is shown on the submitted grid. The earlier crop is mapped outward through the resize ratios and inverse orientation, then read directly from the earlier image. [Frozen exact PNG crop and resize](image-png-exact-crop-resize-2026-11-11.md) · [Frozen orientation and native coordinates](image-reference-oriented-crop-2026-11-12.md) · [Candidate code](../backend/scripts/image_reference_oriented_resize_candidate.py)

## Source-qualified panel

The study uses 21 attributed Wikimedia Commons photographs and seven separate BG-REAL photo groups with source-preserving SDXL or Kandinsky inserts. Reference photographs and generated patches were previously qualified in unrelated work. All transformed deliveries are new to this study. The Commons photos include both earlier PNG and JPEG files, and edit cases apply local blur and brightness changes after transformation. The generated cases transform both the original and the corresponding verified edited RGB image. The target is the exact changed-pixel mask on the delivered grid after interpolation. Tests cover all seven orientations, four resize methods and four authored scale factors. [Commons source attribution and rights](image-affine-native-scale-2026-11-11.md) · [Generated source attribution and exact patch qualification](image-residual-patch-transfer-2026-11-11.md)

The negative cases include fractional shifts, small off-angle rotations, anisotropic resizing, JPEG delivery, global tone changes, unrelated photos, and identity crops. Twelve additional pairs use two separately rendered ordinary documents in the same Chromium template. They share interface structure and have different record content. The second document in each pair is oriented, cropped and resized before comparison. The synthetic interfaces use project-authored text and CSS with no external assets. [Capture runner](../backend/scripts/capture_image_reference_oriented_resize.cjs)

Each correctly reported earlier crop must have the exact authored orientation, rectangle and resize method. A useful local evidence mask must cover at least 50% of changed pixels and reach 0.30 intersection over union, with no selected pixel outside the actual difference. A clean match must reproduce the entire submitted grid. Advancement requires 16 clean matches, 10 useful ordinary local edits, three useful generated edits, at least one proof per orientation, 18 correct comparisons that the earlier serving route left unassessed, and zero incorrect accepted claims. The code, sources, treatments and criteria were frozen before scoring.

## Held results

| Delivery | Current route comparable | Correct new comparisons |
| --- | ---: | ---: |
| Unedited transformed Commons photos | 0/21 | 17/21 |
| Transformed Commons photos with a local edit | 0/21 | 17/21 |
| Unedited transformed BG-REAL photos | 0/7 | 5/7 |
| Transformed BG-REAL photos with a generated edit | 0/7 | 3/7 |
| Ordinary control pairs | 7/173 | 0/173 |

The seven existing control comparisons were unrotated identity controls. The new check abstained on those, the 12 separately rendered document pairs and all other controls. Correct proofs covered all seven orientations and all four resize methods. The selected masks exactly covered all changed pixels in 18 of 20 matched edits. The other two omitted a small number of changed pixels outside the reported connected components. Across all 20, every selected pixel was a true RGB difference on the submitted grid. These findings measure the portion shared by the two supplied copies. The earlier file's provenance and the cause of a difference remain separate questions.

The native readback verified all 229 delivered file pairs, rebuilt the photo deliveries from their source records, and reconstructed all 42 accepted RGB projections without rerunning proposal selection. It independently checked connected components, the submitted mask and 20 native crop pairs. The original freeze is `0b38ec554f1ccc5713d27d87d4f6436ae11984c63a0795fa006131256fc88ce9`. The independent crop digest is `b5447d82c1e1652a8d7f5219b3428b8a46229646b29d564f5488893bbab2a9cb`. Source photos and derived image bytes stay local.

## App verification

The local earlier-copy endpoint tries the combined transformation when the existing paths leave a pair unassessed. A serving replay reproduced all 42 exact proofs and preserved the 187 other outcomes in the frozen panel. A simulated unpaid Stripe event queued no analysis. A paid test event released the actual worker, and a local Chrome session submitted the earlier file to the live isolated API in the plain analysis view. For the generated edit used in that test, the browser, comparison PDF and JSON showed the same native crop positions and 21,372 true changed pixels. The offline reader independently verified both source hashes, both crops and the overlay, and rejected altered crop coordinates in the HTTP regression test. [Local earlier-copy usage](image-reference-comparison-2026-11-11.md)

## Reproduce

The portable reader verifies the frozen input roster, coordinate bounds, proposal witnesses, and decision ledger without the original media.

```sh
python3 -I -S backend/scripts/review_image_reference_oriented_resize_transfer.py \
  docs/evaluation-panels/image-reference-oriented-resize-transfer-2026-11-12.json
```

To reproduce the original frozen study, use its pinned code and incumbent at commit `88edc830f6a156cd18b02a0a27d6c6cbc0eef4a9`. The native replay uses the retained 38-photo Commons archive, the seven-source BG-REAL qualified archive, and local Chromium. The capture command blocks browser network requests. Use new local directories for `ui` and `run`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
node backend/scripts/capture_image_reference_oriented_resize.cjs "$ui"
for phase in freeze run; do
  backend/.venv/bin/python -B -m scripts.image_reference_oriented_resize_transfer \
    "$phase" "$commons" "$generated" "$ui" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_reference_oriented_resize_transfer \
  "$run/result.json" --native "$run" "$commons" "$generated" "$ui"
```

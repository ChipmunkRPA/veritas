# Earlier-copy image comparison after rotation or reflection

An earlier-copy comparison can now align a rotated or mirrored PNG and show the exact changed pixels. It recovered all changed pixels on 27 of 38 local edits and reproduced all pixels on 27 of 38 unchanged transformed copies. The new check made zero orientation claims on 152 incompatible controls. [Frozen comparison and audit](evaluation-panels/image-reference-orientation-transfer-2026-11-12.json)

## Fixed scope

The local app first corrects each file's stored EXIF orientation. It then checks the seven non-identity right-angle rotations or reflections of the earlier RGB pixels, plus the untransformed earlier image as an ambiguity control. It requires a unique matching orientation, exact matches on at least 65% of the submitted grid and at least 12 textured tiles spread across all four quadrants and at least four rows and four columns of the fixed tile grid. Differences must fit inside a compact area covering at most 40% of the submitted grid. All selected pixels differ after the matching transform. Each displayed crop uses its own file's native coordinates and orientation.

This check covers single-frame RGB PNG submissions from 160 pixels on the shorter side up to four megapixels. The earlier PNG or JPEG can supply the exact decoded RGB values. Lossy re-encoding, non-right-angle rotation, crop plus rotation, and combined resizing are outside this exact orientation proof. A symmetric image can have more than one matching orientation. The new check abstains when the evidence does not distinguish them. File differences carry zero AI confidence weight.

## Licensed photo transfer

The test reused 38 Wikimedia Commons photos from the large-image comparison. Their credited contributors and source file digests are distinct. The orientation and edit deliveries are new for this comparison. The source JPEGs supplied evaluation copies at alternating long edges of 768 and 1,536 pixels, with both PNG and JPEG90 earlier copies. Each group used one of the seven fixed orientations. [Source rights and acquisition](image-affine-native-scale-2026-11-11.md)

| Submitted condition | Exact orientation proofs | Correct outcomes among accepted |
| --- | ---: | ---: |
| Unchanged after rotation or reflection | 27/38 | 27/27 |
| Known local edit after rotation or reflection | 27/38 | 27/27 |
| Additional 1.3-degree rotation | 0/38 | — |
| Whole-image tone change | 0/38 | — |
| Unrelated photo | 0/38 | — |
| JPEG90 delivery of the transformed photo | 0/38 | — |

All seven orientations produced a correct local edit result. An independent Pillow replay reconstructed all 228 deliveries and rechecked the orientation proof and exact mismatch mask on all 54 accepted pairs. It also verified the inverse coordinates of each retained crop. A descriptive follow-up ran the earlier app comparator on one of the newly recovered edits per orientation. It left all seven unassessed. The control counts describe admission to the exact orientation check. Other comparison paths can still measure supported ordinary differences.

An isolated paid-flow test used one of the real photo deliveries. Simulated USD 9.99 checkout and an unpaid event queued zero analyses. Confirmed test payment released one analysis, which the CPU worker completed with the recompression check. The authenticated comparison retained the exact 8,891 changed pixels. The rendered PDF and the JSON contained the same two native crops and overlay. The offline reader independently verified the original files and counted the same mask pixels.

## Reproduce

Use the 38 existing bound photos and the corresponding `plan.json` and `acquisition.json` as `archive`. Use a new private directory as `run`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze run verify; do
  backend/.venv/bin/python -B -m scripts.image_reference_orientation_transfer "$phase" "$archive" "$run" || exit 1
done
```

To verify an orientation comparison JSON downloaded from the app, use Python with Pillow and supply both original files. The reader checks the original hashes, unique orientation, exact match counts, source crops, mask support, and overlay pixels.

```sh
backend/.venv/bin/python -I backend/scripts/review_image_reference_export.py comparison.json earlier.jpg submitted.png
```

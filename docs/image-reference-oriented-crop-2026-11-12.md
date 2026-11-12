# Earlier-copy comparison after rotation and cropping

The exact comparison found 22 of 28 known local edits in PNGs that were both rotated or reflected and cropped. It also verified 22 of 28 unedited transformed crops. The current earlier-copy route left all 56 pairs unresolved. The new method made zero claims across 224 other deliveries. These are source-pixel comparisons and carry zero AI confidence weight. [Audited case ledger](evaluation-panels/image-reference-oriented-crop-2026-11-12.json)

| Held outcome | Current comparison | Oriented crop prototype |
| --- | ---: | ---: |
| Unedited transformed crop, fully matched | 0/28 | 22/28 |
| Local edit, correct mask and earlier coordinates | 0/28 | 22/28 |
| Claimed orientation on controls | Outside this comparison | 0/224 |

## Frozen scope

The prototype accepts one single-frame RGB PNG and an earlier decoded image up to 1,024 pixels on the longest side. The submitted dimensions are at least 160 pixels per side and the proposed crop retains 55–90% of the earlier image area. It searches all seven right-angle orientations and the identity at integer pixel offsets. Five exact RGB samples per textured tile propose crop placements. Every accepted placement then requires at least 12 whole tiles with exact RGB equality, spread over four quadrants and at least four rows and four columns. At least 65% of the complete submitted crop must match. Differences must fit inside a compact box no larger than 40% of the crop. The search abstains if more than 32 offsets in any orientation require verification, if the identity is a plausible match or if multiple distinct placements satisfy the full proof.

The earlier-image coordinates are recovered by reversing the recorded orientation on the complete earlier pixel grid. The mask and the submitted crop use the submitted native grid. The comparison describes differences in the portion visible in both files. The earlier file's authenticity and the origin of an edit require separate evidence.

## Held comparison

The roster draws 28 reference photos and one unrelated donor from a prior credited Wikimedia Commons source inventory. The photos had earlier project exposure. The exact crop and edit deliveries are new for this trial. Four source groups cover each orientation. Half of the earlier copies are PNG and half JPEG. The projected PNGs take their exact values from the decoded earlier file. A deliberately compact edit changes a known set of RGB pixels after orientation and crop. The comparison recovered exact masks in all seven orientations. [Source rights, credited photographers and exact bytes](image-affine-native-scale-2026-11-11.md)

The eight control roles cover an identity crop, a slight angle change, global tone adjustment, JPEG re-encoding, an unrelated photo, a crop resized with interpolation, two widely separated changes, and a centered crop of a deliberately symmetric reference. Each role supplies 28 views. The 0/224 figure applies to the new exact-oriented-crop proof. Existing comparison methods can still measure supported differences in ordinary delivery paths. The app was replayed as the incumbent on all 56 positive pairs and returned either unsupported resize geometry or unresolved same framing.

The independent native audit used complete row-byte searches and Pillow operations to recover all 44 unique matches across all eight orientations. It checked all 280 encoded pairs, 935 exact tile witnesses and the source coordinates and pixels of all 22 paired edit crops. No media or crop pixels are redistributed. The public replay checks the decision ledger with the Python standard library.

## Reproduce

```sh
python3 -B -I -S backend/scripts/review_image_reference_oriented_crop_transfer.py \
  docs/evaluation-panels/image-reference-oriented-crop-2026-11-12.json
```

Use the exact 38-source `$archive` from the credited Wikimedia Commons roster and a new `$run` directory for native reproduction.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_reference_oriented_crop_transfer freeze "$archive" "$run"
backend/.venv/bin/python -B -m scripts.image_reference_oriented_crop_transfer run "$archive" "$run"
backend/.venv/bin/python -B backend/scripts/review_image_reference_oriented_crop_transfer.py "$run"
```

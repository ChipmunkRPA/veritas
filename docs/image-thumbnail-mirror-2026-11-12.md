# Compare embedded JPEG previews across mirrored EXIF orientations

The embedded-preview comparison now supports EXIF orientations 2, 4, 5 and 7. It checks the preview on the JPEG's stored axes and on the image's displayed axes. It reports a local difference when exactly one convention matches under both resize settings. The main crop follows the displayed image. The second crop uses the original stored preview coordinates. The comparison carries zero AI authorship weight. [Audited case ledger](evaluation-panels/image-thumbnail-mirror-2026-11-12.json)

## Held comparisons

The 12 feathered edit sources also supplied the earlier rotation test. Six source-bound 64 × 64 generated composites from a separate local construction added small edits with independently known masks. Each of the 18 sources supplies one upright comparison, eight mirrored edit views, 16 mirrored unedited or global-tone views, four missing previews and four unrelated previews. Five genuine photographs supply 40 additional mirrored clean views using their original embedded previews or re-encoded mirrored previews. These are controlled JPEG deliveries. The source photos have earlier project exposure. [Rotation source panel](image-thumbnail-orientation-2026-11-12.md) · [Compact edit qualification and rights](image-flux-photo-expansion-v285-method.md) · [Original preview sources](image-thumbnail-localization-2026-11-12.md)

| Main JPEG orientation | Preview stored on main axes, useful feathered edits | Preview already on display axes, useful feathered edits |
| --- | ---: | ---: |
| 2 · horizontal reflection | 11/12 | 11/12 |
| 4 · vertical reflection | 11/12 | 11/12 |
| 5 · transpose | 11/12 | 11/12 |
| 7 · transverse reflection | 11/12 | 11/12 |

A useful location covers at least half the known changed pixels and reaches 0.30 intersection over union. Each mirror and preview-storage combination retained all 11 useful upright feathered locations. None of the six compact edits met that rule upright or mirrored. Thirty of 48 mirrored compact edit views produced a region with partial overlap below the useful-location rule. No local region appeared on the 328 clean views, and none of the 144 missing or unrelated preview views were accepted. Repeated deliveries of each source share the same source group.

The independent check maps pixel indices through the declared display transform and compares their exact sets with each stored preview crop. The runtime matched the frozen decisions and masks across 634 deliveries, including 119 independently checked crop locations. The changes are confined to preview evidence. The existing metadata comparison retains its own result and can respond to an orientation difference.

## App verification

Three isolated Full worker analyses used simulated checkout. Two edits using different preview conventions showed the same 96 × 38 main crop at (16, 293), with the expected different stored preview bounds. The clean reflected photo showed no local difference. The JSON and both PDF profiles contained the same exact crop pixels. Both signed ZIPs verified, including the Standard PDF that contains those crops. Chrome displayed both crops with their source coordinates in the plain view.

## Replay

The scalar reviewer uses the Python standard library and the public case ledger. Native replay needs the retained original and edited photo archives. Source images and crop pixels stay local.

```sh
python3 -I -S backend/scripts/review_image_thumbnail_mirror_transfer.py \
  docs/evaluation-panels/image-thumbnail-mirror-2026-11-12.json

export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze held runtime; do
  backend/.venv/bin/python -B -m scripts.image_thumbnail_mirror_transfer "$stage" "$run" || exit 1
done
```

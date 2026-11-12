# Compare embedded JPEG previews across EXIF rotations

Chipmunk Verify can now show embedded-preview differences on JPEGs whose display orientation calls for a 180°, 90° clockwise, or 90° counterclockwise rotation. The held test located 11 of 12 known edits in every rotation and preview-storage combination, matching the upright comparison. Across all combinations, it highlighted none of 102 clean views and accepted none of 72 missing or unrelated previews. [Audited result](evaluation-panels/image-thumbnail-orientation-2026-11-12.json)

The comparison considers an embedded preview on the same stored axes as the main JPEG and a preview already on the displayed axes. Both resize settings must corroborate exactly one orientation before the app reports a region. Main image crop coordinates refer to the displayed orientation. The preview crop is inverse-mapped to its stored pixels after covering both subpixel registrations. A separate qualification covers orientations that need mirroring. The comparison uses the preview's resolution, and the evidence carries zero AI authorship weight. [Upright method and source terms](image-thumbnail-localization-2026-11-12.md) · [Mirrored orientation qualification](image-thumbnail-mirror-2026-11-12.md)

| Main JPEG orientation | Same stored axes, edits located /12 | Preview already upright, edits located /12 |
| --- | ---: | ---: |
| 3 · 180° | 11 | 11 |
| 6 · 90° clockwise | 11 | 11 |
| 8 · 90° counterclockwise | 11 | 11 |

The 12 generated-edit source groups were excluded from the earlier thumbnail qualification and appeared in unrelated project studies. Each supplies an upright comparator and two preview conventions for each rotation. The 102 clean views use 12 unedited source photos and five genuine published photographs, all encoded into authored rotation deliveries. The 258-case native replay compared the independent measurement with runtime behavior and checked all 66 selected stored-preview boxes against exact pixel-index mappings. A separate 177-case replay reproduced every previously published upright mask, location, and decision. The units are matched views of the same 17 source groups, so the counts do not estimate customer false-alarm rates.

The normal image view, Standard and Summary PDF reports, and the JSON export carry paired crop locations. The JSON records `main_exif_orientation`, `preview_alignment_orientation`, and both coordinate spaces. A preview alignment orientation of zero means the stored preview required no turn. Older saved upright exhibits retain compatible coordinates and crop references.

Three Full worker runs used simulated checkout. The rotated edit and its matched upright delivery reported the same main-image bounds at (16, 293), 96 × 38 pixels. Their stored-preview crops used their respective original axes. Both Standard and Summary PDFs contained the exact two crop images from each JSON export. The genuine rotated control produced no highlighted preview difference.

## Replay

Use the retained primary and backup photo rosters, generated archive and genuine originals listed by the earlier thumbnail trial. Set `run` to a new private directory.

```sh
set -e
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for stage in freeze held; do
  backend/.venv/bin/python -B -m scripts.image_thumbnail_orientation_transfer "$stage" "$run"
done
```

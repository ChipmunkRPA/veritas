# Compare two matching photo areas in one submitted image

This pilot finds matching textured areas within a single PNG, then maps the exact differing pixels into both areas. It covers screenshots and collages that display two versions of the same photo. The evidence describes an undirected difference between the two displayed areas. It contributes zero AI authorship confidence. The frozen pilot missed its six-additions-per-edit-type gate. [Audited scalar ledger](evaluation-panels/image-embedded-pair-difference-transfer-2026-11-12.json)

## Frozen measurement

Search for matching native 16 × 16 texture tiles at separate locations in the submitted image. A sampled hash proposes a tile correspondence. The check then compares every RGB pixel in each proposed tile. A single supported translation needs at least 12 verified witnesses across six cells of a three-by-three grid and two regions separated by at least 24 pixels. The full aligned comparison requires at least 85% identical RGB pixels. Any difference must occupy a bounded interior area with an eight-pixel support margin and at least one connected component of 48 pixels. Exact duplicates return a comparison with zero differences. Candidate rejection leaves the previous boundary result available.

The held panel has eight previously qualified LaMa source groups and four distinct SDXL and Kandinsky source groups. Each group contributes a sharp edit, a feathered edit and an ordinary local adjustment in the second displayed pane. Controls include identical duplicates, global tone changes, unrelated photos, a shifted view of the same photo and ordinary repeated interfaces. Eight additional credited photographs provide standalone controls. The source photos have earlier project exposure. The submitted pane layouts and exact truth masks are new. [Qualified LaMa sources](image-bg-real-backup-intake-v262-method.md) · [Separate generated sources](image-residual-patch-transfer-2026-11-11.md) · [Ordinary source credits](image-affine-native-scale-2026-11-11.md)

The existing sharp boundary comparator counts a useful region when its selected rectangle reaches 0.30 IoU and 0.50 recall on either pane. The candidate requires pixel-for-pixel recovery of the full known difference in both panes. The predeclared gate requires six new exact sharp locations, six new exact feathered locations and zero newly flagged ordinary groups. The ordinary local adjustment has a real difference mask and is measured separately from the controls that should produce no reported difference.

## Held results

| Held view | Sharp boundary useful | Exact difference in both panes | New exact evidence |
| --- | ---: | ---: | ---: |
| Sharp known edit | 1/12 | 5/12 | 4/12 |
| Feathered known edit | 0/12 | 5/12 | 5/12 |
| Ordinary local adjustment | 0/12 | 12/12 | 12/12 |
| Ordinary no-difference controls | — | 0/68 flagged | 0 groups |

All 12 duplicates and all 12 shifted views produced a matching comparison with zero reported local differences. Unrelated, global-tone, repeated-interface and standalone control inputs produced no difference record. The exact proof added useful evidence on accepted pairs and produced no false difference in this panel. A truth-only post-run check found the same blockers for each edit type. Four known edits came within eight pixels of the displayed pane edge, and three changed more than 15% of the full pane. Both conditions fall outside the fixed support rules. The comparison is limited to areas with distributed exact witnesses in the single submitted PNG.

The independent readback verified all 104 source views, 46 aligned comparisons, 37,437 native texture witnesses, 22 exact two-sided masks and 104 original four-side boundary measurements. The freeze is `0ff4694f313a8389453ba57ccaabb3a7b3eea69a1a3d856d0fe7a4b8674666b7`. The app’s image checks stay in use.

## Reproduce

```sh
python3 -I -S backend/scripts/review_image_embedded_pair_difference_transfer.py \
  docs/evaluation-panels/image-embedded-pair-difference-transfer-2026-11-12.json
```

Run the native worker with the retained Commons, v262 LaMa and seven-group generated archives. Use a fresh private output directory.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for stage in freeze run; do
  backend/.venv/bin/python -B -m scripts.image_embedded_pair_difference_transfer "$stage" \
    "$commons" "$lama" "$generated" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_embedded_pair_difference_transfer "$run/public.json" --native \
  "$commons" "$lama" "$generated" "$run"
```

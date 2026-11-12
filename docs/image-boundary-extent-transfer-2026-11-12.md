# Small sharp edit boundaries in a single PNG

A flexible rectangle search localized 23 of 31 small sharp generated inserts, compared with zero for the current fixed-size check. It retained only six of the current check's 17 locations on the larger inserts. The source-separated follow-up kept the fixed-size findings and added nine conventional paste locations, with zero additional ordinary flags. It recovered none of the six separately generated edits. Both candidates remain experimental. [Audited results and public case ledger](evaluation-panels/image-boundary-extent-transfer-2026-11-12.json)

## Flexible boundary search

The current check measures the four sides of a rectangle and searches three fixed widths and heights. The candidate proposes more sizes and positions from the same native RGB step response. It broadens the edge response only across the direction perpendicular to each side during proposal search. The two passes of refinement then return to the unmodified one-pixel response and move each side within nine pixels of its original proposal. One arm ranks the refined rectangle by the geometric mean of its four side averages. The other splits each side into four sections and uses the second-lowest section mean before combining the sides. No checkpoint or earlier copy is used at inference.

Each candidate's internal review point was the second-largest ordinary maximum from eight separate calibrated source groups. The held panel reused 31 attributed photos from SDXL and Kandinsky studies. It added two smaller edit sizes and matched processing and blur controls. The new sizes came from a fixed subrectangle of each already verified edit region. The exact pre-delivery RGB difference defines the target. Another 13 Wikimedia Commons photos served as ordinary source controls. A useful location covers at least half of the target and has intersection over union of at least 0.30. The photos had earlier project exposure. [Source rights and established edit regions](image-residual-patch-transfer-2026-11-11.md) · [Ordinary photo roster](image-affine-native-scale-2026-11-11.md)

| Outcome | Current fixed | Flexible means | Distributed side support |
| --- | ---: | ---: | ---: |
| Larger sharp generated edits localized | 17/31 | 6/31 | 5/31 |
| Smaller sharp generated edits localized | 0/31 | 23/31 | 22/31 |
| Larger feathered edits localized | 1/31 | 0/31 | 0/31 |
| Smaller feathered edits localized | 0/31 | 0/31 | 0/31 |
| Ordinary Commons photos flagged | 3/13 | 5/13 | 4/13 |

Across the 248 paired ordinary variants, the flexible check had the same number of flags as the current check. Distributed support had one fewer. The source transitions were different. The separate larger-edit preservation requirement failed for both candidates.

## Prospective fallback test

After the first study, the fallback kept every incumbent region. It considered a separate distributed-support rectangle only when the incumbent selected no region. Its two side lengths had to be at least 56 pixels and at most 18% of their respective image dimensions. The size rule was chosen using the first study and fixed before this second test. The distributed operating point carried over without refitting.

The second panel used the 25 remaining Commons source slots and six different licensed photos with previously published FLUX edits. It used new exact paste locations for the Commons sources. For the generated edits it placed the verified 256-pixel composite back into its original larger photo, verified the unedited parent crop and counted the exact changed pixels. The evaluated PNGs remained in the current check's native size band. Each group supplied sharp and feathered edit deliveries and four ordinary deliveries. [Qualified generated sources and weights](image-flux-photo-expansion-v285-result.md)

| Outcome | Current fixed | With fallback |
| --- | ---: | ---: |
| Commons sharp conventional pastes localized | 1/25 | 10/25 |
| Commons feathered conventional pastes localized | 0/25 | 0/25 |
| Sharp generated edits localized | 0/6 | 0/6 |
| Feathered generated edits localized | 0/6 | 0/6 |
| Ordinary views flagged | 32/124 | 32/124 |

The combined check preserved the fixed-size locations. It fell short of both held gain requirements. This result measures local RGB boundary evidence for specific paste constructions. It supplies no AI origin probability.

## Verify and reproduce

The native replays matched all 385 first-study and 186 second-study input and truth hashes, selected boxes, and outcomes. The public ledger exposes per-case response and integer overlap counts. The standalone Python checker rebuilds all 1,155 first-study decisions and the 186 fallback decisions without app dependencies, images or model weights.

```sh
python3 -I -S backend/scripts/review_image_boundary_extent_transfer.py \
  docs/evaluation-panels/image-boundary-extent-transfer-2026-11-12.json
```

The native runners need the retained BG-REAL trial folders, Wikimedia inputs and the six verified generation outputs. `source_cache` is the retained v39 encoded photo cache. Supply new output directories as `selection` and `holdout`.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for phase in freeze fit holdout verify; do
  backend/.venv/bin/python -B -m scripts.image_boundary_extent_transfer "$phase" \
    "$calibration" "$earlier" "$later" "$newest" "$commons" "$selection" || exit 1
done
for phase in freeze run verify; do
  backend/.venv/bin/python -B -m scripts.image_small_boundary_fallback_transfer "$phase" \
    "$selection" "$commons" "$flux" "$source_cache" "$holdout" || exit 1
done
```

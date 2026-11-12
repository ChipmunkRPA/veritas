# Interior flatness in small image-boundary checks

A frozen rule removed all four added flags on flat interface panels. It retained all seven on translucent or patterned panels across four groups and suppressed one correctly localized low-texture conventional paste. This separate prospective test rejects the interior flatness rule for serving. [Audited source roster and case ledger](evaluation-panels/image-small-boundary-interior-transfer-2026-11-12.json)

| Held outcome | Current check | With fallback | With fallback and interior rule |
| --- | ---: | ---: | ---: |
| Low-texture paste, correct location | 0/9 | 7/9 | 6/9 |
| Normal-texture paste, correct location | 0/9 | 5/9 | 5/9 |
| Ordinary photo control with any box | 3/36 | 3/36 | 3/36 |
| Flat-panel screenshots in the incumbent-negative stratum | 0/4 | 4/4 | 0/4 |
| Translucent-panel screenshots in the stratum | 0/2 | 2/2 | 2/2 |
| Patterned-panel screenshots in the stratum | 0/5 | 5/5 | 5/5 |

Every retained translucent or patterned flag had zero constant interior blocks. The removed low-texture paste was a four-color quantized photographic patch with a 0.429 interior flat-block share. Flatness cannot distinguish these textured interface panels from the added normal-texture photo locations, whose interior shares were also zero.

## Frozen comparison

An earlier screenshot study showed that whole-file flatness let sharp interface panels through when most of the image was a photo. A descriptive follow-up on those already scored results found a gap in flatness inside the new candidate boxes. This study froze the resulting 0.35 hypothesis on new sources before obtaining candidate outputs. It reuses the strict boundary score of 1.2805722063718288 and the same 56-pixel minimum and 18% maximum per axis. It preserves the existing result when one is available. On each newly selected box, remove four pixels on every side and skip the box when at least 35% of the remaining 8 × 8 RGB blocks are constant in each channel. A useful location covers at least half the exact changed pixels with 0.30 intersection over union. [Earlier screenshot comparison](image-small-boundary-screenshot-admission-2026-11-12.md)

The roster contains nine target photos, nine distinct donors and nine separate screenshot background photos from distinct recorded cache components and credited photographers. They have commercially reusable CC BY-SA 2.0 terms. The photo sources were outside the earlier boundary panels and may have had unrelated copy-proposal exposure. Each donor supplied its lowest mean-gradient window and a window at the 60th percentile of that donor's measured windows. Five low-texture patches were also quantized to four colors to challenge the rule with broad nearly constant areas. Both pastes use the same exact target coordinates on their matched source. The 36 ordinary photo controls are unchanged evaluation sources, global tone changes, feathered local blur and resize round trips. [Source terms and cache scope](image-fresh-source-cache-v39-results.md)

Nine distinct local Chromium layout groups overlaid flat, translucent or patterned interface panels on credited photos. Each group supplied two exact native viewport captures. The fixed check ran first. It flagged seven of 18 views, leaving 11 views from six groups across all three types for the sealed comparison. The gate required broad challenges on at least two panel types and three low-texture paste locations. It also required no added screenshot or ordinary photo groups and no loss of correct low-texture locations. The screen captures are authored controls, so the result gives no prevalence estimate for customer uploads. No photo, screenshot or crop pixels are redistributed. The method measures conventional boundaries and carries zero AI confidence weight.

## Verify and reproduce

The independent reader checked 18 native capture hashes, 65 scored PNGs, 23 exact interior measurements and 37 source crop records. The public replay uses only the Python standard library.

```sh
python3 -B -I -S backend/scripts/review_image_small_boundary_interior_transfer.py \
  docs/evaluation-panels/image-small-boundary-interior-transfer-2026-11-12.json
```

Use the frozen `roster` from the ledger and place the exact attributed JPEGs in `$cache/<encoded_sha256>.bin`. The capture step renders new local controls from the frozen source plan. It uses the project's Playwright installation and the Chromium version listed in the receipt.

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
node backend/scripts/capture_image_small_boundary_interiors.cjs plan "$run"
backend/.venv/bin/python -B -m scripts.image_small_boundary_interior_transfer prepare "$cache" "$roster" "$run"
node backend/scripts/capture_image_small_boundary_interiors.cjs capture "$run"
for stage in freeze incumbent candidate; do
  backend/.venv/bin/python -B -m scripts.image_small_boundary_interior_transfer "$stage" "$cache" "$roster" "$run" || exit
done
backend/.venv/bin/python -B backend/scripts/review_image_small_boundary_interior_transfer.py "$run"
```

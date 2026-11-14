# Native tile follow-up for JPEG object-removal evidence

The optional regional check already found some small JPEG edits. This experiment tested whether a smaller native-resolution context could recover missed object removals. It ran the same reviewed regional checkpoint on overlapping 512-pixel source tiles when the whole-image pass had no selected region. It used the complete JPEG's reliability mask, removed a 24-pixel margin beside artificial crop edges, and required half of both components to agree after shifting the tile by up to 32 pixels. It retained only the strongest supported region in native coordinates. The local selection rules and checkpoint were fixed before scoring. [Previous JPEG transfer](image-regional-jpeg-transfer-2026-11-13.md) · [November inpainting study](https://arxiv.org/abs/2602.00192v2)

## Completed main bank

The 136 JPEG75 4:2:0 deliveries contain 15 separate photos, seven LaMa object removals and eight Stable Diffusion 1.5 insertions. Each contributes a sharp and a feathered version. The 106 ordinary controls include original photos, same-site blur, tone and copy, resizing and intermediate JPEG encoding, and available local and global autoencoder reconstructions. One LaMa photo was excluded from this bank before scoring because its short side was below the app's 512-pixel scope. The delivered photos had earlier project exposure under other methods. The study uses exact changed pixels before final JPEG encoding. A useful region requires at least 30% intersection over union and 50% changed-pixel recovery. [Source and rights records](image-jpeg-texture-rank-transfer-2026-11-12.md)

| Check | Useful sharp edits | Useful feathered edits | Ordinary highlights | Ordinary groups highlighted | Misplaced edit highlights |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned JPEG luma | 0/15 | 0/15 | 0/106 | 0/15 | 0/30 |
| Native regional | 5/15 | 5/15 | 8/106 | 3/15 | 2/30 |
| Native regional with tile fallback | 5/15 | 5/15 | 13/106 | 5/15 | 4/30 |

All useful regions occurred on Stable Diffusion insertions. Tiles recovered zero useful LaMa locations across seven sharp and seven feathered removals. The two additional LaMa highlights covered zero edited pixels despite the components passing the shifted-view agreement rule. Tiles also added five ordinary-file highlights across two newly flagged groups. The added computation was 435 crop forwards on top of the 136 whole-image forwards.

## Second bank and replay

The second bank contains 64 deliveries from six additional LaMa photos and one shared SDXL/Kandinsky photo. The two diffusion editor variants share a source group. The banks are disjoint by recorded photo ID, author account, credited photographer and original byte hash. The first worker stalled opening a local file before this bank began. After the workspace recovered, the runner matched the original freeze, verified the 136 main-bank rows and masks, and appended the missing 64 deliveries without rewriting the existing rows. [Second-bank sources and rights](image-masked-context-new-editors-2026-11-13.md) · [Historical main-bank receipt](evaluation-panels/image-regional-tile-rescue-main-bank-2026-11-13.json)

| Check | Useful sharp edits | Useful feathered edits | Ordinary highlights | Ordinary groups highlighted | Misplaced edit highlights |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aligned JPEG luma | 0/8 | 0/8 | 3/48 | 3/7 | 0/16 |
| Native regional | 1/8 | 0/8 | 5/48 | 2/7 | 2/16 |
| Native regional with tile fallback | 1/8 | 0/8 | 9/48 | 2/7 | 5/16 |

The native check localized one sharp SDXL edit. The fallback added two LaMa
highlights with zero edit overlap and one Kandinsky highlight below the
useful overlap and recall requirements. Across both banks the fallback
added zero useful locations and raised ordinary highlights from 13/154 to
22/154, across five and seven source groups respectively. Misplaced
edited-file highlights rose from four to nine. No useful LaMa locations
were found on the 26 sharp and feathered deliveries from 13 photos.
The fallback stays experimental. The test made 200 whole-image forwards
and 592 additional crop forwards. These observations carry zero
authorship-confidence weight. [Completed receipt](evaluation-panels/image-regional-tile-rescue-complete-2026-11-13.json)

The local replay rematerialized all 200 JPEGs and their exact changed-pixel
truth masks, verified all packed selected masks and recomputed every
intersection and recall value. The offline reader separately recomputes
both panels, ordinary source-group counts and the paired result from
anonymous pixel counts.

The frozen advancement criteria required at least two additional useful
feathered LaMa source groups, at most two newly highlighted ordinary
source groups and at most three additional misplaced edit highlights.
The observed counts were zero, two and five. The study kept the fixed
model response and area rules throughout.

The separate [released FLAME evaluation](image-flame-jpeg-transfer-2026-11-13.md)
ran the exact Apache-2.0 weights with the matching SAM 2.1 base on both
banks. Its ordinary-control operating point recovered no additional
object-removal location. The candidate remains experimental.

Run the standard-library reader to recompute both banks from anonymous
pixel counts and source groups. It can also read the historical main-bank
receipt.

```sh
python3 -B -IS backend/scripts/review_image_regional_tile_rescue_main.py \
  docs/evaluation-panels/image-regional-tile-rescue-complete-2026-11-13.json
```

Use the original retained inputs from the [regional JPEG study](image-regional-jpeg-transfer-2026-11-13.md#reproduce-the-study)
to repeat inference. Use a new private run directory with the frozen
runner for a full run. The recovery driver requires the exact historical
freeze and journal prefix and appends only still-missing deliveries.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for phase in freeze run review; do
  backend/.venv/bin/python -B -m scripts.image_regional_tile_rescue_transfer "$phase" \
    --archive "$archive" --prior-probe "$probe" --residual "$residual" \
    --early-f5 "$early_f5" --later-f5 "$later_f5" --checkpoint "$checkpoint" \
    --output-fresh "$fresh_transfer" --external "$new_editor_transfer" \
    --output "$run" --publication "$receipt" || exit 1
done
```

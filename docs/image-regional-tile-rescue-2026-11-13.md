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

## Coverage and replay

The planned second bank contains 64 deliveries from six additional LaMa photos and one shared SDXL/Kandinsky photo. Its input-binding operation stalled while opening a local file. The worker recorded zero second-bank inference rows. The retained main-bank ledger matched all 136 frozen row identities, and independent readback verified the SHA-256 commitments and measured area of all 408 packed region masks, mask preservation, and the recorded overlap arithmetic. Re-encoding the original assets and regenerating their truth masks for a second audit remain pending. The completed bank supports keeping the tile fallback experimental. No app behavior changed and these regions carry zero authorship-confidence weight. [Main-bank receipt](evaluation-panels/image-regional-tile-rescue-main-bank-2026-11-13.json)

The released FLAME checkpoint remains a separate comparison candidate. Its publisher licenses the exact v0.1.0 checkpoint and configuration under Apache-2.0. The separate Meta SAM 2.1 base checkpoint is Apache-2.0. The FLAME adapter contains 100 tensors and passed an isolated weights-only load. The base-model inference and detection evaluation are pending. [FLAME release](https://github.com/phoenixnir/FLAME/releases/tag/v0.1.0) · [Meta model license](https://github.com/facebookresearch/sam2#license)

Run the standard-library reader to recompute the main-bank table from
anonymous pixel counts and source groups:

```sh
python3 -B -IS backend/scripts/review_image_regional_tile_rescue_main.py \
  docs/evaluation-panels/image-regional-tile-rescue-main-bank-2026-11-13.json
```

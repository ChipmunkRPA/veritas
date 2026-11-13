# Texture-matched ranking for small JPEG edits

The prior residual encoder put its highest-ranked box at exactly the same place in 44 of 48 LaMa edit views and their unchanged photos, at both JPEG qualities. It repeated the same box in nine of twelve SDXL views at both qualities. Those were prior exposed sources used to diagnose the failure. An operating-point change cannot move the selected boxes. [Prior encoder comparison](image-residual-encoder-transfer-2026-11-12.md)

## Locked source roles and comparison

Freeze an eight-LaMa and eight-Stable Diffusion 1.5 roster from the pinned BG-REAL release before acquiring any media. Exclude prior exposure using publisher recipient, individual Flickr photo, profile, credited author, encoded and decoded hashes, and informative thumbnails. The metadata audit found 26 LaMa and 305 SD1.5 candidates after its preflight filters. Six unresolved photo or owner aliases and one unrecognized attribution URL were excluded. All selected originals resolve to individual CC BY 2.0 photo attribution. Publisher contributions are CC BY 4.0. The recorded LaMa assets use Apache 2.0 and the SD1.5 assets use CreativeML OpenRAIL-M. No editing weights or third-party detector code are used. [Publisher rights and processing controls](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution)

All sixteen photo groups qualified after acquisition. All 56 original, edited, mask and matched processing assets reproduced the pinned publisher revision and linked hashes. The eight LaMa photographs belong to the publisher training split and the eight SD1.5 photographs to its test splits. Their use here is independent of the prior fitted encoder. Exact source-pixel edit footprints occupy 0.78% to 5.39% of the native photos. Each group supplies a sharp edit, a feathered edit and six ordinary controls. SD1.5 supplies two additional matched autoencoder views. The final JPEG recipe is shared across all roles at each quality. [Matched local-edit research](https://arxiv.org/html/2607.26232v3) · [Source-preserving manipulation experiments](https://arxiv.org/html/2602.00192v2)

Freeze the encoder weights, receptive field, proposal dimensions, exact-mask usefulness criterion and ordinary calibration sources. For each decoded-image cell, measure coarse brightness, texture variation, gradient energy, direction and saturation. Identify the twelve closest coarse-texture reference cells at least nine cells away. Use the matched logit median and median absolute deviation when at least eight references meet the fixed distance limit. If the core or ring lacks support, omit that rectangle. Otherwise use the unchanged proposal geometry and compare the texture-conditioned logit map with the original logit map and current aligned JPEG luma. Calibrate each ranker's own point on the original twelve ordinary-only groups before reading held scores. This tests whether the frozen encoder's logit contains an edit-related signal after controlling for scene texture. [Prior within-image texture comparison](image-matched-detail-transfer-2026-11-12.md) · [Primary work on suppressing scene content](https://arxiv.org/abs/1808.08396) · [Primary patch-level self-similarity work](https://openaccess.thecvf.com/content/CVPR2023/papers/Guillaro_TruFor_Leveraging_All-Round_Clues_for_Trustworthy_Image_Forgery_Detection_and_CVPR_2023_paper.pdf)

## Held results

| Quality | Ranking rule | Useful sharp / feathered, out of 16 each | Useful before operating point, out of 32 | Ordinary flags /112 | Wrong edit regions |
| --- | --- | ---: | ---: | ---: | ---: |
| JPEG75 | Current luma | 0 / 0 | 0 | 0 | 0 |
| JPEG75 | Frozen encoder | 0 / 0 | 1 | 10 | 4 |
| JPEG75 | Texture-matched encoder | 1 / 1 | 2 | 9 | 6 |
| JPEG90 | Current luma | 0 / 0 | 0 | 0 | 0 |
| JPEG90 | Frozen encoder | 0 / 0 | 1 | 10 | 4 |
| JPEG90 | Texture-matched encoder | 1 / 1 | 2 | 5 | 5 |

The four useful JPEG deliveries are sharp and feathered versions of one SD1.5 photo, repeated at both qualities. The matched ranking reduced ordinary JPEG90 flags from ten to five compared with the frozen encoder. It still affected three ordinary groups at JPEG75 and two at JPEG90. The frozen encoder ranked the exact same LaMa box on the unchanged photo and the corresponding edited view for all 16 LaMa variants at both qualities. Matching texture cut this to eight variants at JPEG75 and six at JPEG90. None of the new LaMa rectangles reached the required overlap, even before the review point. An independent enumeration found that the fixed geometry allowed a useful rectangle for 30 of the 32 distinct native edit variants before texture support filtering. About 98.55% of held cells had sufficient texture references.

The advancement rule required three additional useful feathered source groups across both families, at most two additional ordinary groups and four wrong edit locations. Both qualities fell short. This candidate stays offline. The fresh test isolates a limit of rescoring the frozen residual representation. Replacing the representation would require another independent holdout.

## Local replay

The local run assessed 144 ordinary calibration deliveries and 288 held deliveries without retraining. The audit rebuilt all 432 file and exact mask bindings and independently measured 1,152 native overlaps. The raw assets, checkpoint and individual measurements remain local. [Aggregate and replay commitments](evaluation-panels/image-jpeg-texture-rank-transfer-2026-11-12.json)

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
for phase in plan acquire run; do
  backend/.venv/bin/python -B -m scripts.image_jpeg_texture_rank_transfer "$phase" \
    --archive "$archive" --encoder "$prior_encoder" --prior-f5 "$prior_f5" --output "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_jpeg_texture_rank_transfer \
  --archive "$archive" --encoder "$prior_encoder" --prior-f5 "$prior_f5" --output "$run" --publication "$receipt"
```

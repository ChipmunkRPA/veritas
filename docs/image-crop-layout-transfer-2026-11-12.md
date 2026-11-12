# Testing whether image crop findings follow the proposed pixels

The local detail check reduced ordinary photo highlights from 15 to 2 and removed all six useful edit locations from the existing experimental crop rule. It localized none of the 12 feathered edits shown inside Chromium layouts. The text-layout rule changed no selection on the frozen crop route. Both candidates remain experimental. [Audited case ledger](evaluation-panels/image-crop-layout-transfer-2026-11-12.json)

## Frozen measurements

The existing experimental crop rule reads 25 overlapping regions of one-third image width and height. It selects the strongest relative crop and applies its earlier fixed check on responses from nonoverlapping crops. The candidate keeps this proposal and adds two independently measured observations. The first blurs four disjoint 96-pixel tiles near its center or four same-size tiles at its corners on the model's 384-pixel grid. Both inputs use a 16-pixel cosine blend and Gaussian sigma 1.5. The outer-blurred response minus the inner-blurred response measures how much the candidate depends on the center relative to the corners. Separate ordinary photo groups set the strict internal point at their second-highest group maximum, with zero as the floor. The second observation checks recognized word boxes for at least three left-aligned text rows inside the proposed crop, using a minimum word-box area and vertical spread. Only geometry is retained. [Earlier crop rule](image-remote-crop-editor-transfer-2026-11-11.md) · [Source-preserving inpainting analysis](https://arxiv.org/html/2602.00192v2)

The local trial uses the pinned MIT image checkpoint and the locally installed Apache-2.0 text recognition engine and English data. It uses no restricted model, training data download, remote scoring or customer content. The software, parameters and exact source commitments were fixed before scoring. [Pinned image checkpoint](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) · [OCR engine license](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) · [English data license](https://github.com/tesseract-ocr/tessdata_fast/blob/main/LICENSE)

## Paired sources

Eight credited Commons photos set the intervention point using 24 original, tone and blur controls. The held photos comprise seven source-preserving SDXL and Kandinsky groups and 12 separately qualified LaMa object-removal groups. Their photos were used by previous project experiments. Each held group has its own exact edited-pixel mask. The added feathered LaMa deliveries blend the original and edited RGB only inside the actual changed pixels. The 12 Chromium groups each contain a repeated record table, a genuine photo preview, and the same preview with the feathered edit. The preview is rendered at native pixel size. The independent audit checks that the two screenshots differ only in the exact displayed photo area. Photos and screenshots stay local. [Commons intake and terms](image-affine-native-scale-2026-11-11.md) · [Seven generated sources and exact masks](image-residual-patch-transfer-2026-11-11.md) · [LaMa source qualification](image-bg-real-backup-intake-v262-method.md)

A useful crop reaches 0.30 native intersection over union and 0.50 changed-pixel recall. The combined check needs four useful feathered crops across both photo cohorts, at most three additional ordinary photo highlights over the existing sharp-boundary check where it applies, at most one ordinary screenshot group highlighted, and at least two useful embedded edit locations. The ordinary screenshot challenge also requires the original crop rule to flag at least three groups. These criteria were frozen before candidate results.

## Held results

| Check | Sharp or native photo edits /19 | Feathered photo edits /19 | Ordinary photo highlights /90 | Ordinary screenshot highlights /24 | Embedded edits /12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing sharp boundary, supported sizes | 5 | 1 | 7 | 21 | 0 |
| Experimental crop with its original remote control | 4 | 2 | 15 | 0 | 0 |
| Crop with text-layout rule | 4 | 2 | 15 | 0 | 0 |
| Crop with local blur comparison | 0 | 0 | 2 | 0 | 0 |
| Crop with both checks | 0 | 0 | 2 | 0 | 0 |

All useful photo locations came from the seven SDXL and Kandinsky groups. The sharp boundary rule assessed 86 of the 90 ordinary photo views. The two useful feathered crop locations had intervention effects of −3.144 and −0.376. Both fail even at the natural zero-effect floor. The calibrated point was 2.968. The learned crop rule had no ordinary screenshot flags for the layout rule to remove. It highlighted two edited screenshots, and neither crop reached the changed-pixel criteria. Applying the same frozen text rule to the sharp-boundary locations as a separate diagnostic identified 10 of 12 repeated-table highlights and zero of nine highlights on the ordinary photo previews. It also identified zero of nine boundary highlights on the edited previews. This distinguishes a narrow text-layout case and leaves the photo-rich interface confound unresolved.

The geometry audit tested all 25 crop positions against the exact masks with scores removed. Even a perfect choice could localize only two of 12 native LaMa edits, one of 12 feathered LaMa edits, and two of 12 embedded edits. All seven sharp and all seven feathered source-preserving patch cases had at least one geometrically useful proposal. The coarse crop cannot cover the tested smaller edits without also covering too much surrounding material.

The offline native review reconstructed 188 source inputs, exact edited-pixel masks and 102 perturbed crop inputs. The 56 reused crop-score sets matched their input and prepared-crop commitments. The separate fit used 648 actual model inputs and the held comparison used 2,082, with exact model-input reuse logged. The freeze is `43412b67b04bf84c234986668feae00217318be17ccdc89e9edf2db5dacbe022`. The native audit digest is `202c6de53ddf19973046b928702183bcbf3bdd3833d4d343d350c33d51561650`.

## Test smaller source regions next

Freeze a new source-separated panel and search windows spanning one-sixth, one-quarter and one-third of the submitted width and height. Use a half-window stride plus the final edge-aligned window. Generate every candidate from dimensions alone, without edit masks or text layout. Rank each scale by its within-image crop contrast, then use ordinary calibration groups to choose one image-level point on the maximum across all three scales. Keep the original remote-crop rule as a direct comparator and measure both raw ranking and useful native mask overlap. An additional fixed arm may require two independently positioned overlapping windows to support the same native region before reporting their intersection.

Keep the original, local blur, tone, ordinary processing and both screenshot controls for every selected source group. Use fresh, independently qualified generated edits and new UI layouts with and without the matched edited photo. Freeze by credited source and actual input hash before any new inference. Score the exact masks only after regions are proposed. Report the per-scale geometry ceiling, useful locations, and any additional ordinary or screenshot highlights at a matched control burden. Advance only with additional useful feathered locations on distinct photos and preservation of useful baseline locations and the frozen false-highlight budget.

## Reproduce

The standard-library reader verifies the source-bound scalar ledger without photographs, masks, weights or OCR output. For native replay, provide the retained source archives, pinned weights and local OCR installation. Chromium blocks network requests when rendering the authored layouts.

```sh
python3 -I -S backend/scripts/review_image_crop_layout_transfer.py \
  docs/evaluation-panels/image-crop-layout-transfer-2026-11-12.json
```

```sh
export PYTHONPATH=backend PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
backend/.venv/bin/python -B -m scripts.image_crop_layout_transfer prepare \
  "$commons" "$lama" "$generated" "$baseline" "$weight" "$tesseract" "$tessdata" "$run"
node backend/scripts/capture_image_crop_layout_transfer.cjs "$run"
for stage in freeze fit held; do
  backend/.venv/bin/python -B -m scripts.image_crop_layout_transfer "$stage" \
    "$commons" "$lama" "$generated" "$baseline" "$weight" "$tesseract" "$tessdata" "$run" || exit 1
done
backend/.venv/bin/python -B -m scripts.review_image_crop_layout_transfer "$run/result.json" --native \
  "$commons" "$lama" "$generated" "$baseline" "$weight" "$tesseract" "$tessdata" "$run"
```

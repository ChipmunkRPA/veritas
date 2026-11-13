# A learned residual encoder for small JPEG edits

A 5,577-parameter convolutional encoder found useful regions in one of 72 edited JPEG75 deliveries and two of 72 edited JPEG90 deliveries. The current JPEG luma check found zero at either quality. The encoder flagged 54 and 67 of 240 ordinary deliveries respectively, compared with five at each quality for the current check. The encoder stays offline. [Audited counts](evaluation-panels/image-residual-encoder-transfer-2026-11-12.json)

## Source roles and controls

The role manifest covers 68 BG-REAL photo groups. Training used eight Stable Diffusion 1.5 local insertions and twelve small native patches cut from publisher SDXL and Kandinsky replacements. Calibration used only twelve separate originals and their authored ordinary controls. The held set includes 24 publisher LaMa local object removals and twelve separate SDXL and Kandinsky patches made by the same native composition rule as the fit set. The LaMa family was absent from fitting. Every photo has individual CC BY 2.0 attribution, including the author and original page. The publisher applies CC BY 4.0 to its own contributions and records the generator terms. No publisher models, implementation, or weights were used. [Publisher dataset and license terms](https://huggingface.co/datasets/anush-123/bg-real#licence-and-attribution) · [Stable Diffusion 1.5 insertion source](image-bg-real-insertions-2026-11-11.md) · [SDXL and Kandinsky composition and rights](image-background-patch-transfer-2026-11-11.md) · [Separate SDXL and Kandinsky intake](image-wavelet-region-transfer-2026-11-11.md) · [LaMa pair qualification](image-bg-real-transport-v225-result.md) · [Calibration photo qualification](image-bg-real-backup-intake-v262-method.md)

Each role is disjoint by publisher recipient, Flickr photo, photographer, encoded original and decoded original RGB. The manifest also checks perceptually informative near-duplicate thumbnails. It preserves per-photo rights records, source digests and exact changed-mask digests. These archived sources had prior project exposure under other methods. The held groups are independent of this encoder and its operating-point calibration. This is a single-publisher study.

Generate sharp and inward cosine-feathered variants at the exact known edit site. For matched ordinary nuisances, use blur, color changes and same-photo copying in that site. Also include the unchanged original, a whole-image resize round trip and a previous JPEG encode. Where the publisher supplies a matched autoencoder pass, include its local and full-image controls. Calibration locations use seeded ellipses chosen independently of any generated edit. Encode each group and variant identically at final JPEG qualities 75 and 90 with 4:2:0 color. The exact edit reference is measured before final encoding, so delivery-induced changes outside the edit do not become positive labels. [Matched-processing methodology](https://arxiv.org/html/2607.26232v3) · [Source-preserving evaluation study](https://arxiv.org/html/2602.00192v2)

## Encoder and comparison

Subtract the 3 × 3 RGB mean and learn three convolutional layers from 3,840 matched native patches. At each positive location, pair a sharp and feathered edit with the ordinary renderings at the same site and the untouched area of edited renderings. Train with a class-balanced logistic loss and an additional paired ranking loss. The 320 fit sites include eight sites per source and final quality. This is an independently implemented small residual encoder. Recent patch-residual research motivates the matched comparison. [DiffusionPrint](https://arxiv.org/html/2604.12443v1)

At inference, slide the encoder every eight pixels and rank rectangles by their mean response against the nearby ring and image-wide median. The largest ordinary score per calibration photo sets its group result. The review point per quality is the second-highest group result, and a region must score strictly above it. Freeze the trial before fitting and save both points before opening held scores. The comparison runs the current aligned JPEG luma localization code on each identical JPEG file. Count a region as useful when it reaches 0.30 intersection over union and 0.50 recall of edited pixels.

| Held delivery | Candidate sharp / feathered useful, out of 36 each | JPEG luma sharp / feathered useful, out of 36 each | Candidate / JPEG luma ordinary flags, out of 240 | Candidate flags at insufficient edit overlap |
| --- | ---: | ---: | ---: | ---: |
| JPEG75 | 0 / 1 | 0 / 0 | 54 / 5 | 21 |
| JPEG90 | 1 / 1 | 0 / 0 | 67 / 5 | 25 |

At either quality, the highest-ranked candidate rectangle was useful before the point in only two of 72 edited deliveries. Both were variants of the same Kandinsky photo across both qualities. No LaMa region was useful even before the point. Ordinary flags affected 11 of 36 groups at JPEG75 and 12 at JPEG90, versus five groups for the current luma check at each quality. Nine and eleven unchanged originals flagged respectively. The luma flags were all on deliberately twice-encoded controls. The frozen advancement criteria called for five added feathered groups across at least two families, with at most three additional ordinary groups and four flags at insufficient edit overlap. Both qualities failed. A lower point would expose more incorrectly placed rectangles with little localization upside in this panel.

## Local replay

The 68.11-second CPU run completed all 144 calibration and 624 held deliveries. Native dense and cropped inference agreed to within 0.000000034 on a deterministic geometry check, and a known response block mapped to the expected source rectangle. The independent readback rebuilt 768 input and truth digests and recalculated 1,392 native rectangle overlaps, then reproduced the ordinary-only points. The published receipt has aggregate results and commitments. The per-photo manifest, raw assets, checkpoint and per-image outputs remain local.

```sh
export PYTHONPATH=backend OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
backend/.venv/bin/python -B -m scripts.image_residual_encoder_transfer freeze \
  --archive "$archive" --output "$run"
backend/.venv/bin/python -B -m scripts.image_residual_encoder_transfer run \
  --archive "$archive" --output "$run" --expected "$freeze_sha256"
backend/.venv/bin/python -B -m scripts.review_image_residual_encoder_transfer \
  --archive "$archive" --output "$run" --publication "$receipt"
```

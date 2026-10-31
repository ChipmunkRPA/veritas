# Ablation: img.univfd

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct ABLATE=img.univfd ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -4.1 pts, CI [-9.4, +0.8] |
| A3b | fail | family midjourney, family sd3 |
| A4 | fail | failing strata: R1, R3 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 23.7 % | 31.6 % | +7.9 |
| F2 | 19 | 36.8 % | 36.8 % | +0.0 |
| F3 | 16 | 56.2 % | 43.8 % | -12.5 |
| F4 | 22 | 63.6 % | 63.6 % | +0.0 |
| F5 | 25 | 32.0 % | 32.0 % | +0.0 |
| F6 | 25 | 56.0 % | 48.0 % | -8.0 |
| F7 | 25 | 88.0 % | 72.0 % | -16.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 50.9 %, with 46.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 17.1 % | 15.9 % | -1.2 | [-10.3, +6.0] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 4.0 % | 5.3 % | +1.3 | [+0.0, +4.0] |
| R4 | 84 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 85 | 8.2 % | 8.2 % | +0.0 | [+0.0, +0.0] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 21.1 % | 23.7 % | +2.6 | [+0.0, +7.9] |
| threat faceswap | 61 | 54.1 % | 55.7 % | +1.6 | [-7.2, +12.0] |
| threat gan_face | 35 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 27 | 48.1 % | 55.6 % | +7.4 | [+0.0, +18.5] |
| threat t2i | 390 | 62.3 % | 63.1 % | +0.8 | [-3.1, +4.6] |
| family faceswap-stills | 61 | 54.1 % | 55.7 % | +1.6 | [-7.2, +12.0] |
| family flux1 | 17 | 94.1 % | 82.4 % | -11.8 | [-29.4, +0.0] |
| family flux2 | 19 | 63.2 % | 63.2 % | +0.0 | [-21.1, +21.1] |
| family gan-faces | 35 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 13 | 76.9 % | 76.9 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 16 | 56.2 % | 50.0 % | -6.2 | [-18.8, +0.0] |
| family hidream | 14 | 92.9 % | 85.7 % | -7.1 | [-21.4, +0.0] |
| family hunyuanimage | 9 | 100.0 % | 88.9 % | -11.1 | [-33.3, +0.0] |
| family ideogram | 9 | 77.8 % | 66.7 % | -11.1 | [-33.3, +0.0] |
| family imagen3 | 16 | 81.2 % | 56.2 % | -25.0 | [-50.0, -6.2] |
| family imagen4 | 25 | 84.0 % | 84.0 % | +0.0 | [-12.0, +12.0] |
| family legacy | 30 | 63.3 % | 70.0 % | +6.7 | [-10.0, +22.6] |
| family long-tail | 41 | 46.3 % | 43.9 % | -2.4 | [-10.3, +5.1] |
| family midjourney | 38 | 26.3 % | 50.0 % | +23.7 | [+10.5, +38.5] |
| family modern-t2i-faces | 29 | 69.0 % | 65.5 % | -3.4 | [-15.8, +8.0] |
| family nano-banana | 22 | 77.3 % | 77.3 % | +0.0 | [-13.6, +13.6] |
| family partial-edits | 38 | 21.1 % | 23.7 % | +2.6 | [+0.0, +7.9] |
| family personalised | 27 | 48.1 % | 55.6 % | +7.4 | [+0.0, +18.5] |
| family qwen-image | 25 | 64.0 % | 68.0 % | +4.0 | [-12.0, +20.0] |
| family recraft | 13 | 84.6 % | 69.2 % | -15.4 | [-38.5, +0.0] |
| family sd3 | 19 | 47.4 % | 73.7 % | +26.3 | [+5.6, +45.0] |
| family seedream3 | 10 | 0.0 % | 20.0 % | +20.0 | [+0.0, +50.0] |
| family seedream4 | 25 | 48.0 % | 40.0 % | -8.0 | [-24.0, +8.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

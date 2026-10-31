# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.univfd, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.3 pts, CI [-3.2, +5.6] |
| A3b | pass | family faceswap-stills, family imagen4, family midjourney, family qwen-image, family sd3, threat faceswap |
| A4 | fail | failing strata: R1, R3 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 28.9 % | 31.6 % | +2.6 |
| F2 | 19 | 36.8 % | 36.8 % | +0.0 |
| F3 | 16 | 37.5 % | 43.8 % | +6.2 |
| F4 | 22 | 63.6 % | 63.6 % | +0.0 |
| F5 | 25 | 28.0 % | 32.0 % | +4.0 |
| F6 | 25 | 52.0 % | 48.0 % | -4.0 |
| F7 | 25 | 72.0 % | 72.0 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.6 %, with 46.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 12.2 % | 15.9 % | +3.7 | [+0.0, +9.7] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 4.0 % | 5.3 % | +1.3 | [+0.0, +4.0] |
| R4 | 84 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 85 | 12.9 % | 8.2 % | -4.7 | [-9.4, -1.2] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 23.7 % | 23.7 % | +0.0 | [-7.9, +7.9] |
| threat faceswap | 61 | 16.4 % | 55.7 % | +39.3 | [+25.8, +53.2] |
| threat gan_face | 35 | 34.3 % | 28.6 % | -5.7 | [-17.1, +5.7] |
| threat personalised | 27 | 77.8 % | 55.6 % | -22.2 | [-40.7, -7.4] |
| threat t2i | 390 | 54.6 % | 63.1 % | +8.5 | [+4.9, +12.0] |
| family faceswap-stills | 61 | 16.4 % | 55.7 % | +39.3 | [+25.8, +53.2] |
| family flux1 | 17 | 70.6 % | 82.4 % | +11.8 | [+0.0, +29.4] |
| family flux2 | 19 | 52.6 % | 63.2 % | +10.5 | [+0.0, +26.3] |
| family gan-faces | 35 | 34.3 % | 28.6 % | -5.7 | [-17.1, +5.7] |
| family gpt-4o-image-1 | 13 | 76.9 % | 76.9 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 16 | 43.8 % | 50.0 % | +6.2 | [+0.0, +18.8] |
| family hidream | 14 | 64.3 % | 85.7 % | +21.4 | [+0.0, +42.9] |
| family hunyuanimage | 9 | 77.8 % | 88.9 % | +11.1 | [-22.2, +44.4] |
| family ideogram | 9 | 66.7 % | 66.7 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 16 | 50.0 % | 56.2 % | +6.2 | [+0.0, +18.8] |
| family imagen4 | 25 | 68.0 % | 84.0 % | +16.0 | [+4.0, +32.0] |
| family legacy | 30 | 60.0 % | 70.0 % | +10.0 | [+0.0, +21.4] |
| family long-tail | 41 | 46.3 % | 43.9 % | -2.4 | [-11.6, +5.0] |
| family midjourney | 38 | 34.2 % | 50.0 % | +15.8 | [+5.4, +28.2] |
| family modern-t2i-faces | 29 | 89.7 % | 65.5 % | -24.1 | [-40.7, -9.7] |
| family nano-banana | 22 | 68.2 % | 77.3 % | +9.1 | [-4.5, +27.3] |
| family partial-edits | 38 | 23.7 % | 23.7 % | +0.0 | [-7.9, +7.9] |
| family personalised | 27 | 77.8 % | 55.6 % | -22.2 | [-40.7, -7.4] |
| family qwen-image | 25 | 52.0 % | 68.0 % | +16.0 | [+4.0, +32.0] |
| family recraft | 13 | 53.8 % | 69.2 % | +15.4 | [+0.0, +38.5] |
| family sd3 | 19 | 36.8 % | 73.7 % | +36.8 | [+11.1, +60.0] |
| family seedream3 | 10 | 0.0 % | 20.0 % | +20.0 | [+0.0, +50.0] |
| family seedream4 | 25 | 36.0 % | 40.0 % | +4.0 | [+0.0, +12.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

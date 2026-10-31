# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,face.gend_clip_l14 ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.univfd, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.6 pts, CI [-1.9, +0.0] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | fail | failing strata: R1 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 31.6 % | 31.6 % | +0.0 |
| F2 | 19 | 36.8 % | 36.8 % | +0.0 |
| F3 | 16 | 43.8 % | 43.8 % | +0.0 |
| F4 | 22 | 63.6 % | 63.6 % | +0.0 |
| F5 | 25 | 32.0 % | 32.0 % | +0.0 |
| F6 | 25 | 48.0 % | 48.0 % | +0.0 |
| F7 | 25 | 72.0 % | 68.0 % | -4.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 46.8 %, with 46.3 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 15.9 % | 17.1 % | +1.2 | [+0.0, +3.1] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 5.3 % | 5.3 % | +0.0 | [+0.0, +0.0] |
| R4 | 84 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 85 | 8.2 % | 7.1 % | -1.2 | [-3.5, +0.0] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 23.7 % | 23.7 % | +0.0 | [-7.9, +7.9] |
| threat faceswap | 61 | 55.7 % | 68.9 % | +13.1 | [+5.4, +23.1] |
| threat gan_face | 35 | 28.6 % | 37.1 % | +8.6 | [+0.0, +17.1] |
| threat personalised | 27 | 55.6 % | 55.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 390 | 63.1 % | 61.5 % | -1.5 | [-3.4, +0.0] |
| family faceswap-stills | 61 | 55.7 % | 68.9 % | +13.1 | [+5.4, +23.1] |
| family flux1 | 17 | 82.4 % | 82.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 19 | 63.2 % | 63.2 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 35 | 28.6 % | 37.1 % | +8.6 | [+0.0, +17.1] |
| family gpt-4o-image-1 | 13 | 76.9 % | 76.9 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 16 | 50.0 % | 50.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 14 | 85.7 % | 85.7 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 9 | 88.9 % | 88.9 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 9 | 66.7 % | 66.7 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 16 | 56.2 % | 62.5 % | +6.2 | [+0.0, +18.8] |
| family imagen4 | 25 | 84.0 % | 84.0 % | +0.0 | [+0.0, +0.0] |
| family legacy | 30 | 70.0 % | 70.0 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 41 | 43.9 % | 43.9 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 38 | 50.0 % | 47.4 % | -2.6 | [-15.0, +8.1] |
| family modern-t2i-faces | 29 | 65.5 % | 62.1 % | -3.4 | [-11.5, +0.0] |
| family nano-banana | 22 | 77.3 % | 77.3 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 38 | 23.7 % | 23.7 % | +0.0 | [-7.9, +7.9] |
| family personalised | 27 | 55.6 % | 55.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 25 | 68.0 % | 68.0 % | +0.0 | [+0.0, +0.0] |
| family recraft | 13 | 69.2 % | 69.2 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 19 | 73.7 % | 57.9 % | -15.8 | [-33.3, +0.0] |
| family seedream3 | 10 | 20.0 % | 10.0 % | -10.0 | [-30.0, +0.0] |
| family seedream4 | 25 | 40.0 % | 36.0 % | -4.0 | [-12.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

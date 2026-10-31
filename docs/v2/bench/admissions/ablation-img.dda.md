# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.dda ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.univfd, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +27.7 pts, CI [+20.8, +34.6] |
| A3b | pass | family faceswap-stills, family gan-faces, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | fail | failing strata: R1, R3, R4, R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 31.6 % | 94.7 % | +63.2 |
| F2 | 19 | 36.8 % | 57.9 % | +21.1 |
| F3 | 16 | 43.8 % | 50.0 % | +6.2 |
| F4 | 22 | 63.6 % | 90.9 % | +27.3 |
| F5 | 25 | 32.0 % | 56.0 % | +24.0 |
| F6 | 25 | 48.0 % | 72.0 % | +24.0 |
| F7 | 25 | 72.0 % | 100.0 % | +28.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 46.8 %, with 74.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 15.9 % | 13.4 % | -2.4 | [-14.0, +5.5] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 5.3 % | 6.7 % | +1.3 | [+0.0, +4.0] |
| R4 | 84 | 0.0 % | 1.2 % | +1.2 | [+0.0, +3.6] |
| R5 | 85 | 8.2 % | 8.2 % | +0.0 | [-4.7, +4.7] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 23.7 % | 92.1 % | +68.4 | [+52.6, +81.6] |
| threat faceswap | 61 | 55.7 % | 85.2 % | +29.5 | [+16.7, +42.2] |
| threat gan_face | 35 | 28.6 % | 48.6 % | +20.0 | [+8.6, +34.3] |
| threat personalised | 27 | 55.6 % | 100.0 % | +44.4 | [+25.9, +63.0] |
| threat t2i | 390 | 63.1 % | 87.9 % | +24.9 | [+20.5, +29.5] |
| family faceswap-stills | 61 | 55.7 % | 85.2 % | +29.5 | [+16.7, +42.2] |
| family flux1 | 17 | 82.4 % | 100.0 % | +17.6 | [+0.0, +35.3] |
| family flux2 | 19 | 63.2 % | 63.2 % | +0.0 | [-21.1, +21.1] |
| family gan-faces | 35 | 28.6 % | 48.6 % | +20.0 | [+8.6, +34.3] |
| family gpt-4o-image-1 | 13 | 76.9 % | 100.0 % | +23.1 | [+0.0, +46.2] |
| family gpt-image-1.5-2 | 16 | 50.0 % | 56.2 % | +6.2 | [-18.8, +25.0] |
| family hidream | 14 | 85.7 % | 100.0 % | +14.3 | [+0.0, +35.7] |
| family hunyuanimage | 9 | 88.9 % | 100.0 % | +11.1 | [+0.0, +33.3] |
| family ideogram | 9 | 66.7 % | 100.0 % | +33.3 | [+0.0, +66.7] |
| family imagen3 | 16 | 56.2 % | 100.0 % | +43.8 | [+18.8, +68.8] |
| family imagen4 | 25 | 84.0 % | 100.0 % | +16.0 | [+4.0, +32.0] |
| family legacy | 30 | 70.0 % | 90.0 % | +20.0 | [+6.9, +33.3] |
| family long-tail | 41 | 43.9 % | 82.9 % | +39.0 | [+24.4, +53.5] |
| family midjourney | 38 | 50.0 % | 94.7 % | +44.7 | [+28.9, +62.2] |
| family modern-t2i-faces | 29 | 65.5 % | 93.1 % | +27.6 | [+12.5, +42.4] |
| family nano-banana | 22 | 77.3 % | 95.5 % | +18.2 | [+4.5, +36.4] |
| family partial-edits | 38 | 23.7 % | 92.1 % | +68.4 | [+52.6, +81.6] |
| family personalised | 27 | 55.6 % | 100.0 % | +44.4 | [+25.9, +63.0] |
| family qwen-image | 25 | 68.0 % | 76.0 % | +8.0 | [-8.0, +24.0] |
| family recraft | 13 | 69.2 % | 92.3 % | +23.1 | [+0.0, +46.2] |
| family sd3 | 19 | 73.7 % | 100.0 % | +26.3 | [+9.1, +50.0] |
| family seedream3 | 10 | 20.0 % | 80.0 % | +60.0 | [+30.0, +90.0] |
| family seedream4 | 25 | 40.0 % | 64.0 % | +24.0 | [+8.0, +40.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

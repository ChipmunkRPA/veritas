# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.univfd, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +16.0 pts, CI [+9.0, +22.7] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family hidream, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family qwen-image, family sd3, threat edit, threat t2i |
| A4 | fail | failing strata: R1, R3, R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 21.1 % | 31.6 % | +10.5 |
| F2 | 19 | 21.1 % | 36.8 % | +15.8 |
| F3 | 16 | 43.8 % | 43.8 % | +0.0 |
| F4 | 22 | 18.2 % | 63.6 % | +45.5 |
| F5 | 25 | 24.0 % | 32.0 % | +8.0 |
| F6 | 25 | 36.0 % | 48.0 % | +12.0 |
| F7 | 25 | 52.0 % | 72.0 % | +20.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 30.9 %, with 46.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 15.9 % | 15.9 % | +0.0 | [-12.3, +6.5] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 1.3 % | 5.3 % | +4.0 | [+0.0, +9.3] |
| R4 | 84 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 85 | 11.8 % | 8.2 % | -3.5 | [-9.4, +2.4] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 7.9 % | 23.7 % | +15.8 | [+5.3, +26.3] |
| threat faceswap | 61 | 55.7 % | 55.7 % | +0.0 | [-6.8, +6.7] |
| threat gan_face | 35 | 28.6 % | 28.6 % | +0.0 | [-8.6, +8.6] |
| threat personalised | 27 | 55.6 % | 55.6 % | +0.0 | [-22.2, +22.2] |
| threat t2i | 390 | 35.4 % | 63.1 % | +27.7 | [+22.6, +32.8] |
| family faceswap-stills | 61 | 55.7 % | 55.7 % | +0.0 | [-6.8, +6.7] |
| family flux1 | 17 | 47.1 % | 82.4 % | +35.3 | [+11.8, +58.8] |
| family flux2 | 19 | 21.1 % | 63.2 % | +42.1 | [+21.1, +63.2] |
| family gan-faces | 35 | 28.6 % | 28.6 % | +0.0 | [-8.6, +8.6] |
| family gpt-4o-image-1 | 13 | 38.5 % | 76.9 % | +38.5 | [+15.4, +61.5] |
| family gpt-image-1.5-2 | 16 | 50.0 % | 50.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 14 | 14.3 % | 85.7 % | +71.4 | [+35.7, +100.0] |
| family hunyuanimage | 9 | 66.7 % | 88.9 % | +22.2 | [+0.0, +55.6] |
| family ideogram | 9 | 33.3 % | 66.7 % | +33.3 | [-11.1, +66.7] |
| family imagen3 | 16 | 62.5 % | 56.2 % | -6.2 | [-31.2, +18.8] |
| family imagen4 | 25 | 56.0 % | 84.0 % | +28.0 | [+4.0, +52.0] |
| family legacy | 30 | 43.3 % | 70.0 % | +26.7 | [+10.0, +44.8] |
| family long-tail | 41 | 19.5 % | 43.9 % | +24.4 | [+11.9, +38.5] |
| family midjourney | 38 | 18.4 % | 50.0 % | +31.6 | [+16.2, +46.3] |
| family modern-t2i-faces | 29 | 34.5 % | 65.5 % | +31.0 | [+13.0, +51.7] |
| family nano-banana | 22 | 36.4 % | 77.3 % | +40.9 | [+22.7, +59.1] |
| family partial-edits | 38 | 7.9 % | 23.7 % | +15.8 | [+5.3, +26.3] |
| family personalised | 27 | 55.6 % | 55.6 % | +0.0 | [-22.2, +22.2] |
| family qwen-image | 25 | 36.0 % | 68.0 % | +32.0 | [+16.0, +52.0] |
| family recraft | 13 | 61.5 % | 69.2 % | +7.7 | [+0.0, +23.1] |
| family sd3 | 19 | 47.4 % | 73.7 % | +26.3 | [+5.6, +50.0] |
| family seedream3 | 10 | 0.0 % | 20.0 % | +20.0 | [+0.0, +50.0] |
| family seedream4 | 25 | 24.0 % | 40.0 % | +16.0 | [+0.0, +32.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

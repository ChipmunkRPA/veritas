# Ablation: img.hf.wvolf

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.0-core DETECTORS=img.univfd,img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.hf.wvolf ABLATE=img.hf.wvolf ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `d0224343a5c0ad38605ade2b5225deba3cb607ace950e2ae3a127ba9dab26d25`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.univfd, img.watermark.dwtdct
- Items (fit ∪ cal): 930 (379 real, 551 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | failing strata: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 38 | 31.6 % | 31.6 % | +0.0 |
| F2 | 19 | 36.8 % | 36.8 % | +0.0 |
| F3 | 16 | 43.8 % | 43.8 % | +0.0 |
| F4 | 22 | 63.6 % | 63.6 % | +0.0 |
| F5 | 25 | 32.0 % | 32.0 % | +0.0 |
| F6 | 25 | 48.0 % | 48.0 % | +0.0 |
| F7 | 25 | 72.0 % | 72.0 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 46.8 %, with 46.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 82 | 15.9 % | 15.9 % | +0.0 | [+0.0, +0.0] |
| R2 | 49 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R3 | 75 | 5.3 % | 5.3 % | +0.0 | [+0.0, +0.0] |
| R4 | 84 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 85 | 8.2 % | 8.2 % | +0.0 | [+0.0, +0.0] |
| R6 | 4 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 38 | 23.7 % | 23.7 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 61 | 55.7 % | 55.7 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 35 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 27 | 55.6 % | 55.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 390 | 63.1 % | 63.1 % | +0.0 | [+0.0, +0.0] |
| family faceswap-stills | 61 | 55.7 % | 55.7 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 17 | 82.4 % | 82.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 19 | 63.2 % | 63.2 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 35 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 13 | 76.9 % | 76.9 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 16 | 50.0 % | 50.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 14 | 85.7 % | 85.7 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 9 | 88.9 % | 88.9 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 9 | 66.7 % | 66.7 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 16 | 56.2 % | 56.2 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 25 | 84.0 % | 84.0 % | +0.0 | [+0.0, +0.0] |
| family legacy | 30 | 70.0 % | 70.0 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 41 | 43.9 % | 43.9 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 38 | 50.0 % | 50.0 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 29 | 65.5 % | 65.5 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 22 | 77.3 % | 77.3 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 38 | 23.7 % | 23.7 % | +0.0 | [+0.0, +0.0] |
| family personalised | 27 | 55.6 % | 55.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 25 | 68.0 % | 68.0 % | +0.0 | [+0.0, +0.0] |
| family recraft | 13 | 69.2 % | 69.2 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 19 | 73.7 % | 73.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 10 | 20.0 % | 20.0 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 25 | 40.0 % | 40.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

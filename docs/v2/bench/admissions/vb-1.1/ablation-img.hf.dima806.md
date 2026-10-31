# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.6 pts, CI [-3.0, -0.3] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.5, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 62.6 % | +0.0 |
| F2 | 80 | 68.8 % | 63.7 % | -5.0 |
| F3 | 62 | 45.2 % | 41.9 % | -3.2 |
| F4 | 130 | 63.8 % | 61.5 % | -2.3 |
| F5 | 94 | 38.3 % | 36.2 % | -2.1 |
| F6 | 113 | 75.2 % | 73.5 % | -1.8 |
| F7 | 118 | 78.8 % | 82.2 % | +3.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 61.8 %, with 60.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 7.1 % | 6.4 % | -0.7 | [-2.7, +0.0] |
| R2 | 611 | 0.2 % | 0.2 % | +0.0 | [+0.0, +0.0] |
| R3 | 587 | 14.5 % | 13.5 % | -1.0 | [-2.1, +0.0] |
| R4 | 495 | 0.8 % | 0.6 % | -0.2 | [-0.6, +0.0] |
| R5 | 549 | 7.1 % | 9.1 % | +2.0 | [+0.2, +3.8] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 53.1 % | 49.1 % | -4.0 | [-7.4, -0.6] |
| threat faceswap | 301 | 71.1 % | 67.8 % | -3.3 | [-7.4, +0.6] |
| threat gan_face | 163 | 12.9 % | 19.0 % | +6.1 | [+1.2, +11.7] |
| threat personalised | 124 | 33.1 % | 30.6 % | -2.4 | [-8.1, +2.4] |
| threat t2i | 1764 | 66.6 % | 63.9 % | -2.6 | [-3.6, -1.7] |
| family faceswap-stills | 301 | 71.1 % | 67.8 % | -3.3 | [-7.4, +0.6] |
| family flux1 | 93 | 75.3 % | 75.3 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 80 | 75.0 % | 70.0 % | -5.0 | [-10.6, -1.2] |
| family gan-faces | 163 | 12.9 % | 19.0 % | +6.1 | [+1.2, +11.7] |
| family gpt-4o-image-1 | 46 | 69.6 % | 69.6 % | +0.0 | [-6.5, +6.5] |
| family gpt-image-1.5-2 | 62 | 48.4 % | 43.5 % | -4.8 | [-11.3, +0.0] |
| family hidream | 52 | 96.2 % | 96.2 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 29 | 86.2 % | 82.8 % | -3.4 | [-10.3, +0.0] |
| family ideogram | 52 | 76.9 % | 76.9 % | +0.0 | [-5.7, +5.7] |
| family imagen3 | 60 | 46.7 % | 46.7 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 118 | 89.0 % | 84.7 % | -4.2 | [-8.3, -0.9] |
| family legacy | 163 | 38.7 % | 36.2 % | -2.5 | [-4.8, -0.6] |
| family long-tail | 203 | 67.0 % | 64.5 % | -2.5 | [-5.5, +0.0] |
| family midjourney | 163 | 66.9 % | 65.6 % | -1.2 | [-4.2, +1.8] |
| family modern-t2i-faces | 124 | 98.4 % | 86.3 % | -12.1 | [-18.0, -7.1] |
| family nano-banana | 130 | 63.8 % | 63.1 % | -0.8 | [-3.1, +1.6] |
| family partial-edits | 175 | 53.1 % | 49.1 % | -4.0 | [-7.4, -0.6] |
| family personalised | 124 | 33.1 % | 30.6 % | -2.4 | [-8.1, +2.4] |
| family qwen-image | 113 | 76.1 % | 75.2 % | -0.9 | [-2.8, +0.0] |
| family recraft | 57 | 63.2 % | 61.4 % | -1.8 | [-5.5, +0.0] |
| family sd3 | 84 | 45.2 % | 46.4 % | +1.2 | [+0.0, +3.7] |
| family seedream3 | 41 | 53.7 % | 51.2 % | -2.4 | [-7.7, +0.0] |
| family seedream4 | 94 | 41.5 % | 37.2 % | -4.3 | [-8.6, -1.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

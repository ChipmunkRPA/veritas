# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.dda GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +15.0 pts, CI [+11.7, +18.3] |
| A3b | pass | family faceswap-stills, family flux1, family gan-faces, family gpt-4o-image-1, family gpt-image-1.5-2, family hunyuanimage, family ideogram, family imagen3, family legacy, family long-tail, family midjourney, family nano-banana, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.9, +0.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 93.9 % | +31.3 |
| F2 | 80 | 63.7 % | 65.0 % | +1.3 |
| F3 | 62 | 41.9 % | 51.6 % | +9.7 |
| F4 | 130 | 61.5 % | 79.2 % | +17.7 |
| F5 | 94 | 36.2 % | 69.1 % | +33.0 |
| F6 | 113 | 73.5 % | 78.8 % | +5.3 |
| F7 | 118 | 82.2 % | 89.0 % | +6.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 60.2 %, with 75.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 8.3 % | +1.9 | [+0.2, +5.6] |
| R2 | 611 | 0.2 % | 0.3 % | +0.2 | [+0.0, +0.5] |
| R3 | 587 | 13.5 % | 12.6 % | -0.9 | [-3.1, +1.4] |
| R4 | 495 | 0.6 % | 1.8 % | +1.2 | [+0.2, +2.4] |
| R5 | 549 | 9.1 % | 6.7 % | -2.4 | [-4.6, +0.0] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 49.1 % | 86.9 % | +37.7 | [+29.7, +45.7] |
| threat faceswap | 301 | 67.8 % | 80.1 % | +12.3 | [+7.4, +18.2] |
| threat gan_face | 163 | 19.0 % | 50.9 % | +31.9 | [+25.2, +39.3] |
| threat personalised | 124 | 30.6 % | 97.6 % | +66.9 | [+58.1, +75.0] |
| threat t2i | 1764 | 63.9 % | 86.9 % | +23.0 | [+20.5, +25.4] |
| family faceswap-stills | 301 | 67.8 % | 80.1 % | +12.3 | [+7.4, +18.2] |
| family flux1 | 93 | 75.3 % | 92.5 % | +17.2 | [+10.1, +25.3] |
| family flux2 | 80 | 70.0 % | 71.2 % | +1.3 | [-7.9, +11.1] |
| family gan-faces | 163 | 19.0 % | 50.9 % | +31.9 | [+25.2, +39.3] |
| family gpt-4o-image-1 | 46 | 69.6 % | 87.0 % | +17.4 | [+6.5, +28.3] |
| family gpt-image-1.5-2 | 62 | 43.5 % | 54.8 % | +11.3 | [+1.6, +21.0] |
| family hidream | 52 | 96.2 % | 100.0 % | +3.8 | [+0.0, +10.0] |
| family hunyuanimage | 29 | 82.8 % | 100.0 % | +17.2 | [+6.9, +31.0] |
| family ideogram | 52 | 76.9 % | 98.1 % | +21.2 | [+10.2, +32.1] |
| family imagen3 | 60 | 46.7 % | 88.3 % | +41.7 | [+30.0, +55.0] |
| family imagen4 | 118 | 84.7 % | 90.7 % | +5.9 | [+0.8, +11.7] |
| family legacy | 163 | 36.2 % | 84.7 % | +48.5 | [+41.8, +55.5] |
| family long-tail | 203 | 64.5 % | 88.7 % | +24.1 | [+16.8, +31.6] |
| family midjourney | 163 | 65.6 % | 94.5 % | +28.8 | [+20.8, +36.5] |
| family modern-t2i-faces | 124 | 86.3 % | 91.9 % | +5.6 | [+0.8, +11.7] |
| family nano-banana | 130 | 63.1 % | 82.3 % | +19.2 | [+11.6, +27.7] |
| family partial-edits | 175 | 49.1 % | 86.9 % | +37.7 | [+29.7, +45.7] |
| family personalised | 124 | 30.6 % | 97.6 % | +66.9 | [+58.1, +75.0] |
| family qwen-image | 113 | 75.2 % | 77.9 % | +2.7 | [-5.3, +10.7] |
| family recraft | 57 | 61.4 % | 94.7 % | +33.3 | [+21.3, +47.3] |
| family sd3 | 84 | 46.4 % | 100.0 % | +53.6 | [+42.0, +64.6] |
| family seedream3 | 41 | 51.2 % | 97.6 % | +46.3 | [+31.7, +61.5] |
| family seedream4 | 94 | 37.2 % | 69.1 % | +31.9 | [+21.3, +41.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

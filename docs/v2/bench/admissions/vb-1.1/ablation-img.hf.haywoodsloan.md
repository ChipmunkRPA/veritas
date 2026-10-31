# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.hf.haywoodsloan GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +12.8 pts, CI [+9.0, +16.5] |
| A3b | pass | family flux1, family gan-faces, family gpt-4o-image-1, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-1.0, +0.8] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R4 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 90.8 % | +28.2 |
| F2 | 80 | 63.7 % | 52.5 % | -11.2 |
| F3 | 62 | 41.9 % | 56.5 % | +14.5 |
| F4 | 130 | 61.5 % | 78.5 % | +16.9 |
| F5 | 94 | 36.2 % | 52.1 % | +16.0 |
| F6 | 113 | 73.5 % | 89.4 % | +15.9 |
| F7 | 118 | 82.2 % | 91.5 % | +9.3 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 60.2 %, with 73.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 6.4 % | +0.0 | [-1.8, +1.7] |
| R2 | 611 | 0.2 % | 0.0 % | -0.2 | [-0.7, +0.0] |
| R3 | 587 | 13.5 % | 9.9 % | -3.6 | [-6.1, -1.0] |
| R4 | 495 | 0.6 % | 3.0 % | +2.4 | [+1.0, +4.0] |
| R5 | 549 | 9.1 % | 10.4 % | +1.3 | [-1.1, +3.8] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 49.1 % | 52.6 % | +3.4 | [-3.4, +10.3] |
| threat faceswap | 301 | 67.8 % | 58.5 % | -9.3 | [-15.0, -4.6] |
| threat gan_face | 163 | 19.0 % | 30.7 % | +11.7 | [+4.9, +18.4] |
| threat personalised | 124 | 30.6 % | 49.2 % | +18.5 | [+10.5, +26.6] |
| threat t2i | 1764 | 63.9 % | 82.7 % | +18.8 | [+16.6, +21.1] |
| family faceswap-stills | 301 | 67.8 % | 58.5 % | -9.3 | [-15.0, -4.6] |
| family flux1 | 93 | 75.3 % | 88.2 % | +12.9 | [+4.4, +21.9] |
| family flux2 | 80 | 70.0 % | 60.0 % | -10.0 | [-21.3, +1.2] |
| family gan-faces | 163 | 19.0 % | 30.7 % | +11.7 | [+4.9, +18.4] |
| family gpt-4o-image-1 | 46 | 69.6 % | 89.1 % | +19.6 | [+4.3, +34.8] |
| family gpt-image-1.5-2 | 62 | 43.5 % | 56.5 % | +12.9 | [+0.0, +25.8] |
| family hidream | 52 | 96.2 % | 94.2 % | -1.9 | [-8.2, +4.0] |
| family hunyuanimage | 29 | 82.8 % | 96.6 % | +13.8 | [+3.4, +27.6] |
| family ideogram | 52 | 76.9 % | 96.2 % | +19.2 | [+8.0, +30.9] |
| family imagen3 | 60 | 46.7 % | 78.3 % | +31.7 | [+18.3, +45.0] |
| family imagen4 | 118 | 84.7 % | 94.9 % | +10.2 | [+3.5, +17.2] |
| family legacy | 163 | 36.2 % | 67.5 % | +31.3 | [+23.5, +38.6] |
| family long-tail | 203 | 64.5 % | 84.2 % | +19.7 | [+12.9, +26.1] |
| family midjourney | 163 | 65.6 % | 92.6 % | +27.0 | [+19.3, +34.6] |
| family modern-t2i-faces | 124 | 86.3 % | 100.0 % | +13.7 | [+8.5, +19.7] |
| family nano-banana | 130 | 63.1 % | 83.8 % | +20.8 | [+12.3, +29.6] |
| family partial-edits | 175 | 49.1 % | 52.6 % | +3.4 | [-3.4, +10.3] |
| family personalised | 124 | 30.6 % | 49.2 % | +18.5 | [+10.5, +26.6] |
| family qwen-image | 113 | 75.2 % | 93.8 % | +18.6 | [+10.0, +27.5] |
| family recraft | 57 | 61.4 % | 91.2 % | +29.8 | [+17.5, +43.6] |
| family sd3 | 84 | 46.4 % | 63.1 % | +16.7 | [+8.2, +25.3] |
| family seedream3 | 41 | 51.2 % | 78.0 % | +26.8 | [+10.3, +43.6] |
| family seedream4 | 94 | 37.2 % | 62.8 % | +25.5 | [+16.1, +35.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

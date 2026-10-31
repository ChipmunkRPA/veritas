# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +35.5 pts, CI [+31.1, +39.6] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat t2i |
| A4 | fail | pooled reals: Δ FPR +0.2 pts, CI [-0.7, +1.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 38.0 % | 62.6 % | +24.5 |
| F2 | 80 | 26.2 % | 63.7 % | +37.5 |
| F3 | 62 | 19.4 % | 41.9 % | +22.6 |
| F4 | 130 | 13.1 % | 61.5 % | +48.5 |
| F5 | 94 | 10.6 % | 36.2 % | +25.5 |
| F6 | 113 | 28.3 % | 73.5 % | +45.1 |
| F7 | 118 | 37.3 % | 82.2 % | +44.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 24.7 %, with 60.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 9.9 % | 6.4 % | -3.5 | [-9.8, -1.1] |
| R2 | 611 | 0.0 % | 0.2 % | +0.2 | [+0.0, +0.7] |
| R3 | 587 | 13.5 % | 13.5 % | +0.0 | [-3.3, +3.2] |
| R4 | 495 | 1.0 % | 0.6 % | -0.4 | [-1.2, +0.4] |
| R5 | 549 | 5.3 % | 9.1 % | +3.8 | [+1.8, +5.8] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 38.9 % | 49.1 % | +10.3 | [+3.4, +17.1] |
| threat faceswap | 301 | 59.5 % | 67.8 % | +8.3 | [+3.4, +15.0] |
| threat gan_face | 163 | 12.3 % | 19.0 % | +6.7 | [+1.8, +12.3] |
| threat personalised | 124 | 22.6 % | 30.6 % | +8.1 | [+4.0, +12.9] |
| threat t2i | 1764 | 30.7 % | 63.9 % | +33.2 | [+30.3, +36.0] |
| family faceswap-stills | 301 | 59.5 % | 67.8 % | +8.3 | [+3.4, +15.0] |
| family flux1 | 93 | 19.4 % | 75.3 % | +55.9 | [+44.2, +66.7] |
| family flux2 | 80 | 35.0 % | 70.0 % | +35.0 | [+20.8, +47.6] |
| family gan-faces | 163 | 12.3 % | 19.0 % | +6.7 | [+1.8, +12.3] |
| family gpt-4o-image-1 | 46 | 17.4 % | 69.6 % | +52.2 | [+37.0, +67.4] |
| family gpt-image-1.5-2 | 62 | 22.6 % | 43.5 % | +21.0 | [+9.7, +32.3] |
| family hidream | 52 | 59.6 % | 96.2 % | +36.5 | [+22.6, +50.0] |
| family hunyuanimage | 29 | 37.9 % | 82.8 % | +44.8 | [+27.6, +62.1] |
| family ideogram | 52 | 55.8 % | 76.9 % | +21.2 | [+3.8, +37.7] |
| family imagen3 | 60 | 21.7 % | 46.7 % | +25.0 | [+13.3, +36.7] |
| family imagen4 | 118 | 45.8 % | 84.7 % | +39.0 | [+29.0, +48.3] |
| family legacy | 163 | 14.7 % | 36.2 % | +21.5 | [+14.5, +28.3] |
| family long-tail | 203 | 35.5 % | 64.5 % | +29.1 | [+22.0, +36.6] |
| family midjourney | 163 | 42.9 % | 65.6 % | +22.7 | [+13.6, +31.5] |
| family modern-t2i-faces | 124 | 44.4 % | 86.3 % | +41.9 | [+34.8, +48.8] |
| family nano-banana | 130 | 17.7 % | 63.1 % | +45.4 | [+36.4, +54.6] |
| family partial-edits | 175 | 38.9 % | 49.1 % | +10.3 | [+3.4, +17.1] |
| family personalised | 124 | 22.6 % | 30.6 % | +8.1 | [+4.0, +12.9] |
| family qwen-image | 113 | 33.6 % | 75.2 % | +41.6 | [+31.2, +52.3] |
| family recraft | 57 | 21.1 % | 61.4 % | +40.4 | [+26.3, +54.0] |
| family sd3 | 84 | 22.6 % | 46.4 % | +23.8 | [+14.1, +34.1] |
| family seedream3 | 41 | 22.0 % | 51.2 % | +29.3 | [+9.8, +48.8] |
| family seedream4 | 94 | 14.9 % | 37.2 % | +22.3 | [+10.5, +33.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

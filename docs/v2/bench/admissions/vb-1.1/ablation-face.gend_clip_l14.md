# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,face.gend_clip_l14 GATES='img.dfdc_b7=has_face;face.gend_clip_l14=has_face' CONTEXT=scored ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -2.3 pts, CI [-4.0, -0.7] |
| A3b | fail | family faceswap-stills, family gan-faces, family personalised, threat faceswap, threat gan_face, threat personalised |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.5, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 61.3 % | -1.2 |
| F2 | 80 | 63.7 % | 65.0 % | +1.3 |
| F3 | 62 | 41.9 % | 38.7 % | -3.2 |
| F4 | 130 | 61.5 % | 58.5 % | -3.1 |
| F5 | 94 | 36.2 % | 38.3 % | +2.1 |
| F6 | 113 | 73.5 % | 63.7 % | -9.7 |
| F7 | 118 | 82.2 % | 79.7 % | -2.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 60.2 %, with 57.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 7.3 % | +0.9 | [-0.5, +2.0] |
| R2 | 611 | 0.2 % | 0.2 % | +0.0 | [+0.0, +0.0] |
| R3 | 587 | 13.5 % | 13.5 % | +0.0 | [-0.9, +1.0] |
| R4 | 495 | 0.6 % | 0.6 % | +0.0 | [+0.0, +0.0] |
| R5 | 549 | 9.1 % | 8.2 % | -0.9 | [-2.9, +0.9] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 49.1 % | 52.6 % | +3.4 | [+0.0, +6.9] |
| threat faceswap | 301 | 67.8 % | 81.4 % | +13.6 | [+8.6, +19.7] |
| threat gan_face | 163 | 19.0 % | 42.9 % | +23.9 | [+17.2, +31.3] |
| threat personalised | 124 | 30.6 % | 44.4 % | +13.7 | [+6.5, +21.0] |
| threat t2i | 1764 | 63.9 % | 62.9 % | -1.0 | [-1.8, -0.2] |
| family faceswap-stills | 301 | 67.8 % | 81.4 % | +13.6 | [+8.6, +19.7] |
| family flux1 | 93 | 75.3 % | 75.3 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 80 | 70.0 % | 70.0 % | +0.0 | [-5.1, +5.1] |
| family gan-faces | 163 | 19.0 % | 42.9 % | +23.9 | [+17.2, +31.3] |
| family gpt-4o-image-1 | 46 | 69.6 % | 65.2 % | -4.3 | [-10.9, +0.0] |
| family gpt-image-1.5-2 | 62 | 43.5 % | 40.3 % | -3.2 | [-8.1, +0.0] |
| family hidream | 52 | 96.2 % | 96.2 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 29 | 82.8 % | 82.8 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 52 | 76.9 % | 71.2 % | -5.8 | [-13.2, +0.0] |
| family imagen3 | 60 | 46.7 % | 41.7 % | -5.0 | [-11.7, +0.0] |
| family imagen4 | 118 | 84.7 % | 83.9 % | -0.8 | [-4.0, +1.7] |
| family legacy | 163 | 36.2 % | 39.9 % | +3.7 | [+1.2, +6.8] |
| family long-tail | 203 | 64.5 % | 63.5 % | -1.0 | [-4.1, +2.0] |
| family midjourney | 163 | 65.6 % | 65.0 % | -0.6 | [-3.0, +1.3] |
| family modern-t2i-faces | 124 | 86.3 % | 83.1 % | -3.2 | [-7.8, +0.9] |
| family nano-banana | 130 | 63.1 % | 60.8 % | -2.3 | [-5.3, +0.0] |
| family partial-edits | 175 | 49.1 % | 52.6 % | +3.4 | [+0.0, +6.9] |
| family personalised | 124 | 30.6 % | 44.4 % | +13.7 | [+6.5, +21.0] |
| family qwen-image | 113 | 75.2 % | 71.7 % | -3.5 | [-7.4, -0.9] |
| family recraft | 57 | 61.4 % | 63.2 % | +1.8 | [+0.0, +5.5] |
| family sd3 | 84 | 46.4 % | 45.2 % | -1.2 | [-3.7, +0.0] |
| family seedream3 | 41 | 51.2 % | 51.2 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 94 | 37.2 % | 38.3 % | +1.1 | [-2.2, +5.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

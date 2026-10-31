# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806,img.dfdc_b7,img.forensic.spectrum,img.forensic.ela,img.forensic.noise,img.forensic.jpeg_dq,img.forensic.copymove,img.meta.generator,img.meta.exif,img.meta.consistency,img.meta.qtables,img.watermark.dwtdct,img.commfor384 GATES='img.dfdc_b7=has_face' CONTEXT=scored ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.forensic.copymove, img.forensic.ela, img.forensic.jpeg_dq, img.forensic.noise, img.forensic.spectrum, img.hf.ateeqq, img.hf.dima806, img.meta.consistency, img.meta.exif, img.meta.generator, img.meta.qtables, img.watermark.dwtdct
- Items (fit ∪ cal): 5221 (2694 real, 2527 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05
- Criteria recomputed with D-44 (`app.bench.ablation.criteria`, H-4) from the stored LGO and cross-fit tables; the JSON keeps the D-41 criteria of the run.

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +19.3 pts, CI [+15.8, +22.8] |
| A3b | pass | family faceswap-stills, family flux1, family gan-faces, family gpt-image-1.5-2, family hunyuanimage, family ideogram, family imagen3, family legacy, family long-tail, family midjourney, family nano-banana, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.9, +0.8] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 163 | 62.6 % | 98.2 % | +35.6 |
| F2 | 80 | 63.7 % | 63.7 % | +0.0 |
| F3 | 62 | 41.9 % | 53.2 % | +11.3 |
| F4 | 130 | 61.5 % | 76.9 % | +15.4 |
| F5 | 94 | 36.2 % | 86.2 % | +50.0 |
| F6 | 113 | 73.5 % | 85.0 % | +11.5 |
| F7 | 118 | 82.2 % | 93.2 % | +11.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 60.2 %, with 79.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 7.8 % | +1.4 | [-0.6, +5.2] |
| R2 | 611 | 0.2 % | 0.2 % | +0.0 | [-0.5, +0.4] |
| R3 | 587 | 13.5 % | 15.3 % | +1.9 | [-0.7, +4.6] |
| R4 | 495 | 0.6 % | 1.6 % | +1.0 | [+0.0, +2.0] |
| R5 | 549 | 9.1 % | 4.7 % | -4.4 | [-6.7, -2.0] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 175 | 49.1 % | 83.4 % | +34.3 | [+25.7, +42.9] |
| threat faceswap | 301 | 67.8 % | 83.7 % | +15.9 | [+10.2, +22.6] |
| threat gan_face | 163 | 19.0 % | 63.2 % | +44.2 | [+36.2, +51.5] |
| threat personalised | 124 | 30.6 % | 96.0 % | +65.3 | [+56.5, +73.4] |
| threat t2i | 1764 | 63.9 % | 88.4 % | +24.4 | [+21.8, +26.9] |
| family faceswap-stills | 301 | 67.8 % | 83.7 % | +15.9 | [+10.2, +22.6] |
| family flux1 | 93 | 75.3 % | 92.5 % | +17.2 | [+9.5, +25.5] |
| family flux2 | 80 | 70.0 % | 66.2 % | -3.7 | [-12.3, +5.1] |
| family gan-faces | 163 | 19.0 % | 63.2 % | +44.2 | [+36.2, +51.5] |
| family gpt-4o-image-1 | 46 | 69.6 % | 80.4 % | +10.9 | [+0.0, +21.7] |
| family gpt-image-1.5-2 | 62 | 43.5 % | 58.1 % | +14.5 | [+4.8, +24.2] |
| family hidream | 52 | 96.2 % | 100.0 % | +3.8 | [+0.0, +10.0] |
| family hunyuanimage | 29 | 82.8 % | 96.6 % | +13.8 | [+3.4, +27.6] |
| family ideogram | 52 | 76.9 % | 96.2 % | +19.2 | [+7.8, +30.8] |
| family imagen3 | 60 | 46.7 % | 90.0 % | +43.3 | [+30.0, +58.3] |
| family imagen4 | 118 | 84.7 % | 94.1 % | +9.3 | [+2.7, +15.8] |
| family legacy | 163 | 36.2 % | 85.3 % | +49.1 | [+42.1, +56.8] |
| family long-tail | 203 | 64.5 % | 88.7 % | +24.1 | [+16.8, +31.7] |
| family midjourney | 163 | 65.6 % | 98.2 % | +32.5 | [+24.8, +40.0] |
| family modern-t2i-faces | 124 | 86.3 % | 94.4 % | +8.1 | [+3.1, +13.1] |
| family nano-banana | 130 | 63.1 % | 76.9 % | +13.8 | [+6.9, +21.7] |
| family partial-edits | 175 | 49.1 % | 83.4 % | +34.3 | [+25.7, +42.9] |
| family personalised | 124 | 30.6 % | 96.0 % | +65.3 | [+56.5, +73.4] |
| family qwen-image | 113 | 75.2 % | 85.0 % | +9.7 | [+1.8, +17.4] |
| family recraft | 57 | 61.4 % | 94.7 % | +33.3 | [+21.3, +45.9] |
| family sd3 | 84 | 46.4 % | 100.0 % | +53.6 | [+42.0, +64.6] |
| family seedream3 | 41 | 51.2 % | 95.1 % | +43.9 | [+29.3, +59.0] |
| family seedream4 | 94 | 37.2 % | 88.3 % | +51.1 | [+40.4, +61.5] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

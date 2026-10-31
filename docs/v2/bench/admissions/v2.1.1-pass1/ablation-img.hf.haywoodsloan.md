# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.haywoodsloan GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +26.0 pts, CI [+22.4, +29.5] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.1 pts, CI [-1.7, +1.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1, R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 96.1 % | +13.7 |
| F2 | 70 | 27.1 % | 47.1 % | +20.0 |
| F3 | 57 | 14.0 % | 38.6 % | +24.6 |
| F4 | 118 | 38.1 % | 72.9 % | +34.7 |
| F5 | 80 | 75.0 % | 83.8 % | +8.8 |
| F6 | 106 | 28.3 % | 71.7 % | +43.4 |
| F7 | 106 | 50.9 % | 87.7 % | +36.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.1 %, with 71.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.9 % | 6.1 % | +4.2 | [-2.5, +9.2] |
| R2 | 612 | 5.6 % | 3.9 % | -1.6 | [-3.9, +2.2] |
| R3 | 589 | 3.9 % | 3.6 % | -0.3 | [-2.1, +1.3] |
| R4 | 493 | 8.9 % | 3.0 % | -5.9 | [-8.5, -3.2] |
| R5 | 548 | 1.3 % | 4.7 % | +3.5 | [+1.8, +5.3] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-15.4, +15.4] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 68.9 % | +4.3 | [+0.0, +8.7] |
| threat faceswap | 272 | 59.2 % | 65.4 % | +6.2 | [+3.3, +9.6] |
| threat gan_face | 153 | 51.6 % | 56.9 % | +5.2 | [+1.3, +9.8] |
| threat personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-5.2, +1.7] |
| threat t2i | 1600 | 61.6 % | 79.8 % | +18.2 | [+16.1, +20.4] |
| family faceswap-stills | 272 | 59.2 % | 65.4 % | +6.2 | [+3.3, +9.6] |
| family flux1 | 85 | 61.2 % | 75.3 % | +14.1 | [+6.1, +22.4] |
| family flux2 | 70 | 28.6 % | 44.3 % | +15.7 | [+6.1, +25.0] |
| family gan-faces | 153 | 51.6 % | 56.9 % | +5.2 | [+1.3, +9.8] |
| family gpt-4o-image-1 | 41 | 17.1 % | 58.5 % | +41.5 | [+26.8, +56.1] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 36.8 % | +22.8 | [+12.3, +33.3] |
| family hidream | 50 | 80.0 % | 90.0 % | +10.0 | [+0.0, +22.0] |
| family hunyuanimage | 24 | 100.0 % | 95.8 % | -4.2 | [-12.5, +0.0] |
| family ideogram | 47 | 76.6 % | 85.1 % | +8.5 | [+0.0, +19.1] |
| family imagen3 | 53 | 49.1 % | 69.8 % | +20.8 | [+5.7, +35.8] |
| family imagen4 | 106 | 52.8 % | 82.1 % | +29.2 | [+18.3, +39.8] |
| family legacy | 145 | 86.9 % | 92.4 % | +5.5 | [+1.4, +9.8] |
| family long-tail | 186 | 65.1 % | 84.4 % | +19.4 | [+12.8, +26.4] |
| family midjourney | 153 | 82.4 % | 95.4 % | +13.1 | [+7.5, +18.8] |
| family modern-t2i-faces | 115 | 54.8 % | 87.0 % | +32.2 | [+23.1, +41.4] |
| family nano-banana | 118 | 37.3 % | 66.1 % | +28.8 | [+19.5, +39.0] |
| family partial-edits | 161 | 64.6 % | 68.9 % | +4.3 | [+0.0, +8.7] |
| family personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-5.2, +1.7] |
| family qwen-image | 106 | 28.3 % | 67.9 % | +39.6 | [+30.1, +49.0] |
| family recraft | 54 | 87.0 % | 92.6 % | +5.6 | [-1.9, +13.5] |
| family sd3 | 75 | 97.3 % | 96.0 % | -1.3 | [-5.6, +2.7] |
| family seedream3 | 35 | 77.1 % | 85.7 % | +8.6 | [-2.9, +20.0] |
| family seedream4 | 80 | 75.0 % | 82.5 % | +7.5 | [+2.5, +13.9] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

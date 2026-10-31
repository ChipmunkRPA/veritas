# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +11.3 pts, CI [+8.2, +14.4] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family gpt-4o-image-1, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat faceswap, threat t2i |
| A4 | fail | pooled reals: Δ FPR +1.2 pts, CI [+0.2, +2.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 74.5 % | 82.4 % | +7.8 |
| F2 | 70 | 17.1 % | 27.1 % | +10.0 |
| F3 | 57 | 8.8 % | 14.0 % | +5.3 |
| F4 | 118 | 22.9 % | 38.1 % | +15.3 |
| F5 | 80 | 70.0 % | 75.0 % | +5.0 |
| F6 | 106 | 22.6 % | 28.3 % | +5.7 |
| F7 | 106 | 20.8 % | 50.9 % | +30.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 33.8 %, with 45.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.2 % | 1.9 % | -2.4 | [-4.2, +1.0] |
| R2 | 612 | 5.1 % | 5.6 % | +0.5 | [-2.8, +2.2] |
| R3 | 589 | 4.6 % | 3.9 % | -0.7 | [-2.1, +0.7] |
| R4 | 493 | 1.6 % | 8.9 % | +7.3 | [+4.9, +9.5] |
| R5 | 548 | 0.0 % | 1.3 % | +1.3 | [+0.4, +2.4] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-21.4, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 55.9 % | 64.6 % | +8.7 | [+4.3, +13.7] |
| threat faceswap | 272 | 39.7 % | 59.2 % | +19.5 | [+11.6, +28.3] |
| threat gan_face | 153 | 45.8 % | 51.6 % | +5.9 | [+2.0, +10.5] |
| threat personalised | 115 | 93.0 % | 96.5 % | +3.5 | [-0.9, +7.8] |
| threat t2i | 1600 | 42.2 % | 61.6 % | +19.4 | [+17.3, +21.4] |
| family faceswap-stills | 272 | 39.7 % | 59.2 % | +19.5 | [+11.6, +28.3] |
| family flux1 | 85 | 36.5 % | 61.2 % | +24.7 | [+14.5, +34.1] |
| family flux2 | 70 | 15.7 % | 28.6 % | +12.9 | [+5.6, +21.2] |
| family gan-faces | 153 | 45.8 % | 51.6 % | +5.9 | [+2.0, +10.5] |
| family gpt-4o-image-1 | 41 | 0.0 % | 17.1 % | +17.1 | [+7.3, +29.3] |
| family gpt-image-1.5-2 | 57 | 7.0 % | 14.0 % | +7.0 | [+0.0, +15.8] |
| family hidream | 50 | 56.0 % | 80.0 % | +24.0 | [+13.0, +36.0] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 48.9 % | 76.6 % | +27.7 | [+15.4, +41.3] |
| family imagen3 | 53 | 24.5 % | 49.1 % | +24.5 | [+11.3, +37.7] |
| family imagen4 | 106 | 17.0 % | 52.8 % | +35.8 | [+26.7, +45.9] |
| family legacy | 145 | 78.6 % | 86.9 % | +8.3 | [+3.4, +13.6] |
| family long-tail | 186 | 46.2 % | 65.1 % | +18.8 | [+12.6, +25.3] |
| family midjourney | 153 | 66.7 % | 82.4 % | +15.7 | [+10.1, +21.6] |
| family modern-t2i-faces | 115 | 30.4 % | 54.8 % | +24.3 | [+18.5, +30.8] |
| family nano-banana | 118 | 19.5 % | 37.3 % | +17.8 | [+11.0, +25.4] |
| family partial-edits | 161 | 55.9 % | 64.6 % | +8.7 | [+4.3, +13.7] |
| family personalised | 115 | 93.0 % | 96.5 % | +3.5 | [-0.9, +7.8] |
| family qwen-image | 106 | 17.9 % | 28.3 % | +10.4 | [+3.8, +17.3] |
| family recraft | 54 | 51.9 % | 87.0 % | +35.2 | [+23.1, +50.0] |
| family sd3 | 75 | 69.3 % | 97.3 % | +28.0 | [+18.1, +38.7] |
| family seedream3 | 35 | 42.9 % | 77.1 % | +34.3 | [+18.9, +50.0] |
| family seedream4 | 80 | 63.7 % | 75.0 % | +11.3 | [+2.5, +21.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.5 pts, CI [-1.1, +4.1] |
| A3b | pass | family gan-faces, family seedream4, threat gan_face |
| A4 | fail | pooled reals: Δ FPR +0.4 pts, CI [-0.6, +1.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 79.7 % | 82.4 % | +2.6 |
| F2 | 70 | 30.0 % | 27.1 % | -2.9 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 41.5 % | 38.1 % | -3.4 |
| F5 | 80 | 57.5 % | 75.0 % | +17.5 |
| F6 | 106 | 25.5 % | 28.3 % | +2.8 |
| F7 | 106 | 55.7 % | 50.9 % | -4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 43.7 %, with 45.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.7 % | 1.9 % | +0.2 | [-0.7, +2.3] |
| R2 | 612 | 7.7 % | 5.6 % | -2.1 | [-3.9, +1.1] |
| R3 | 589 | 4.2 % | 3.9 % | -0.3 | [-1.9, +1.2] |
| R4 | 493 | 4.5 % | 8.9 % | +4.5 | [+2.0, +7.1] |
| R5 | 548 | 0.9 % | 1.3 % | +0.4 | [-0.5, +1.3] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-10.7, +10.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 64.6 % | -0.6 | [-4.3, +3.1] |
| threat faceswap | 272 | 51.1 % | 59.2 % | +8.1 | [+4.1, +13.1] |
| threat gan_face | 153 | 20.9 % | 51.6 % | +30.7 | [+23.5, +37.9] |
| threat personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-2.6, +4.3] |
| threat t2i | 1600 | 60.6 % | 61.6 % | +1.0 | [-0.5, +2.5] |
| family faceswap-stills | 272 | 51.1 % | 59.2 % | +8.1 | [+4.1, +13.1] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [-6.7, +6.7] |
| family flux2 | 70 | 30.0 % | 28.6 % | -1.4 | [-8.8, +5.7] |
| family gan-faces | 153 | 20.9 % | 51.6 % | +30.7 | [+23.5, +37.9] |
| family gpt-4o-image-1 | 41 | 31.7 % | 17.1 % | -14.6 | [-26.8, -4.9] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 14.0 % | -1.8 | [-8.8, +3.5] |
| family hidream | 50 | 82.0 % | 80.0 % | -2.0 | [-9.6, +4.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 76.6 % | +6.4 | [+0.0, +13.7] |
| family imagen3 | 53 | 50.9 % | 49.1 % | -1.9 | [-13.2, +9.4] |
| family imagen4 | 106 | 56.6 % | 52.8 % | -3.8 | [-11.7, +4.6] |
| family legacy | 145 | 82.8 % | 86.9 % | +4.1 | [+1.4, +7.5] |
| family long-tail | 186 | 63.4 % | 65.1 % | +1.6 | [-2.4, +5.7] |
| family midjourney | 153 | 79.1 % | 82.4 % | +3.3 | [-0.7, +7.8] |
| family modern-t2i-faces | 115 | 58.3 % | 54.8 % | -3.5 | [-7.9, +0.9] |
| family nano-banana | 118 | 41.5 % | 37.3 % | -4.2 | [-10.9, +1.7] |
| family partial-edits | 161 | 65.2 % | 64.6 % | -0.6 | [-4.3, +3.1] |
| family personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-2.6, +4.3] |
| family qwen-image | 106 | 24.5 % | 28.3 % | +3.8 | [-1.9, +10.5] |
| family recraft | 54 | 81.5 % | 87.0 % | +5.6 | [+0.0, +13.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 74.3 % | 77.1 % | +2.9 | [-5.9, +14.3] |
| family seedream4 | 80 | 57.5 % | 75.0 % | +17.5 | [+9.8, +26.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

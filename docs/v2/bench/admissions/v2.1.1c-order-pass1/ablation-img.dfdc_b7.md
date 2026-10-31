# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.2 pts, CI [-2.2, -0.4] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | fail | pooled reals: Δ FPR -0.4 pts, CI [-1.2, +0.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 81.0 % | +0.0 |
| F2 | 70 | 28.6 % | 25.7 % | -2.9 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 34.7 % | 33.1 % | -1.7 |
| F5 | 80 | 75.0 % | 75.0 % | +0.0 |
| F6 | 106 | 30.2 % | 29.2 % | -0.9 |
| F7 | 106 | 53.8 % | 50.9 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.3 %, with 44.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.1 % | 1.4 % | -0.7 | [-1.4, +0.0] |
| R2 | 612 | 4.9 % | 4.7 % | -0.2 | [-0.7, +0.0] |
| R3 | 589 | 3.6 % | 3.7 % | +0.2 | [-0.5, +0.9] |
| R4 | 493 | 7.7 % | 15.0 % | +7.3 | [+4.9, +9.7] |
| R5 | 548 | 9.5 % | 1.5 % | -8.0 | [-10.6, -5.5] |
| R6 | 28 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 63.4 % | +0.0 | [-1.9, +1.9] |
| threat faceswap | 272 | 22.1 % | 62.9 % | +40.8 | [+24.0, +54.0] |
| threat gan_face | 153 | 52.9 % | 53.6 % | +0.7 | [-5.9, +7.8] |
| threat personalised | 115 | 98.3 % | 96.5 % | -1.7 | [-4.3, +0.0] |
| threat t2i | 1600 | 62.1 % | 61.1 % | -0.9 | [-1.8, -0.1] |
| family faceswap-stills | 272 | 22.1 % | 62.9 % | +40.8 | [+24.0, +54.0] |
| family flux1 | 85 | 62.4 % | 61.2 % | -1.2 | [-3.7, +0.0] |
| family flux2 | 70 | 28.6 % | 27.1 % | -1.4 | [-4.4, +0.0] |
| family gan-faces | 153 | 52.9 % | 53.6 % | +0.7 | [-5.9, +7.8] |
| family gpt-4o-image-1 | 41 | 12.2 % | 7.3 % | -4.9 | [-12.2, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 78.0 % | 80.0 % | +2.0 | [+0.0, +6.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 45.3 % | 41.5 % | -3.8 | [-9.4, +0.0] |
| family imagen4 | 106 | 57.5 % | 50.9 % | -6.6 | [-11.9, -1.9] |
| family legacy | 145 | 87.6 % | 96.6 % | +9.0 | [+4.9, +13.6] |
| family long-tail | 186 | 64.0 % | 63.4 % | -0.5 | [-1.6, +0.0] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 65.2 % | 54.8 % | -10.4 | [-16.3, -5.3] |
| family nano-banana | 118 | 34.7 % | 32.2 % | -2.5 | [-5.8, +0.0] |
| family partial-edits | 161 | 63.4 % | 63.4 % | +0.0 | [-1.9, +1.9] |
| family personalised | 115 | 98.3 % | 96.5 % | -1.7 | [-4.3, +0.0] |
| family qwen-image | 106 | 29.2 % | 29.2 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 74.3 % | 74.3 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

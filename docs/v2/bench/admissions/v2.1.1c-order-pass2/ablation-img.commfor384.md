# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.commfor384 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.8 pts, CI [-4.7, +1.0] |
| A3b | fail | family gan-faces, family seedream4, threat gan_face |
| A4 | fail | pooled reals: Δ FPR -0.4 pts, CI [-1.6, +1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 81.0 % | -0.7 |
| F2 | 70 | 30.0 % | 25.7 % | -4.3 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 44.9 % | 33.1 % | -11.9 |
| F5 | 80 | 61.3 % | 75.0 % | +13.7 |
| F6 | 106 | 27.4 % | 29.2 % | +1.9 |
| F7 | 106 | 60.4 % | 50.9 % | -9.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 44.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 1.4 % | -1.4 | [-3.3, +1.3] |
| R2 | 612 | 8.8 % | 4.7 % | -4.1 | [-6.9, +0.0] |
| R3 | 589 | 4.9 % | 3.7 % | -1.2 | [-2.8, +0.3] |
| R4 | 493 | 8.7 % | 15.0 % | +6.3 | [+3.2, +9.3] |
| R5 | 548 | 1.5 % | 1.5 % | +0.0 | [-0.9, +1.1] |
| R6 | 28 | 14.3 % | 3.6 % | -10.7 | [-25.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 63.4 % | -3.1 | [-7.5, +1.2] |
| threat faceswap | 272 | 55.5 % | 62.9 % | +7.4 | [+3.9, +11.6] |
| threat gan_face | 153 | 25.5 % | 53.6 % | +28.1 | [+20.9, +35.3] |
| threat personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-2.6, +4.3] |
| threat t2i | 1600 | 63.8 % | 61.1 % | -2.7 | [-4.5, -1.1] |
| family faceswap-stills | 272 | 55.5 % | 62.9 % | +7.4 | [+3.9, +11.6] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [-5.7, +5.7] |
| family flux2 | 70 | 30.0 % | 27.1 % | -2.9 | [-10.8, +5.3] |
| family gan-faces | 153 | 25.5 % | 53.6 % | +28.1 | [+20.9, +35.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 7.3 % | -26.8 | [-41.5, -14.6] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 14.0 % | -1.8 | [-8.8, +3.5] |
| family hidream | 50 | 82.0 % | 80.0 % | -2.0 | [-9.6, +4.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 74.5 % | -2.1 | [-11.4, +6.7] |
| family imagen3 | 53 | 52.8 % | 41.5 % | -11.3 | [-24.5, +0.0] |
| family imagen4 | 106 | 59.4 % | 50.9 % | -8.5 | [-16.8, +0.0] |
| family legacy | 145 | 89.0 % | 96.6 % | +7.6 | [+3.0, +12.2] |
| family long-tail | 186 | 65.6 % | 63.4 % | -2.2 | [-7.1, +2.6] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [-4.8, +4.6] |
| family modern-t2i-faces | 115 | 66.1 % | 54.8 % | -11.3 | [-16.7, -6.1] |
| family nano-banana | 118 | 44.9 % | 32.2 % | -12.7 | [-20.7, -5.1] |
| family partial-edits | 161 | 66.5 % | 63.4 % | -3.1 | [-7.5, +1.2] |
| family personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-2.6, +4.3] |
| family qwen-image | 106 | 28.3 % | 29.2 % | +0.9 | [-5.8, +8.4] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [-5.6, +5.6] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 74.3 % | -8.6 | [-22.9, +5.9] |
| family seedream4 | 80 | 61.3 % | 75.0 % | +13.7 | [+5.1, +23.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -4.0 pts, CI [-5.7, -2.6] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | fail | pooled reals: Δ FPR +0.5 pts, CI [-0.1, +1.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 82.4 % | -2.6 |
| F2 | 70 | 31.4 % | 27.1 % | -4.3 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 43.2 % | 38.1 % | -5.1 |
| F5 | 80 | 76.2 % | 75.0 % | -1.2 |
| F6 | 106 | 32.1 % | 28.3 % | -3.8 |
| F7 | 106 | 60.4 % | 50.9 % | -9.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 49.2 %, with 45.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.6 % | 1.9 % | -0.7 | [-1.4, +0.0] |
| R2 | 612 | 7.0 % | 5.6 % | -1.5 | [-2.4, -0.6] |
| R3 | 589 | 4.1 % | 3.9 % | -0.2 | [-0.5, +0.0] |
| R4 | 493 | 4.3 % | 8.9 % | +4.7 | [+2.6, +6.9] |
| R5 | 548 | 0.5 % | 1.3 % | +0.7 | [-0.2, +1.8] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-16.7, +7.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 64.6 % | -1.2 | [-3.1, +0.0] |
| threat faceswap | 272 | 15.8 % | 59.2 % | +43.4 | [+28.4, +55.5] |
| threat gan_face | 153 | 44.4 % | 51.6 % | +7.2 | [+3.3, +11.8] |
| threat personalised | 115 | 99.1 % | 96.5 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 63.8 % | 61.6 % | -2.2 | [-3.0, -1.4] |
| family faceswap-stills | 272 | 15.8 % | 59.2 % | +43.4 | [+28.4, +55.5] |
| family flux1 | 85 | 62.4 % | 61.2 % | -1.2 | [-3.7, +0.0] |
| family flux2 | 70 | 31.4 % | 28.6 % | -2.9 | [-7.2, +0.0] |
| family gan-faces | 153 | 44.4 % | 51.6 % | +7.2 | [+3.3, +11.8] |
| family gpt-4o-image-1 | 41 | 17.1 % | 17.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 14.0 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 49.1 % | -3.8 | [-9.4, +0.0] |
| family imagen4 | 106 | 58.5 % | 52.8 % | -5.7 | [-10.4, -1.9] |
| family legacy | 145 | 85.5 % | 86.9 % | +1.4 | [+0.0, +3.4] |
| family long-tail | 186 | 66.7 % | 65.1 % | -1.6 | [-3.4, +0.0] |
| family midjourney | 153 | 83.7 % | 82.4 % | -1.3 | [-3.3, +0.0] |
| family modern-t2i-faces | 115 | 65.2 % | 54.8 % | -10.4 | [-16.1, -5.2] |
| family nano-banana | 118 | 41.5 % | 37.3 % | -4.2 | [-8.4, -0.8] |
| family partial-edits | 161 | 65.8 % | 64.6 % | -1.2 | [-3.1, +0.0] |
| family personalised | 115 | 99.1 % | 96.5 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 30.2 % | 28.3 % | -1.9 | [-4.7, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 77.1 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

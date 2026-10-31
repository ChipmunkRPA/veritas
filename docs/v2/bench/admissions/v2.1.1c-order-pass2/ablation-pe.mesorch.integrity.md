# Ablation: pe.mesorch.integrity

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,pe.mesorch.integrity GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=pe.mesorch.integrity ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.9 pts, CI [-2.2, +0.5] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.5 pts, CI [-1.3, +0.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 80.4 % | -1.3 |
| F2 | 70 | 30.0 % | 31.4 % | +1.4 |
| F3 | 57 | 15.8 % | 17.5 % | +1.8 |
| F4 | 118 | 44.9 % | 43.2 % | -1.7 |
| F5 | 80 | 61.3 % | 58.8 % | -2.5 |
| F6 | 106 | 27.4 % | 26.4 % | -0.9 |
| F7 | 106 | 60.4 % | 57.5 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 45.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 1.9 % | -0.9 | [-1.9, +0.0] |
| R2 | 612 | 8.8 % | 7.2 % | -1.6 | [-4.1, +0.1] |
| R3 | 589 | 4.9 % | 3.6 % | -1.4 | [-2.5, -0.3] |
| R4 | 493 | 8.7 % | 7.3 % | -1.4 | [-3.7, +1.0] |
| R5 | 548 | 1.5 % | 4.6 % | +3.1 | [+1.5, +4.7] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 65.2 % | -1.2 | [-3.7, +1.2] |
| threat faceswap | 272 | 55.5 % | 56.2 % | +0.7 | [-5.3, +6.3] |
| threat gan_face | 153 | 25.5 % | 23.5 % | -2.0 | [-7.2, +3.3] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 63.8 % | 58.9 % | -4.9 | [-6.0, -3.8] |
| family faceswap-stills | 272 | 55.5 % | 56.2 % | +0.7 | [-5.3, +6.3] |
| family flux1 | 85 | 61.2 % | 58.8 % | -2.4 | [-5.9, +0.0] |
| family flux2 | 70 | 30.0 % | 28.6 % | -1.4 | [-6.7, +2.9] |
| family gan-faces | 153 | 25.5 % | 23.5 % | -2.0 | [-7.2, +3.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 26.8 % | -7.3 | [-17.1, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 82.0 % | 72.0 % | -10.0 | [-20.0, -2.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 70.2 % | -6.4 | [-14.6, +0.0] |
| family imagen3 | 53 | 52.8 % | 49.1 % | -3.8 | [-9.4, +0.0] |
| family imagen4 | 106 | 59.4 % | 51.9 % | -7.5 | [-13.1, -2.9] |
| family legacy | 145 | 89.0 % | 84.8 % | -4.1 | [-8.9, +0.6] |
| family long-tail | 186 | 65.6 % | 61.3 % | -4.3 | [-7.6, -1.6] |
| family midjourney | 153 | 81.7 % | 76.5 % | -5.2 | [-9.1, -1.9] |
| family modern-t2i-faces | 115 | 66.1 % | 58.3 % | -7.8 | [-11.9, -4.2] |
| family nano-banana | 118 | 44.9 % | 39.0 % | -5.9 | [-10.3, -2.5] |
| family partial-edits | 161 | 66.5 % | 65.2 % | -1.2 | [-3.7, +1.2] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 23.6 % | -4.7 | [-9.1, -1.0] |
| family recraft | 54 | 87.0 % | 79.6 % | -7.4 | [-14.5, -1.9] |
| family sd3 | 75 | 97.3 % | 96.0 % | -1.3 | [-3.9, +0.0] |
| family seedream3 | 35 | 82.9 % | 74.3 % | -8.6 | [-17.1, +0.0] |
| family seedream4 | 80 | 61.3 % | 57.5 % | -3.8 | [-9.9, +1.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

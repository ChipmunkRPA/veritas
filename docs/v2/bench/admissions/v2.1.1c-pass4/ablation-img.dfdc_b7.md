# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.dfdc_b7 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -7.3 pts, CI [-9.3, -5.3] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.7, +0.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R4; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 24.2 % | -13.1 |
| F2 | 70 | 42.9 % | 37.1 % | -5.7 |
| F3 | 57 | 40.4 % | 29.8 % | -10.5 |
| F4 | 118 | 49.2 % | 43.2 % | -5.9 |
| F5 | 80 | 40.0 % | 33.8 % | -6.2 |
| F6 | 106 | 34.9 % | 29.2 % | -5.7 |
| F7 | 106 | 32.1 % | 28.3 % | -3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 32.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 3.1 % | -1.4 | [-2.8, -0.2] |
| R2 | 612 | 4.2 % | 2.6 % | -1.6 | [-3.6, -0.6] |
| R3 | 589 | 10.0 % | 9.5 % | -0.5 | [-2.2, +1.0] |
| R4 | 493 | 4.9 % | 7.9 % | +3.0 | [+1.4, +4.7] |
| R5 | 548 | 1.5 % | 1.8 % | +0.4 | [-0.4, +1.1] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [-10.7, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.8 % | 15.5 % | -1.2 | [-3.7, +1.2] |
| threat faceswap | 272 | 0.4 % | 13.6 % | +13.2 | [+4.5, +24.1] |
| threat gan_face | 153 | 0.0 % | 0.7 % | +0.7 | [+0.0, +2.0] |
| threat personalised | 115 | 9.6 % | 7.8 % | -1.7 | [-4.3, +0.0] |
| threat t2i | 1600 | 36.4 % | 34.9 % | -1.4 | [-2.4, -0.5] |
| family faceswap-stills | 272 | 0.4 % | 13.6 % | +13.2 | [+4.5, +24.1] |
| family flux1 | 85 | 58.8 % | 50.6 % | -8.2 | [-14.5, -2.4] |
| family flux2 | 70 | 42.9 % | 41.4 % | -1.4 | [-4.6, +0.0] |
| family gan-faces | 153 | 0.0 % | 0.7 % | +0.7 | [+0.0, +2.0] |
| family gpt-4o-image-1 | 41 | 41.5 % | 36.6 % | -4.9 | [-12.2, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 35.1 % | -5.3 | [-12.3, +0.0] |
| family hidream | 50 | 62.0 % | 62.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 62.5 % | 62.5 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 27.7 % | 25.5 % | -2.1 | [-6.7, +0.0] |
| family imagen3 | 53 | 18.9 % | 15.1 % | -3.8 | [-9.4, +0.0] |
| family imagen4 | 106 | 31.1 % | 30.2 % | -0.9 | [-4.6, +2.0] |
| family legacy | 145 | 39.3 % | 43.4 % | +4.1 | [-0.7, +8.9] |
| family long-tail | 186 | 40.3 % | 41.9 % | +1.6 | [-1.0, +4.5] |
| family midjourney | 153 | 35.9 % | 32.0 % | -3.9 | [-7.3, -1.3] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 48.3 % | 45.8 % | -2.5 | [-5.9, +0.0] |
| family partial-edits | 161 | 16.8 % | 15.5 % | -1.2 | [-3.7, +1.2] |
| family personalised | 115 | 9.6 % | 7.8 % | -1.7 | [-4.3, +0.0] |
| family qwen-image | 106 | 33.0 % | 33.0 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 27.8 % | 27.8 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 29.3 % | 26.7 % | -2.7 | [-6.9, +0.0] |
| family seedream3 | 35 | 28.6 % | 20.0 % | -8.6 | [-18.9, +0.0] |
| family seedream4 | 80 | 38.8 % | 37.5 % | -1.3 | [-6.2, +2.5] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

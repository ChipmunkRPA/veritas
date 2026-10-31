# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.2 pts, CI [-2.2, -0.5] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -1.9 pts, CI [-2.7, -1.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 81.7 % | -0.7 |
| F2 | 70 | 31.4 % | 30.0 % | -1.4 |
| F3 | 57 | 17.5 % | 15.8 % | -1.8 |
| F4 | 118 | 44.9 % | 44.9 % | +0.0 |
| F5 | 80 | 61.3 % | 61.3 % | +0.0 |
| F6 | 106 | 28.3 % | 27.4 % | -0.9 |
| F7 | 106 | 64.2 % | 60.4 % | -3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 47.1 %, with 45.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.8 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 8.8 % | 8.8 % | +0.0 | [-0.6, +0.3] |
| R3 | 589 | 4.8 % | 4.9 % | +0.2 | [+0.0, +0.5] |
| R4 | 493 | 8.7 % | 8.7 % | +0.0 | [-0.6, +0.6] |
| R5 | 548 | 11.1 % | 1.5 % | -9.7 | [-12.6, -7.1] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 28.3 % | 55.5 % | +27.2 | [+11.7, +42.3] |
| threat gan_face | 153 | 30.7 % | 25.5 % | -5.2 | [-11.8, +1.3] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 64.6 % | 63.8 % | -0.7 | [-1.2, -0.3] |
| family faceswap-stills | 272 | 28.3 % | 55.5 % | +27.2 | [+11.7, +42.3] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 31.4 % | 30.0 % | -1.4 | [-4.5, +0.0] |
| family gan-faces | 153 | 30.7 % | 25.5 % | -5.2 | [-11.8, +1.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 34.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 63.2 % | 59.4 % | -3.8 | [-7.5, -0.9] |
| family legacy | 145 | 89.0 % | 89.0 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 65.6 % | 65.6 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 82.4 % | 81.7 % | -0.7 | [-2.0, +0.0] |
| family modern-t2i-faces | 115 | 71.3 % | 66.1 % | -5.2 | [-9.8, -1.7] |
| family nano-banana | 118 | 44.9 % | 44.9 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 28.3 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 61.3 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

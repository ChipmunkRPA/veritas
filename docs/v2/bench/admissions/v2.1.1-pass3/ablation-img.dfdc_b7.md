# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.8 pts, CI [-2.4, +0.8] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.3, +0.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 85.0 % | +0.0 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 22.8 % | 19.3 % | -3.5 |
| F4 | 118 | 52.5 % | 51.7 % | -0.8 |
| F5 | 80 | 63.7 % | 63.7 % | +0.0 |
| F6 | 106 | 39.6 % | 40.6 % | +0.9 |
| F7 | 106 | 72.6 % | 70.8 % | -1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.2 %, with 53.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 2.4 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 2.8 % | 2.5 % | -0.3 | [-0.6, +0.0] |
| R3 | 589 | 5.1 % | 4.9 % | -0.2 | [-1.2, +0.9] |
| R4 | 493 | 4.7 % | 4.5 % | -0.2 | [-0.6, +0.0] |
| R5 | 548 | 0.5 % | 1.3 % | +0.7 | [+0.2, +1.5] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 64.6 % | +0.0 | [-1.9, +1.9] |
| threat faceswap | 272 | 5.9 % | 29.8 % | +23.9 | [+15.4, +34.8] |
| threat gan_face | 153 | 14.4 % | 9.8 % | -4.6 | [-9.2, -0.7] |
| threat personalised | 115 | 91.3 % | 91.3 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 65.4 % | 63.7 % | -1.7 | [-2.5, -0.9] |
| family faceswap-stills | 272 | 5.9 % | 29.8 % | +23.9 | [+15.4, +34.8] |
| family flux1 | 85 | 62.4 % | 62.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 38.6 % | 38.6 % | +0.0 | [-4.3, +4.2] |
| family gan-faces | 153 | 14.4 % | 9.8 % | -4.6 | [-9.2, -0.7] |
| family gpt-4o-image-1 | 41 | 31.7 % | 31.7 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 88.0 % | 84.0 % | -4.0 | [-10.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 70.2 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 50.9 % | -1.9 | [-5.7, +0.0] |
| family imagen4 | 106 | 65.1 % | 60.4 % | -4.7 | [-8.8, -1.0] |
| family legacy | 145 | 90.3 % | 91.7 % | +1.4 | [+0.0, +3.4] |
| family long-tail | 186 | 67.2 % | 67.7 % | +0.5 | [-1.2, +2.5] |
| family midjourney | 153 | 82.4 % | 79.7 % | -2.6 | [-5.4, -0.6] |
| family modern-t2i-faces | 115 | 73.0 % | 65.2 % | -7.8 | [-12.7, -3.5] |
| family nano-banana | 118 | 44.9 % | 40.7 % | -4.2 | [-8.5, -0.9] |
| family partial-edits | 161 | 64.6 % | 64.6 % | +0.0 | [-1.9, +1.9] |
| family personalised | 115 | 91.3 % | 91.3 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 34.9 % | 33.0 % | -1.9 | [-5.8, +1.9] |
| family recraft | 54 | 81.5 % | 79.6 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 77.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 57.5 % | 56.2 % | -1.2 | [-6.2, +2.6] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

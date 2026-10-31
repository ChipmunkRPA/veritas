# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.7 pts, CI [+1.4, +4.2] |
| A3b | pass | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.4, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 85.0 % | +1.3 |
| F2 | 70 | 37.1 % | 42.9 % | +5.7 |
| F3 | 57 | 15.8 % | 17.5 % | +1.8 |
| F4 | 118 | 48.3 % | 49.2 % | +0.8 |
| F5 | 80 | 76.2 % | 78.8 % | +2.5 |
| F6 | 106 | 39.6 % | 44.3 % | +4.7 |
| F7 | 106 | 66.0 % | 67.9 % | +1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 52.4 %, with 55.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.2 % | 3.5 % | -0.7 | [-1.4, +0.0] |
| R2 | 612 | 3.8 % | 3.8 % | +0.0 | [-0.7, +0.3] |
| R3 | 589 | 4.6 % | 4.9 % | +0.3 | [-0.5, +1.3] |
| R4 | 493 | 4.5 % | 5.5 % | +1.0 | [+0.2, +2.0] |
| R5 | 548 | 2.0 % | 1.3 % | -0.7 | [-1.8, +0.4] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.8 % | +0.6 | [-1.2, +3.1] |
| threat faceswap | 272 | 20.6 % | 51.5 % | +30.9 | [+20.7, +41.4] |
| threat gan_face | 153 | 43.8 % | 43.1 % | -0.7 | [-3.3, +2.0] |
| threat personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 67.6 % | 66.9 % | -0.7 | [-1.4, +0.0] |
| family faceswap-stills | 272 | 20.6 % | 51.5 % | +30.9 | [+20.7, +41.4] |
| family flux1 | 85 | 67.1 % | 67.1 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 37.1 % | 35.7 % | -1.4 | [-6.1, +2.9] |
| family gan-faces | 153 | 43.8 % | 43.1 % | -0.7 | [-3.3, +2.0] |
| family gpt-4o-image-1 | 41 | 22.0 % | 22.0 % | +0.0 | [-7.3, +7.3] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 17.5 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 86.0 % | 86.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 58.5 % | +5.7 | [+0.0, +13.2] |
| family imagen4 | 106 | 65.1 % | 63.2 % | -1.9 | [-6.5, +2.8] |
| family legacy | 145 | 95.2 % | 95.2 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 69.9 % | 69.4 % | -0.5 | [-1.7, +0.0] |
| family midjourney | 153 | 83.7 % | 83.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 72.2 % | 65.2 % | -7.0 | [-13.0, -1.0] |
| family nano-banana | 118 | 44.1 % | 41.5 % | -2.5 | [-5.9, +0.0] |
| family partial-edits | 161 | 65.2 % | 65.8 % | +0.6 | [-1.2, +3.1] |
| family personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 37.7 % | 38.7 % | +0.9 | [-1.9, +4.5] |
| family recraft | 54 | 85.2 % | 85.2 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +9.9 pts, CI [+7.7, +12.2] |
| A3b | pass | family imagen4, family modern-t2i-faces, family qwen-image |
| A4 | pass | pooled reals: Δ FPR -0.6 pts, CI [-1.4, +0.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 85.0 % | +2.6 |
| F2 | 70 | 27.1 % | 42.9 % | +15.7 |
| F3 | 57 | 14.0 % | 17.5 % | +3.5 |
| F4 | 118 | 38.1 % | 49.2 % | +11.0 |
| F5 | 80 | 75.0 % | 78.8 % | +3.7 |
| F6 | 106 | 28.3 % | 44.3 % | +16.0 |
| F7 | 106 | 50.9 % | 67.9 % | +17.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.1 %, with 55.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.9 % | 3.5 % | +1.7 | [-1.1, +3.2] |
| R2 | 612 | 5.6 % | 3.8 % | -1.8 | [-2.9, +0.0] |
| R3 | 589 | 3.9 % | 4.9 % | +1.0 | [-0.2, +2.3] |
| R4 | 493 | 8.9 % | 5.5 % | -3.4 | [-5.9, -1.0] |
| R5 | 548 | 1.3 % | 1.3 % | +0.0 | [-1.1, +1.1] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-11.5, +11.5] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.8 % | +1.2 | [-1.2, +3.7] |
| threat faceswap | 272 | 59.2 % | 51.5 % | -7.7 | [-18.0, +3.9] |
| threat gan_face | 153 | 51.6 % | 43.1 % | -8.5 | [-13.7, -3.9] |
| threat personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 61.6 % | 66.9 % | +5.2 | [+3.9, +6.6] |
| family faceswap-stills | 272 | 59.2 % | 51.5 % | -7.7 | [-18.0, +3.9] |
| family flux1 | 85 | 61.2 % | 67.1 % | +5.9 | [+1.2, +11.6] |
| family flux2 | 70 | 28.6 % | 35.7 % | +7.1 | [+0.0, +14.7] |
| family gan-faces | 153 | 51.6 % | 43.1 % | -8.5 | [-13.7, -3.9] |
| family gpt-4o-image-1 | 41 | 17.1 % | 22.0 % | +4.9 | [-7.3, +17.1] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 17.5 % | +3.5 | [+0.0, +8.8] |
| family hidream | 50 | 80.0 % | 86.0 % | +6.0 | [-2.1, +15.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 74.5 % | -2.1 | [-9.8, +4.4] |
| family imagen3 | 53 | 49.1 % | 58.5 % | +9.4 | [+0.0, +20.8] |
| family imagen4 | 106 | 52.8 % | 63.2 % | +10.4 | [+4.7, +16.5] |
| family legacy | 145 | 86.9 % | 95.2 % | +8.3 | [+3.4, +13.4] |
| family long-tail | 186 | 65.1 % | 69.4 % | +4.3 | [+0.0, +8.2] |
| family midjourney | 153 | 82.4 % | 83.7 % | +1.3 | [-1.4, +4.7] |
| family modern-t2i-faces | 115 | 54.8 % | 65.2 % | +10.4 | [+5.6, +15.8] |
| family nano-banana | 118 | 37.3 % | 41.5 % | +4.2 | [-0.8, +10.1] |
| family partial-edits | 161 | 64.6 % | 65.8 % | +1.2 | [-1.2, +3.7] |
| family personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 28.3 % | 38.7 % | +10.4 | [+4.8, +16.8] |
| family recraft | 54 | 87.0 % | 85.2 % | -1.9 | [-7.7, +3.7] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 77.1 % | 82.9 % | +5.7 | [-5.9, +16.7] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [-3.7, +3.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

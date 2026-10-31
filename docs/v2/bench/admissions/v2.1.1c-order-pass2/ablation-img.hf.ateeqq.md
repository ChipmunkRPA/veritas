# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +4.1 pts, CI [+1.5, +6.7] |
| A3b | pass | family modern-t2i-faces, family qwen-image |
| A4 | fail | pooled reals: Δ FPR -0.6 pts, CI [-2.1, +1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 82.4 % | +0.7 |
| F2 | 70 | 30.0 % | 35.7 % | +5.7 |
| F3 | 57 | 15.8 % | 19.3 % | +3.5 |
| F4 | 118 | 44.9 % | 47.5 % | +2.5 |
| F5 | 80 | 61.3 % | 57.5 % | -3.8 |
| F6 | 106 | 27.4 % | 38.7 % | +11.3 |
| F7 | 106 | 60.4 % | 68.9 % | +8.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 50.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.6 % | -0.2 | [-1.5, +1.6] |
| R2 | 612 | 8.8 % | 2.9 % | -5.9 | [-9.3, -0.8] |
| R3 | 589 | 4.9 % | 5.6 % | +0.7 | [-0.9, +2.2] |
| R4 | 493 | 8.7 % | 5.9 % | -2.8 | [-5.1, -0.6] |
| R5 | 548 | 1.5 % | 7.3 % | +5.8 | [+3.5, +8.2] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [-10.7, +10.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 65.8 % | -0.6 | [-3.7, +2.5] |
| threat faceswap | 272 | 55.5 % | 46.0 % | -9.6 | [-21.3, +3.4] |
| threat gan_face | 153 | 25.5 % | 17.6 % | -7.8 | [-13.7, -2.0] |
| threat personalised | 115 | 95.7 % | 94.8 % | -0.9 | [-3.5, +1.7] |
| threat t2i | 1600 | 63.8 % | 66.9 % | +3.1 | [+1.5, +4.7] |
| family faceswap-stills | 272 | 55.5 % | 46.0 % | -9.6 | [-21.3, +3.4] |
| family flux1 | 85 | 61.2 % | 63.5 % | +2.4 | [-1.2, +7.2] |
| family flux2 | 70 | 30.0 % | 40.0 % | +10.0 | [+0.0, +20.3] |
| family gan-faces | 153 | 25.5 % | 17.6 % | -7.8 | [-13.7, -2.0] |
| family gpt-4o-image-1 | 41 | 34.1 % | 31.7 % | -2.4 | [-14.6, +9.8] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [-5.3, +5.3] |
| family hidream | 50 | 82.0 % | 88.0 % | +6.0 | [-2.1, +14.8] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 72.3 % | -4.3 | [-13.0, +4.3] |
| family imagen3 | 53 | 52.8 % | 56.6 % | +3.8 | [-7.5, +13.3] |
| family imagen4 | 106 | 59.4 % | 66.0 % | +6.6 | [+2.0, +11.2] |
| family legacy | 145 | 89.0 % | 91.7 % | +2.8 | [-2.4, +7.7] |
| family long-tail | 186 | 65.6 % | 69.9 % | +4.3 | [-0.6, +8.9] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [-3.8, +3.7] |
| family modern-t2i-faces | 115 | 66.1 % | 76.5 % | +10.4 | [+4.8, +16.5] |
| family nano-banana | 118 | 44.9 % | 46.6 % | +1.7 | [-6.7, +9.4] |
| family partial-edits | 161 | 66.5 % | 65.8 % | -0.6 | [-3.7, +2.5] |
| family personalised | 115 | 95.7 % | 94.8 % | -0.9 | [-3.5, +1.7] |
| family qwen-image | 106 | 28.3 % | 38.7 % | +10.4 | [+3.7, +17.8] |
| family recraft | 54 | 87.0 % | 83.3 % | -3.7 | [-11.1, +3.6] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 82.9 % | 77.1 % | -5.7 | [-14.7, +0.0] |
| family seedream4 | 80 | 61.3 % | 58.8 % | -2.5 | [-9.0, +3.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

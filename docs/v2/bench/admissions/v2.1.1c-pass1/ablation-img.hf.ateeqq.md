# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +5.9 pts, CI [+3.7, +8.1] |
| A3b | pass | family flux2, family gpt-4o-image-1, family imagen3, family imagen4, family modern-t2i-faces, family nano-banana, family qwen-image |
| A4 | fail | pooled reals: Δ FPR -0.2 pts, CI [-1.3, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 83.0 % | +2.0 |
| F2 | 70 | 25.7 % | 31.4 % | +5.7 |
| F3 | 57 | 14.0 % | 12.3 % | -1.8 |
| F4 | 118 | 33.1 % | 41.5 % | +8.5 |
| F5 | 80 | 75.0 % | 76.2 % | +1.2 |
| F6 | 106 | 29.2 % | 40.6 % | +11.3 |
| F7 | 106 | 50.9 % | 65.1 % | +14.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 50.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 3.5 % | +2.1 | [-1.2, +4.6] |
| R2 | 612 | 4.7 % | 3.8 % | -1.0 | [-1.8, +0.7] |
| R3 | 589 | 3.7 % | 4.6 % | +0.8 | [-0.5, +2.3] |
| R4 | 493 | 15.0 % | 5.1 % | -9.9 | [-13.0, -6.9] |
| R5 | 548 | 1.5 % | 7.5 % | +6.0 | [+3.8, +8.2] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 65.2 % | +1.9 | [+0.0, +4.3] |
| threat faceswap | 272 | 62.9 % | 62.5 % | -0.4 | [-12.6, +12.6] |
| threat gan_face | 153 | 53.6 % | 45.8 % | -7.8 | [-13.7, -2.6] |
| threat personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-4.3, +2.6] |
| threat t2i | 1600 | 61.1 % | 67.2 % | +6.1 | [+4.6, +7.5] |
| family faceswap-stills | 272 | 62.9 % | 62.5 % | -0.4 | [-12.6, +12.6] |
| family flux1 | 85 | 61.2 % | 64.7 % | +3.5 | [-1.2, +9.0] |
| family flux2 | 70 | 27.1 % | 37.1 % | +10.0 | [+4.2, +17.1] |
| family gan-faces | 153 | 53.6 % | 45.8 % | -7.8 | [-13.7, -2.6] |
| family gpt-4o-image-1 | 41 | 7.3 % | 19.5 % | +12.2 | [+2.4, +22.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 15.8 % | +1.8 | [-3.5, +8.8] |
| family hidream | 50 | 80.0 % | 86.0 % | +6.0 | [-2.1, +15.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [-8.7, +8.5] |
| family imagen3 | 53 | 41.5 % | 54.7 % | +13.2 | [+1.9, +24.5] |
| family imagen4 | 106 | 50.9 % | 64.2 % | +13.2 | [+7.5, +19.3] |
| family legacy | 145 | 96.6 % | 95.2 % | -1.4 | [-5.4, +2.6] |
| family long-tail | 186 | 63.4 % | 71.0 % | +7.5 | [+4.0, +11.5] |
| family midjourney | 153 | 81.7 % | 82.4 % | +0.7 | [-2.0, +3.8] |
| family modern-t2i-faces | 115 | 54.8 % | 67.8 % | +13.0 | [+8.0, +18.1] |
| family nano-banana | 118 | 32.2 % | 42.4 % | +10.2 | [+4.2, +16.1] |
| family partial-edits | 161 | 63.4 % | 65.2 % | +1.9 | [+0.0, +4.3] |
| family personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-4.3, +2.6] |
| family qwen-image | 106 | 29.2 % | 40.6 % | +11.3 | [+5.6, +17.9] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [-9.3, +8.9] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 82.9 % | +8.6 | [+0.0, +18.4] |
| family seedream4 | 80 | 75.0 % | 76.2 % | +1.2 | [-2.5, +5.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

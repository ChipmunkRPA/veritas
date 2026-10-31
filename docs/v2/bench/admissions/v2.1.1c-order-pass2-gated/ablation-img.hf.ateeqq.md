# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +10.6 pts, CI [+7.7, +13.5] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.5 pts, CI [-1.8, +0.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R3; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 85.6 % | +3.9 |
| F2 | 70 | 30.0 % | 45.7 % | +15.7 |
| F3 | 57 | 15.8 % | 28.1 % | +12.3 |
| F4 | 118 | 44.9 % | 56.8 % | +11.9 |
| F5 | 80 | 61.3 % | 66.2 % | +5.0 |
| F6 | 106 | 27.4 % | 42.5 % | +15.1 |
| F7 | 106 | 60.4 % | 70.8 % | +10.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 56.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 4.0 % | +1.2 | [-0.3, +2.3] |
| R2 | 612 | 8.8 % | 4.7 % | -4.1 | [-6.9, +0.3] |
| R3 | 589 | 4.9 % | 7.3 % | +2.4 | [+0.7, +4.2] |
| R4 | 493 | 8.7 % | 6.3 % | -2.4 | [-4.7, -0.2] |
| R5 | 548 | 1.5 % | 1.8 % | +0.4 | [+0.0, +0.9] |
| R6 | 28 | 14.3 % | 21.4 % | +7.1 | [+0.0, +17.9] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 67.1 % | +0.6 | [-1.9, +3.7] |
| threat faceswap | 272 | 55.5 % | 53.3 % | -2.2 | [-4.3, -0.4] |
| threat gan_face | 153 | 25.5 % | 24.2 % | -1.3 | [-3.3, +0.0] |
| threat personalised | 115 | 95.7 % | 96.5 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 63.8 % | 67.0 % | +3.2 | [+1.7, +4.6] |
| family faceswap-stills | 272 | 55.5 % | 53.3 % | -2.2 | [-4.3, -0.4] |
| family flux1 | 85 | 61.2 % | 63.5 % | +2.4 | [-1.2, +7.2] |
| family flux2 | 70 | 30.0 % | 40.0 % | +10.0 | [+0.0, +20.6] |
| family gan-faces | 153 | 25.5 % | 24.2 % | -1.3 | [-3.3, +0.0] |
| family gpt-4o-image-1 | 41 | 34.1 % | 41.5 % | +7.3 | [-2.4, +19.5] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 19.3 % | +3.5 | [-3.5, +10.5] |
| family hidream | 50 | 82.0 % | 88.0 % | +6.0 | [-2.1, +14.8] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [-9.4, +9.4] |
| family imagen4 | 106 | 59.4 % | 64.2 % | +4.7 | [+1.0, +9.3] |
| family legacy | 145 | 89.0 % | 91.7 % | +2.8 | [-2.4, +7.7] |
| family long-tail | 186 | 65.6 % | 69.9 % | +4.3 | [+0.0, +8.4] |
| family midjourney | 153 | 81.7 % | 83.0 % | +1.3 | [-2.0, +5.1] |
| family modern-t2i-faces | 115 | 66.1 % | 68.7 % | +2.6 | [+0.0, +5.9] |
| family nano-banana | 118 | 44.9 % | 49.2 % | +4.2 | [-3.4, +12.0] |
| family partial-edits | 161 | 66.5 % | 67.1 % | +0.6 | [-1.9, +3.7] |
| family personalised | 115 | 95.7 % | 96.5 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 28.3 % | 36.8 % | +8.5 | [+1.9, +16.5] |
| family recraft | 54 | 87.0 % | 83.3 % | -3.7 | [-12.5, +5.4] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 82.9 % | 80.0 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 61.3 % | 61.3 % | +0.0 | [-6.2, +6.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

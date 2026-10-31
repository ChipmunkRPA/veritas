# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +9.8 pts, CI [+7.6, +12.1] |
| A3b | pass | family flux2, family gpt-4o-image-1, family hidream, family imagen3, family modern-t2i-faces, family nano-banana, family qwen-image |
| A4 | pass | pooled reals: Δ FPR -0.2 pts, CI [-1.2, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1, R3; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 86.3 % | +5.2 |
| F2 | 70 | 25.7 % | 38.6 % | +12.9 |
| F3 | 57 | 14.0 % | 19.3 % | +5.3 |
| F4 | 118 | 33.1 % | 48.3 % | +15.3 |
| F5 | 80 | 75.0 % | 78.8 % | +3.7 |
| F6 | 106 | 29.2 % | 42.5 % | +13.2 |
| F7 | 106 | 50.9 % | 64.2 % | +13.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 54.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 4.7 % | +3.3 | [-0.7, +5.1] |
| R2 | 612 | 4.7 % | 5.2 % | +0.5 | [-0.4, +2.3] |
| R3 | 589 | 3.7 % | 6.3 % | +2.5 | [+1.0, +4.2] |
| R4 | 493 | 15.0 % | 6.5 % | -8.5 | [-11.6, -5.5] |
| R5 | 548 | 1.5 % | 1.6 % | +0.2 | [-0.7, +1.1] |
| R6 | 28 | 3.6 % | 17.9 % | +14.3 | [+3.3, +29.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 65.8 % | +2.5 | [+0.6, +5.0] |
| threat faceswap | 272 | 62.9 % | 66.9 % | +4.0 | [+2.0, +6.7] |
| threat gan_face | 153 | 53.6 % | 56.9 % | +3.3 | [+0.0, +7.2] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-2.6, +2.6] |
| threat t2i | 1600 | 61.1 % | 68.2 % | +7.1 | [+5.7, +8.6] |
| family faceswap-stills | 272 | 62.9 % | 66.9 % | +4.0 | [+2.0, +6.7] |
| family flux1 | 85 | 61.2 % | 68.2 % | +7.1 | [+2.3, +13.3] |
| family flux2 | 70 | 27.1 % | 38.6 % | +11.4 | [+4.5, +19.1] |
| family gan-faces | 153 | 53.6 % | 56.9 % | +3.3 | [+0.0, +7.2] |
| family gpt-4o-image-1 | 41 | 7.3 % | 22.0 % | +14.6 | [+4.9, +26.8] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 19.3 % | +5.3 | [+0.0, +12.3] |
| family hidream | 50 | 80.0 % | 92.0 % | +12.0 | [+4.0, +22.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 76.6 % | +2.1 | [-4.4, +10.4] |
| family imagen3 | 53 | 41.5 % | 54.7 % | +13.2 | [+5.6, +22.6] |
| family imagen4 | 106 | 50.9 % | 60.4 % | +9.4 | [+4.0, +15.2] |
| family legacy | 145 | 96.6 % | 95.2 % | -1.4 | [-5.4, +2.6] |
| family long-tail | 186 | 63.4 % | 73.1 % | +9.7 | [+5.9, +13.9] |
| family midjourney | 153 | 81.7 % | 85.6 % | +3.9 | [+0.6, +7.8] |
| family modern-t2i-faces | 115 | 54.8 % | 65.2 % | +10.4 | [+5.7, +15.7] |
| family nano-banana | 118 | 32.2 % | 44.1 % | +11.9 | [+5.8, +17.9] |
| family partial-edits | 161 | 63.4 % | 65.8 % | +2.5 | [+0.6, +5.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-2.6, +2.6] |
| family qwen-image | 106 | 29.2 % | 39.6 % | +10.4 | [+4.7, +17.1] |
| family recraft | 54 | 87.0 % | 90.7 % | +3.7 | [-3.8, +13.0] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 82.9 % | +8.6 | [+0.0, +18.4] |
| family seedream4 | 80 | 75.0 % | 77.5 % | +2.5 | [-2.5, +7.6] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

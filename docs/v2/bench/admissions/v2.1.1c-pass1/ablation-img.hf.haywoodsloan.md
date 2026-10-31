# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.haywoodsloan GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +25.6 pts, CI [+22.3, +29.0] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family seedream3, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.3 pts, CI [-1.9, +1.8] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 96.1 % | +15.0 |
| F2 | 70 | 25.7 % | 45.7 % | +20.0 |
| F3 | 57 | 14.0 % | 36.8 % | +22.8 |
| F4 | 118 | 33.1 % | 67.8 % | +34.7 |
| F5 | 80 | 75.0 % | 85.0 % | +10.0 |
| F6 | 106 | 29.2 % | 71.7 % | +42.5 |
| F7 | 106 | 50.9 % | 84.9 % | +34.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 69.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 6.8 % | +5.4 | [-1.1, +11.0] |
| R2 | 612 | 4.7 % | 4.1 % | -0.7 | [-2.5, +2.4] |
| R3 | 589 | 3.7 % | 3.9 % | +0.2 | [-1.4, +1.7] |
| R4 | 493 | 15.0 % | 2.8 % | -12.2 | [-15.2, -9.1] |
| R5 | 548 | 1.5 % | 6.9 % | +5.5 | [+3.3, +7.8] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 70.2 % | +6.8 | [+2.5, +11.8] |
| threat faceswap | 272 | 62.9 % | 65.8 % | +2.9 | [+0.0, +6.3] |
| threat gan_face | 153 | 53.6 % | 58.2 % | +4.6 | [+0.0, +9.2] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-3.5, +3.5] |
| threat t2i | 1600 | 61.1 % | 79.9 % | +18.8 | [+16.6, +21.0] |
| family faceswap-stills | 272 | 62.9 % | 65.8 % | +2.9 | [+0.0, +6.3] |
| family flux1 | 85 | 61.2 % | 75.3 % | +14.1 | [+7.1, +22.6] |
| family flux2 | 70 | 27.1 % | 44.3 % | +17.1 | [+7.2, +27.1] |
| family gan-faces | 153 | 53.6 % | 58.2 % | +4.6 | [+0.0, +9.2] |
| family gpt-4o-image-1 | 41 | 7.3 % | 53.7 % | +46.3 | [+31.7, +61.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 29.8 % | +15.8 | [+5.3, +26.3] |
| family hidream | 50 | 80.0 % | 92.0 % | +12.0 | [+2.2, +23.1] |
| family hunyuanimage | 24 | 100.0 % | 95.8 % | -4.2 | [-12.5, +0.0] |
| family ideogram | 47 | 74.5 % | 83.0 % | +8.5 | [+0.0, +19.6] |
| family imagen3 | 53 | 41.5 % | 75.5 % | +34.0 | [+20.8, +47.2] |
| family imagen4 | 106 | 50.9 % | 80.2 % | +29.2 | [+18.2, +39.4] |
| family legacy | 145 | 96.6 % | 92.4 % | -4.1 | [-8.9, +0.7] |
| family long-tail | 186 | 63.4 % | 86.6 % | +23.1 | [+16.7, +30.3] |
| family midjourney | 153 | 81.7 % | 96.1 % | +14.4 | [+8.7, +20.3] |
| family modern-t2i-faces | 115 | 54.8 % | 89.6 % | +34.8 | [+25.2, +44.0] |
| family nano-banana | 118 | 32.2 % | 61.9 % | +29.7 | [+21.7, +38.3] |
| family partial-edits | 161 | 63.4 % | 70.2 % | +6.8 | [+2.5, +11.8] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [-3.5, +3.5] |
| family qwen-image | 106 | 29.2 % | 68.9 % | +39.6 | [+30.8, +48.5] |
| family recraft | 54 | 87.0 % | 94.4 % | +7.4 | [+0.0, +16.7] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [-3.8, +3.9] |
| family seedream3 | 35 | 74.3 % | 85.7 % | +11.4 | [+2.9, +22.2] |
| family seedream4 | 80 | 75.0 % | 82.5 % | +7.5 | [+2.5, +13.9] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

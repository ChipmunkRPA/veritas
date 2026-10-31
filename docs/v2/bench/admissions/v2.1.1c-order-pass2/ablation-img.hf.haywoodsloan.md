# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.haywoodsloan GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +24.1 pts, CI [+20.3, +27.8] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family seedream4, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.7 pts, CI [-2.7, +1.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 95.4 % | +13.7 |
| F2 | 70 | 30.0 % | 48.6 % | +18.6 |
| F3 | 57 | 15.8 % | 40.4 % | +24.6 |
| F4 | 118 | 44.9 % | 69.5 % | +24.6 |
| F5 | 80 | 61.3 % | 77.5 % | +16.2 |
| F6 | 106 | 27.4 % | 68.9 % | +41.5 |
| F7 | 106 | 60.4 % | 89.6 % | +29.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 70.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 6.4 % | +3.5 | [-1.9, +7.9] |
| R2 | 612 | 8.8 % | 3.1 % | -5.7 | [-10.3, +0.0] |
| R3 | 589 | 4.9 % | 4.1 % | -0.8 | [-2.8, +0.9] |
| R4 | 493 | 8.7 % | 2.8 % | -5.9 | [-8.5, -3.2] |
| R5 | 548 | 1.5 % | 8.0 % | +6.6 | [+4.4, +9.1] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-16.7, +7.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 68.3 % | +1.9 | [-3.1, +6.8] |
| threat faceswap | 272 | 55.5 % | 42.3 % | -13.2 | [-19.0, -7.5] |
| threat gan_face | 153 | 25.5 % | 27.5 % | +2.0 | [-3.9, +7.8] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [-3.5, +3.5] |
| threat t2i | 1600 | 63.8 % | 80.2 % | +16.4 | [+14.4, +18.6] |
| family faceswap-stills | 272 | 55.5 % | 42.3 % | -13.2 | [-19.0, -7.5] |
| family flux1 | 85 | 61.2 % | 74.1 % | +12.9 | [+5.9, +20.9] |
| family flux2 | 70 | 30.0 % | 48.6 % | +18.6 | [+7.2, +30.1] |
| family gan-faces | 153 | 25.5 % | 27.5 % | +2.0 | [-3.9, +7.8] |
| family gpt-4o-image-1 | 41 | 34.1 % | 70.7 % | +36.6 | [+22.0, +51.2] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 40.4 % | +24.6 | [+14.0, +35.1] |
| family hidream | 50 | 82.0 % | 92.0 % | +10.0 | [+0.0, +21.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 85.1 % | +8.5 | [+0.0, +18.8] |
| family imagen3 | 53 | 52.8 % | 73.6 % | +20.8 | [+5.7, +35.8] |
| family imagen4 | 106 | 59.4 % | 88.7 % | +29.2 | [+20.4, +38.5] |
| family legacy | 145 | 89.0 % | 90.3 % | +1.4 | [-2.2, +5.3] |
| family long-tail | 186 | 65.6 % | 81.7 % | +16.1 | [+10.0, +22.9] |
| family midjourney | 153 | 81.7 % | 95.4 % | +13.7 | [+8.2, +19.5] |
| family modern-t2i-faces | 115 | 66.1 % | 89.6 % | +23.5 | [+14.9, +32.8] |
| family nano-banana | 118 | 44.9 % | 65.3 % | +20.3 | [+11.8, +29.2] |
| family partial-edits | 161 | 66.5 % | 68.3 % | +1.9 | [-3.1, +6.8] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [-3.5, +3.5] |
| family qwen-image | 106 | 28.3 % | 66.0 % | +37.7 | [+27.6, +47.1] |
| family recraft | 54 | 87.0 % | 92.6 % | +5.6 | [-1.9, +14.8] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [-3.8, +3.9] |
| family seedream3 | 35 | 82.9 % | 85.7 % | +2.9 | [-5.9, +14.3] |
| family seedream4 | 80 | 61.3 % | 75.0 % | +13.7 | [+4.9, +24.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

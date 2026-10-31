# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.7 pts, CI [+12.0, +17.6] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family hidream, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat faceswap, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.4 pts, CI [-1.4, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 67.3 % | 81.0 % | +13.7 |
| F2 | 70 | 15.7 % | 25.7 % | +10.0 |
| F3 | 57 | 7.0 % | 14.0 % | +7.0 |
| F4 | 118 | 18.6 % | 33.1 % | +14.4 |
| F5 | 80 | 62.5 % | 75.0 % | +12.5 |
| F6 | 106 | 17.0 % | 29.2 % | +12.3 |
| F7 | 106 | 17.9 % | 50.9 % | +33.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 29.4 %, with 44.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.8 % | 1.4 % | -2.4 | [-4.5, +0.0] |
| R2 | 612 | 4.7 % | 4.7 % | +0.0 | [-2.5, +1.3] |
| R3 | 589 | 4.6 % | 3.7 % | -0.8 | [-2.4, +0.7] |
| R4 | 493 | 14.6 % | 15.0 % | +0.4 | [-3.2, +3.9] |
| R5 | 548 | 1.1 % | 1.5 % | +0.4 | [-0.4, +1.1] |
| R6 | 28 | 7.1 % | 3.6 % | -3.6 | [-16.7, +8.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 55.9 % | 63.4 % | +7.5 | [+3.1, +12.4] |
| threat faceswap | 272 | 48.2 % | 62.9 % | +14.7 | [+5.5, +26.6] |
| threat gan_face | 153 | 49.7 % | 53.6 % | +3.9 | [+0.7, +7.8] |
| threat personalised | 115 | 93.0 % | 96.5 % | +3.5 | [-0.9, +7.8] |
| threat t2i | 1600 | 42.2 % | 61.1 % | +18.9 | [+16.8, +20.9] |
| family faceswap-stills | 272 | 48.2 % | 62.9 % | +14.7 | [+5.5, +26.6] |
| family flux1 | 85 | 35.3 % | 61.2 % | +25.9 | [+15.3, +36.0] |
| family flux2 | 70 | 15.7 % | 27.1 % | +11.4 | [+4.4, +19.5] |
| family gan-faces | 153 | 49.7 % | 53.6 % | +3.9 | [+0.7, +7.8] |
| family gpt-4o-image-1 | 41 | 0.0 % | 7.3 % | +7.3 | [+0.0, +17.1] |
| family gpt-image-1.5-2 | 57 | 7.0 % | 14.0 % | +7.0 | [+0.0, +15.8] |
| family hidream | 50 | 54.0 % | 80.0 % | +26.0 | [+14.3, +38.5] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 48.9 % | 74.5 % | +25.5 | [+13.3, +38.6] |
| family imagen3 | 53 | 20.8 % | 41.5 % | +20.8 | [+7.5, +34.0] |
| family imagen4 | 106 | 17.0 % | 50.9 % | +34.0 | [+25.0, +43.9] |
| family legacy | 145 | 82.1 % | 96.6 % | +14.5 | [+8.5, +20.9] |
| family long-tail | 186 | 46.8 % | 63.4 % | +16.7 | [+10.8, +22.5] |
| family midjourney | 153 | 66.7 % | 81.7 % | +15.0 | [+9.4, +21.1] |
| family modern-t2i-faces | 115 | 32.2 % | 54.8 % | +22.6 | [+15.0, +30.5] |
| family nano-banana | 118 | 19.5 % | 32.2 % | +12.7 | [+6.7, +19.5] |
| family partial-edits | 161 | 55.9 % | 63.4 % | +7.5 | [+3.1, +12.4] |
| family personalised | 115 | 93.0 % | 96.5 % | +3.5 | [-0.9, +7.8] |
| family qwen-image | 106 | 17.0 % | 29.2 % | +12.3 | [+6.5, +18.7] |
| family recraft | 54 | 50.0 % | 87.0 % | +37.0 | [+24.1, +51.9] |
| family sd3 | 75 | 68.0 % | 97.3 % | +29.3 | [+19.0, +40.0] |
| family seedream3 | 35 | 42.9 % | 74.3 % | +31.4 | [+16.7, +47.1] |
| family seedream4 | 80 | 62.5 % | 75.0 % | +12.5 | [+3.8, +22.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

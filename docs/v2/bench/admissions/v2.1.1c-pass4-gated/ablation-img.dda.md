# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.dda GATES='img.dda=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.5 pts, CI [+9.5, +19.5] |
| A3b | pass | family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.0 pts, CI [-1.5, +1.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1, R4; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 74.5 % | +37.3 |
| F2 | 70 | 42.9 % | 52.9 % | +10.0 |
| F3 | 57 | 40.4 % | 21.1 % | -19.3 |
| F4 | 118 | 49.2 % | 51.7 % | +2.5 |
| F5 | 80 | 41.2 % | 50.0 % | +8.8 |
| F6 | 106 | 34.9 % | 55.7 % | +20.8 |
| F7 | 106 | 32.1 % | 73.6 % | +41.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.7 %, with 54.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.2 % | 8.3 % | +3.1 | [-3.0, +8.4] |
| R2 | 612 | 4.9 % | 2.9 % | -2.0 | [-6.5, +0.4] |
| R3 | 589 | 11.9 % | 9.5 % | -2.4 | [-4.9, +0.2] |
| R4 | 493 | 3.0 % | 5.3 % | +2.2 | [+0.4, +4.3] |
| R5 | 548 | 1.3 % | 1.5 % | +0.2 | [-1.1, +1.6] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [-10.0, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 19.9 % | 47.2 % | +27.3 | [+18.6, +36.0] |
| threat faceswap | 272 | 0.7 % | 1.5 % | +0.7 | [-0.7, +2.4] |
| threat gan_face | 153 | 0.0 % | 3.9 % | +3.9 | [+1.3, +7.2] |
| threat personalised | 115 | 13.9 % | 27.0 % | +13.0 | [+7.0, +20.0] |
| threat t2i | 1600 | 41.4 % | 63.6 % | +22.2 | [+18.6, +25.7] |
| family faceswap-stills | 272 | 0.7 % | 1.5 % | +0.7 | [-0.7, +2.4] |
| family flux1 | 85 | 64.7 % | 68.2 % | +3.5 | [-9.5, +15.9] |
| family flux2 | 70 | 45.7 % | 48.6 % | +2.9 | [-9.3, +14.5] |
| family gan-faces | 153 | 0.0 % | 3.9 % | +3.9 | [+1.3, +7.2] |
| family gpt-4o-image-1 | 41 | 41.5 % | 39.0 % | -2.4 | [-19.5, +14.6] |
| family gpt-image-1.5-2 | 57 | 43.9 % | 26.3 % | -17.5 | [-33.3, -1.8] |
| family hidream | 50 | 68.0 % | 88.0 % | +20.0 | [+5.9, +35.3] |
| family hunyuanimage | 24 | 66.7 % | 87.5 % | +20.8 | [+8.3, +37.5] |
| family ideogram | 47 | 38.3 % | 59.6 % | +21.3 | [+8.3, +35.4] |
| family imagen3 | 53 | 32.1 % | 49.1 % | +17.0 | [+1.9, +30.2] |
| family imagen4 | 106 | 40.6 % | 60.4 % | +19.8 | [+10.5, +29.5] |
| family legacy | 145 | 33.8 % | 88.3 % | +54.5 | [+45.4, +62.9] |
| family long-tail | 186 | 41.4 % | 65.1 % | +23.7 | [+12.2, +34.8] |
| family midjourney | 153 | 37.3 % | 73.2 % | +35.9 | [+28.3, +43.8] |
| family modern-t2i-faces | 115 | 28.7 % | 60.9 % | +32.2 | [+13.7, +54.6] |
| family nano-banana | 118 | 50.8 % | 50.0 % | -0.8 | [-9.3, +6.8] |
| family partial-edits | 161 | 19.9 % | 47.2 % | +27.3 | [+18.6, +36.0] |
| family personalised | 115 | 13.9 % | 27.0 % | +13.0 | [+7.0, +20.0] |
| family qwen-image | 106 | 43.4 % | 42.5 % | -0.9 | [-9.9, +8.5] |
| family recraft | 54 | 29.6 % | 70.4 % | +40.7 | [+27.8, +53.8] |
| family sd3 | 75 | 32.0 % | 89.3 % | +57.3 | [+44.6, +70.4] |
| family seedream3 | 35 | 31.4 % | 71.4 % | +40.0 | [+22.2, +58.8] |
| family seedream4 | 80 | 40.0 % | 58.8 % | +18.8 | [+6.2, +30.9] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

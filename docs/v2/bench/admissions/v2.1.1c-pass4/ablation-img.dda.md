# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.dda GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +13.4 pts, CI [+8.9, +17.8] |
| A3b | pass | family faceswap-stills, family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat gan_face, threat personalised, threat t2i |
| A4 | fail | pooled reals: Δ FPR +1.4 pts, CI [+0.3, +2.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R4; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 85.6 % | +48.4 |
| F2 | 70 | 42.9 % | 40.0 % | -2.9 |
| F3 | 57 | 40.4 % | 17.5 % | -22.8 |
| F4 | 118 | 49.2 % | 53.4 % | +4.2 |
| F5 | 80 | 40.0 % | 62.5 % | +22.5 |
| F6 | 106 | 34.9 % | 39.6 % | +4.7 |
| F7 | 106 | 32.1 % | 71.7 % | +39.6 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 52.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 3.5 % | -0.9 | [-3.3, +0.5] |
| R2 | 612 | 4.2 % | 3.4 % | -0.8 | [-5.1, +1.6] |
| R3 | 589 | 10.0 % | 6.5 % | -3.6 | [-6.0, -1.0] |
| R4 | 493 | 4.9 % | 6.9 % | +2.0 | [-0.4, +4.3] |
| R5 | 548 | 1.5 % | 11.3 % | +9.9 | [+7.3, +12.6] |
| R6 | 28 | 7.1 % | 17.9 % | +10.7 | [+0.0, +25.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.8 % | 67.1 % | +50.3 | [+41.0, +60.2] |
| threat faceswap | 272 | 0.4 % | 23.9 % | +23.5 | [+13.4, +36.6] |
| threat gan_face | 153 | 0.0 % | 28.8 % | +28.8 | [+22.2, +35.9] |
| threat personalised | 115 | 9.6 % | 96.5 % | +87.0 | [+80.9, +93.0] |
| threat t2i | 1600 | 36.4 % | 66.6 % | +30.2 | [+26.6, +33.6] |
| family faceswap-stills | 272 | 0.4 % | 23.9 % | +23.5 | [+13.4, +36.6] |
| family flux1 | 85 | 58.8 % | 63.5 % | +4.7 | [-9.6, +17.1] |
| family flux2 | 70 | 42.9 % | 40.0 % | -2.9 | [-14.9, +9.9] |
| family gan-faces | 153 | 0.0 % | 28.8 % | +28.8 | [+22.2, +35.9] |
| family gpt-4o-image-1 | 41 | 41.5 % | 41.5 % | +0.0 | [-19.5, +19.5] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 15.8 % | -24.6 | [-40.4, -7.0] |
| family hidream | 50 | 62.0 % | 88.0 % | +26.0 | [+8.0, +44.2] |
| family hunyuanimage | 24 | 62.5 % | 100.0 % | +37.5 | [+16.7, +58.3] |
| family ideogram | 47 | 27.7 % | 74.5 % | +46.8 | [+29.5, +62.5] |
| family imagen3 | 53 | 18.9 % | 52.8 % | +34.0 | [+18.9, +49.1] |
| family imagen4 | 106 | 31.1 % | 63.2 % | +32.1 | [+20.6, +43.0] |
| family legacy | 145 | 39.3 % | 91.7 % | +52.4 | [+44.8, +60.0] |
| family long-tail | 186 | 40.3 % | 68.8 % | +28.5 | [+17.3, +39.1] |
| family midjourney | 153 | 35.9 % | 82.4 % | +46.4 | [+37.8, +55.3] |
| family modern-t2i-faces | 115 | 2.6 % | 69.6 % | +67.0 | [+57.4, +75.9] |
| family nano-banana | 118 | 48.3 % | 50.0 % | +1.7 | [-7.5, +11.0] |
| family partial-edits | 161 | 16.8 % | 67.1 % | +50.3 | [+41.0, +60.2] |
| family personalised | 115 | 9.6 % | 96.5 % | +87.0 | [+80.9, +93.0] |
| family qwen-image | 106 | 33.0 % | 35.8 % | +2.8 | [-6.7, +12.8] |
| family recraft | 54 | 27.8 % | 85.2 % | +57.4 | [+42.9, +73.1] |
| family sd3 | 75 | 29.3 % | 98.7 % | +69.3 | [+58.6, +79.5] |
| family seedream3 | 35 | 28.6 % | 77.1 % | +48.6 | [+30.6, +67.6] |
| family seedream4 | 80 | 38.8 % | 60.0 % | +21.2 | [+7.6, +33.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

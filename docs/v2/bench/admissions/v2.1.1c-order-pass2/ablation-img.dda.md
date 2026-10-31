# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -24.4 pts, CI [-29.2, -19.5] |
| A3b | fail | family flux1, family flux2, family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family partial-edits, family personalised, family qwen-image, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -11.1 pts, CI [-14.9, -7.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 76.5 % | 81.7 % | +5.2 |
| F2 | 70 | 64.3 % | 30.0 % | -34.3 |
| F3 | 57 | 73.7 % | 15.8 % | -57.9 |
| F4 | 118 | 80.5 % | 44.9 % | -35.6 |
| F5 | 80 | 77.5 % | 61.3 % | -16.2 |
| F6 | 106 | 62.3 % | 27.4 % | -34.9 |
| F7 | 106 | 57.5 % | 60.4 % | +2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 70.3 %, with 45.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 14.6 % | 2.8 % | -11.8 | [-18.5, -6.1] |
| R2 | 612 | 23.9 % | 8.8 % | -15.0 | [-32.1, -2.6] |
| R3 | 589 | 22.4 % | 4.9 % | -17.5 | [-22.1, -12.9] |
| R4 | 493 | 17.6 % | 8.7 % | -8.9 | [-12.6, -5.3] |
| R5 | 548 | 3.6 % | 1.5 % | -2.2 | [-3.8, -0.5] |
| R6 | 28 | 3.6 % | 14.3 % | +10.7 | [-3.6, +28.6] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 19.9 % | 66.5 % | +46.6 | [+37.3, +55.3] |
| threat faceswap | 272 | 54.4 % | 55.5 % | +1.1 | [-3.4, +5.9] |
| threat gan_face | 153 | 11.1 % | 25.5 % | +14.4 | [+7.8, +20.9] |
| threat personalised | 115 | 7.0 % | 95.7 % | +88.7 | [+82.6, +93.9] |
| threat t2i | 1600 | 17.1 % | 63.8 % | +46.8 | [+43.3, +50.1] |
| family faceswap-stills | 272 | 54.4 % | 55.5 % | +1.1 | [-3.4, +5.9] |
| family flux1 | 85 | 24.7 % | 61.2 % | +36.5 | [+23.8, +50.0] |
| family flux2 | 70 | 12.9 % | 30.0 % | +17.1 | [+4.2, +30.0] |
| family gan-faces | 153 | 11.1 % | 25.5 % | +14.4 | [+7.8, +20.9] |
| family gpt-4o-image-1 | 41 | 14.6 % | 34.1 % | +19.5 | [+0.0, +36.6] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 15.8 % | +1.8 | [-10.5, +14.0] |
| family hidream | 50 | 24.0 % | 82.0 % | +58.0 | [+36.7, +77.8] |
| family hunyuanimage | 24 | 12.5 % | 100.0 % | +87.5 | [+75.0, +100.0] |
| family ideogram | 47 | 12.8 % | 76.6 % | +63.8 | [+46.9, +79.6] |
| family imagen3 | 53 | 18.9 % | 52.8 % | +34.0 | [+17.0, +50.9] |
| family imagen4 | 106 | 20.8 % | 59.4 % | +38.7 | [+26.2, +51.4] |
| family legacy | 145 | 26.9 % | 89.0 % | +62.1 | [+52.4, +71.4] |
| family long-tail | 186 | 15.6 % | 65.6 % | +50.0 | [+40.7, +58.8] |
| family midjourney | 153 | 14.4 % | 81.7 % | +67.3 | [+58.9, +75.0] |
| family modern-t2i-faces | 115 | 6.1 % | 66.1 % | +60.0 | [+50.5, +69.0] |
| family nano-banana | 118 | 17.8 % | 44.9 % | +27.1 | [+16.1, +38.5] |
| family partial-edits | 161 | 19.9 % | 66.5 % | +46.6 | [+37.3, +55.3] |
| family personalised | 115 | 7.0 % | 95.7 % | +88.7 | [+82.6, +93.9] |
| family qwen-image | 106 | 14.2 % | 28.3 % | +14.2 | [+2.8, +25.2] |
| family recraft | 54 | 18.5 % | 87.0 % | +68.5 | [+51.9, +83.9] |
| family sd3 | 75 | 21.3 % | 97.3 % | +76.0 | [+65.8, +85.3] |
| family seedream3 | 35 | 14.3 % | 82.9 % | +68.6 | [+52.8, +82.4] |
| family seedream4 | 80 | 15.0 % | 61.3 % | +46.2 | [+32.1, +59.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

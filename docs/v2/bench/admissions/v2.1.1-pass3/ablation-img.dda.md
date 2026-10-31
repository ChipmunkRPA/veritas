# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +13.9 pts, CI [+9.5, +18.2] |
| A3b | pass | family faceswap-stills, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat faceswap, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -2.0 pts, CI [-3.0, -1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 32.7 % | 85.0 % | +52.3 |
| F2 | 70 | 48.6 % | 42.9 % | -5.7 |
| F3 | 57 | 42.1 % | 19.3 % | -22.8 |
| F4 | 118 | 47.5 % | 51.7 % | +4.2 |
| F5 | 80 | 37.5 % | 63.7 % | +26.2 |
| F6 | 106 | 31.1 % | 40.6 % | +9.4 |
| F7 | 106 | 36.8 % | 70.8 % | +34.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 53.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.6 % | 2.4 % | -4.2 | [-8.0, -1.1] |
| R2 | 612 | 3.1 % | 2.5 % | -0.7 | [-3.5, +1.0] |
| R3 | 589 | 10.7 % | 4.9 % | -5.8 | [-8.4, -3.2] |
| R4 | 493 | 4.5 % | 4.5 % | +0.0 | [-2.0, +2.0] |
| R5 | 548 | 1.3 % | 1.3 % | +0.0 | [-1.1, +1.3] |
| R6 | 28 | 3.6 % | 14.3 % | +10.7 | [+0.0, +25.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 18.0 % | 64.6 % | +46.6 | [+36.6, +57.1] |
| threat faceswap | 272 | 11.4 % | 29.8 % | +18.4 | [+13.7, +23.6] |
| threat gan_face | 153 | 0.7 % | 9.8 % | +9.2 | [+4.6, +14.4] |
| threat personalised | 115 | 11.3 % | 91.3 % | +80.0 | [+72.2, +87.0] |
| threat t2i | 1600 | 38.1 % | 63.7 % | +25.7 | [+22.2, +29.0] |
| family faceswap-stills | 272 | 11.4 % | 29.8 % | +18.4 | [+13.7, +23.6] |
| family flux1 | 85 | 56.5 % | 62.4 % | +5.9 | [-9.2, +18.8] |
| family flux2 | 70 | 48.6 % | 38.6 % | -10.0 | [-21.6, +1.6] |
| family gan-faces | 153 | 0.7 % | 9.8 % | +9.2 | [+4.6, +14.4] |
| family gpt-4o-image-1 | 41 | 43.9 % | 31.7 % | -12.2 | [-29.3, +4.9] |
| family gpt-image-1.5-2 | 57 | 38.6 % | 15.8 % | -22.8 | [-38.6, -7.0] |
| family hidream | 50 | 70.0 % | 84.0 % | +14.0 | [-3.8, +32.7] |
| family hunyuanimage | 24 | 62.5 % | 100.0 % | +37.5 | [+16.7, +58.3] |
| family ideogram | 47 | 27.7 % | 70.2 % | +42.6 | [+26.7, +58.3] |
| family imagen3 | 53 | 20.8 % | 50.9 % | +30.2 | [+15.1, +45.3] |
| family imagen4 | 106 | 35.8 % | 60.4 % | +24.5 | [+13.3, +35.0] |
| family legacy | 145 | 42.8 % | 91.7 % | +49.0 | [+41.3, +57.0] |
| family long-tail | 186 | 45.2 % | 67.7 % | +22.6 | [+11.2, +33.5] |
| family midjourney | 153 | 32.7 % | 79.7 % | +47.1 | [+38.3, +55.9] |
| family modern-t2i-faces | 115 | 10.4 % | 65.2 % | +54.8 | [+44.4, +65.1] |
| family nano-banana | 118 | 47.5 % | 40.7 % | -6.8 | [-16.1, +2.5] |
| family partial-edits | 161 | 18.0 % | 64.6 % | +46.6 | [+36.6, +57.1] |
| family personalised | 115 | 11.3 % | 91.3 % | +80.0 | [+72.2, +87.0] |
| family qwen-image | 106 | 34.0 % | 33.0 % | -0.9 | [-10.7, +9.4] |
| family recraft | 54 | 27.8 % | 79.6 % | +51.9 | [+37.0, +66.7] |
| family sd3 | 75 | 28.0 % | 98.7 % | +70.7 | [+59.7, +81.1] |
| family seedream3 | 35 | 22.9 % | 77.1 % | +54.3 | [+37.1, +70.6] |
| family seedream4 | 80 | 38.8 % | 56.2 % | +17.5 | [+3.7, +31.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

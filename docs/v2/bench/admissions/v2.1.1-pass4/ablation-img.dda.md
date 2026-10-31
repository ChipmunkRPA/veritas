# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.5 pts, CI [+10.1, +18.9] |
| A3b | pass | family gan-faces, family hidream, family hunyuanimage, family ideogram, family imagen3, family imagen4, family legacy, family long-tail, family midjourney, family modern-t2i-faces, family partial-edits, family personalised, family recraft, family sd3, family seedream3, family seedream4, threat edit, threat gan_face, threat personalised, threat t2i |
| A4 | pass | pooled reals: Δ FPR -1.4 pts, CI [-2.4, -0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R4, R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 85.0 % | +47.7 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 40.4 % | 22.8 % | -17.5 |
| F4 | 118 | 49.2 % | 52.5 % | +3.4 |
| F5 | 80 | 41.2 % | 63.7 % | +22.5 |
| F6 | 106 | 34.9 % | 39.6 % | +4.7 |
| F7 | 106 | 32.1 % | 72.6 % | +40.6 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.7 %, with 54.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 2.4 % | -2.1 | [-3.9, -0.4] |
| R2 | 612 | 4.2 % | 2.8 % | -1.5 | [-5.5, +0.5] |
| R3 | 589 | 10.4 % | 5.1 % | -5.3 | [-8.0, -2.6] |
| R4 | 493 | 2.6 % | 4.7 % | +2.0 | [+0.2, +3.9] |
| R5 | 548 | 0.9 % | 0.5 % | -0.4 | [-1.5, +0.5] |
| R6 | 28 | 7.1 % | 14.3 % | +7.1 | [-7.1, +23.1] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.1 % | 64.6 % | +48.4 | [+38.5, +58.4] |
| threat faceswap | 272 | 0.4 % | 5.9 % | +5.5 | [+2.7, +9.1] |
| threat gan_face | 153 | 0.0 % | 14.4 % | +14.4 | [+9.2, +20.3] |
| threat personalised | 115 | 9.6 % | 91.3 % | +81.7 | [+74.8, +87.8] |
| threat t2i | 1600 | 35.4 % | 65.4 % | +30.0 | [+26.2, +33.5] |
| family faceswap-stills | 272 | 0.4 % | 5.9 % | +5.5 | [+2.7, +9.1] |
| family flux1 | 85 | 58.8 % | 62.4 % | +3.5 | [-11.1, +16.5] |
| family flux2 | 70 | 42.9 % | 38.6 % | -4.3 | [-16.7, +8.7] |
| family gan-faces | 153 | 0.0 % | 14.4 % | +14.4 | [+9.2, +20.3] |
| family gpt-4o-image-1 | 41 | 41.5 % | 31.7 % | -9.8 | [-26.8, +7.3] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 15.8 % | -24.6 | [-40.4, -7.0] |
| family hidream | 50 | 62.0 % | 88.0 % | +26.0 | [+8.0, +44.2] |
| family hunyuanimage | 24 | 62.5 % | 100.0 % | +37.5 | [+16.7, +58.3] |
| family ideogram | 47 | 27.7 % | 70.2 % | +42.6 | [+26.7, +58.3] |
| family imagen3 | 53 | 18.9 % | 52.8 % | +34.0 | [+18.9, +49.1] |
| family imagen4 | 106 | 31.1 % | 65.1 % | +34.0 | [+22.1, +45.1] |
| family legacy | 145 | 33.8 % | 90.3 % | +56.6 | [+48.1, +64.6] |
| family long-tail | 186 | 38.7 % | 67.2 % | +28.5 | [+16.6, +39.2] |
| family midjourney | 153 | 35.3 % | 82.4 % | +47.1 | [+38.5, +56.1] |
| family modern-t2i-faces | 115 | 2.6 % | 73.0 % | +70.4 | [+62.4, +78.3] |
| family nano-banana | 118 | 47.5 % | 44.9 % | -2.5 | [-11.0, +5.9] |
| family partial-edits | 161 | 16.1 % | 64.6 % | +48.4 | [+38.5, +58.4] |
| family personalised | 115 | 9.6 % | 91.3 % | +81.7 | [+74.8, +87.8] |
| family qwen-image | 106 | 33.0 % | 34.9 % | +1.9 | [-8.3, +11.7] |
| family recraft | 54 | 27.8 % | 81.5 % | +53.7 | [+39.3, +69.2] |
| family sd3 | 75 | 28.0 % | 98.7 % | +70.7 | [+59.7, +81.1] |
| family seedream3 | 35 | 28.6 % | 77.1 % | +48.6 | [+30.6, +67.6] |
| family seedream4 | 80 | 37.5 % | 57.5 % | +20.0 | [+7.4, +32.9] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

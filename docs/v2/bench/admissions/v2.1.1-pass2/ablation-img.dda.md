# Ablation: img.dda

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dda ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (MSCOCO train2017)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +9.8 pts, CI [+6.8, +13.1] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family ideogram, family imagen3, family imagen4, family midjourney, family modern-t2i-faces, family nano-banana, family recraft, family sd3, family seedream3, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.6 pts, CI [-1.4, +0.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 71.2 % | 83.7 % | +12.4 |
| F2 | 70 | 30.0 % | 37.1 % | +7.1 |
| F3 | 57 | 14.0 % | 15.8 % | +1.8 |
| F4 | 118 | 36.4 % | 48.3 % | +11.9 |
| F5 | 80 | 67.5 % | 76.2 % | +8.7 |
| F6 | 106 | 38.7 % | 39.6 % | +0.9 |
| F7 | 106 | 40.6 % | 66.0 % | +25.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 42.6 %, with 52.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 6.4 % | 4.2 % | -2.1 | [-3.5, +1.5] |
| R2 | 612 | 5.4 % | 3.8 % | -1.6 | [-4.4, -0.1] |
| R3 | 589 | 5.1 % | 4.6 % | -0.5 | [-1.8, +1.0] |
| R4 | 493 | 4.5 % | 4.5 % | +0.0 | [-1.6, +1.6] |
| R5 | 548 | 1.3 % | 2.0 % | +0.7 | [-0.4, +1.8] |
| R6 | 28 | 7.1 % | 10.7 % | +3.6 | [-7.7, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 60.2 % | 65.2 % | +5.0 | [+0.0, +9.9] |
| threat faceswap | 272 | 18.0 % | 20.6 % | +2.6 | [+0.6, +5.0] |
| threat gan_face | 153 | 43.8 % | 43.8 % | +0.0 | [-2.6, +2.6] |
| threat personalised | 115 | 93.9 % | 96.5 % | +2.6 | [+0.0, +6.1] |
| threat t2i | 1600 | 52.8 % | 67.6 % | +14.8 | [+12.7, +16.8] |
| family faceswap-stills | 272 | 18.0 % | 20.6 % | +2.6 | [+0.6, +5.0] |
| family flux1 | 85 | 50.6 % | 67.1 % | +16.5 | [+7.1, +25.8] |
| family flux2 | 70 | 22.9 % | 37.1 % | +14.3 | [+5.6, +23.5] |
| family gan-faces | 153 | 43.8 % | 43.8 % | +0.0 | [-2.6, +2.6] |
| family gpt-4o-image-1 | 41 | 2.4 % | 22.0 % | +19.5 | [+7.3, +31.7] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 17.5 % | +3.5 | [-5.3, +12.3] |
| family hidream | 50 | 78.0 % | 86.0 % | +8.0 | [-2.0, +19.6] |
| family hunyuanimage | 24 | 95.8 % | 100.0 % | +4.2 | [+0.0, +12.5] |
| family ideogram | 47 | 57.4 % | 74.5 % | +17.0 | [+7.7, +29.2] |
| family imagen3 | 53 | 35.8 % | 52.8 % | +17.0 | [+3.7, +30.2] |
| family imagen4 | 106 | 36.8 % | 65.1 % | +28.3 | [+19.4, +37.5] |
| family legacy | 145 | 92.4 % | 95.2 % | +2.8 | [+0.7, +5.8] |
| family long-tail | 186 | 60.8 % | 69.9 % | +9.1 | [+2.5, +15.5] |
| family midjourney | 153 | 68.0 % | 83.7 % | +15.7 | [+10.1, +21.9] |
| family modern-t2i-faces | 115 | 40.0 % | 72.2 % | +32.2 | [+25.4, +39.4] |
| family nano-banana | 118 | 33.9 % | 44.1 % | +10.2 | [+3.4, +17.7] |
| family partial-edits | 161 | 60.2 % | 65.2 % | +5.0 | [+0.0, +9.9] |
| family personalised | 115 | 93.9 % | 96.5 % | +2.6 | [+0.0, +6.1] |
| family qwen-image | 106 | 34.0 % | 37.7 % | +3.8 | [-2.8, +10.7] |
| family recraft | 54 | 57.4 % | 85.2 % | +27.8 | [+13.5, +42.6] |
| family sd3 | 75 | 72.0 % | 98.7 % | +26.7 | [+16.9, +36.8] |
| family seedream3 | 35 | 54.3 % | 82.9 % | +28.6 | [+14.3, +44.1] |
| family seedream4 | 80 | 66.2 % | 75.0 % | +8.8 | [+0.0, +17.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

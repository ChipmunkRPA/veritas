# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7,face.gend_clip_l14 GATES='img.dfdc_b7=has_face;face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.8 pts, CI [-3.4, -0.2] |
| A3b | fail | family faceswap-stills, family gan-faces, threat faceswap, threat gan_face |
| A4 | pass | pooled reals: Δ FPR -0.2 pts, CI [-1.1, +1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 82.4 % | -2.6 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 19.3 % | 19.3 % | +0.0 |
| F4 | 118 | 51.7 % | 50.0 % | -1.7 |
| F5 | 80 | 63.7 % | 61.3 % | -2.5 |
| F6 | 106 | 40.6 % | 39.6 % | -0.9 |
| F7 | 106 | 70.8 % | 66.0 % | -4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.4 %, with 51.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 5.4 % | +3.1 | [-1.2, +6.8] |
| R2 | 612 | 2.5 % | 1.5 % | -1.0 | [-2.0, -0.2] |
| R3 | 589 | 4.9 % | 3.2 % | -1.7 | [-3.0, -0.5] |
| R4 | 493 | 4.5 % | 3.7 % | -0.8 | [-1.8, +0.0] |
| R5 | 548 | 1.3 % | 1.6 % | +0.4 | [-0.5, +1.1] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 62.1 % | -2.5 | [-5.0, -0.6] |
| threat faceswap | 272 | 29.8 % | 60.3 % | +30.5 | [+20.2, +39.4] |
| threat gan_face | 153 | 9.8 % | 23.5 % | +13.7 | [+8.5, +19.6] |
| threat personalised | 115 | 91.3 % | 87.0 % | -4.3 | [-7.8, -0.9] |
| threat t2i | 1600 | 63.7 % | 60.6 % | -3.2 | [-4.1, -2.3] |
| family faceswap-stills | 272 | 29.8 % | 60.3 % | +30.5 | [+20.2, +39.4] |
| family flux1 | 85 | 62.4 % | 62.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 38.6 % | 30.0 % | -8.6 | [-15.3, -2.9] |
| family gan-faces | 153 | 9.8 % | 23.5 % | +13.7 | [+8.5, +19.6] |
| family gpt-4o-image-1 | 41 | 31.7 % | 24.4 % | -7.3 | [-17.1, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 84.0 % | 80.0 % | -4.0 | [-10.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 68.1 % | -2.1 | [-6.8, +0.0] |
| family imagen3 | 53 | 50.9 % | 45.3 % | -5.7 | [-13.2, +0.0] |
| family imagen4 | 106 | 60.4 % | 58.5 % | -1.9 | [-5.6, +1.9] |
| family legacy | 145 | 91.7 % | 89.7 % | -2.1 | [-4.8, +0.0] |
| family long-tail | 186 | 67.7 % | 65.6 % | -2.2 | [-5.1, +0.6] |
| family midjourney | 153 | 79.7 % | 77.1 % | -2.6 | [-5.4, -0.6] |
| family modern-t2i-faces | 115 | 65.2 % | 59.1 % | -6.1 | [-11.1, -2.3] |
| family nano-banana | 118 | 40.7 % | 37.3 % | -3.4 | [-6.8, -0.8] |
| family partial-edits | 161 | 64.6 % | 62.1 % | -2.5 | [-5.0, -0.6] |
| family personalised | 115 | 91.3 % | 87.0 % | -4.3 | [-7.8, -0.9] |
| family qwen-image | 106 | 33.0 % | 26.4 % | -6.6 | [-12.0, -1.8] |
| family recraft | 54 | 79.6 % | 79.6 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 98.7 % | 97.3 % | -1.3 | [-4.2, +0.0] |
| family seedream3 | 35 | 77.1 % | 74.3 % | -2.9 | [-8.1, +0.0] |
| family seedream4 | 80 | 56.2 % | 52.5 % | -3.7 | [-8.4, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

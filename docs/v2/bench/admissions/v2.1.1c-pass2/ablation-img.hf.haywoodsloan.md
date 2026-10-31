# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq,img.hf.haywoodsloan GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +14.9 pts, CI [+11.3, +18.6] |
| A3b | pass | family flux1, family gpt-4o-image-1, family gpt-image-1.5-2, family imagen4, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family seedream4, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.2 pts, CI [-1.5, +1.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 93.5 % | +10.5 |
| F2 | 70 | 44.3 % | 47.1 % | +2.9 |
| F3 | 57 | 21.1 % | 36.8 % | +15.8 |
| F4 | 118 | 50.8 % | 69.5 % | +18.6 |
| F5 | 80 | 68.8 % | 80.0 % | +11.3 |
| F6 | 106 | 40.6 % | 68.9 % | +28.3 |
| F7 | 106 | 67.9 % | 84.9 % | +17.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.8 %, with 68.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 6.8 % | +2.4 | [-2.7, +6.5] |
| R2 | 612 | 3.9 % | 1.1 % | -2.8 | [-4.2, -0.5] |
| R3 | 589 | 5.8 % | 3.7 % | -2.0 | [-3.9, -0.3] |
| R4 | 493 | 5.9 % | 2.8 % | -3.0 | [-4.9, -1.2] |
| R5 | 548 | 4.2 % | 10.0 % | +5.8 | [+3.1, +8.6] |
| R6 | 28 | 17.9 % | 7.1 % | -10.7 | [-25.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 66.5 % | +1.2 | [-3.7, +6.2] |
| threat faceswap | 272 | 71.3 % | 64.3 % | -7.0 | [-11.3, -2.9] |
| threat gan_face | 153 | 54.2 % | 52.3 % | -2.0 | [-8.5, +4.6] |
| threat personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-7.0, +1.7] |
| threat t2i | 1600 | 65.1 % | 76.8 % | +11.7 | [+9.8, +13.9] |
| family faceswap-stills | 272 | 71.3 % | 64.3 % | -7.0 | [-11.3, -2.9] |
| family flux1 | 85 | 62.4 % | 76.5 % | +14.1 | [+5.8, +24.1] |
| family flux2 | 70 | 40.0 % | 45.7 % | +5.7 | [-1.5, +13.5] |
| family gan-faces | 153 | 54.2 % | 52.3 % | -2.0 | [-8.5, +4.6] |
| family gpt-4o-image-1 | 41 | 36.6 % | 61.0 % | +24.4 | [+9.8, +39.0] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 33.3 % | +15.8 | [+3.5, +28.1] |
| family hidream | 50 | 84.0 % | 90.0 % | +6.0 | [-4.2, +17.6] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 83.0 % | +6.4 | [-4.3, +17.8] |
| family imagen3 | 53 | 49.1 % | 58.5 % | +9.4 | [+0.0, +18.9] |
| family imagen4 | 106 | 62.3 % | 78.3 % | +16.0 | [+8.1, +24.3] |
| family legacy | 145 | 91.0 % | 89.7 % | -1.4 | [-6.5, +3.7] |
| family long-tail | 186 | 71.0 % | 79.0 % | +8.1 | [+2.2, +14.5] |
| family midjourney | 153 | 80.4 % | 92.8 % | +12.4 | [+7.1, +18.1] |
| family modern-t2i-faces | 115 | 66.1 % | 89.6 % | +23.5 | [+16.1, +31.2] |
| family nano-banana | 118 | 42.4 % | 62.7 % | +20.3 | [+10.2, +31.1] |
| family partial-edits | 161 | 65.2 % | 66.5 % | +1.2 | [-3.7, +6.2] |
| family personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-7.0, +1.7] |
| family qwen-image | 106 | 33.0 % | 62.3 % | +29.2 | [+18.8, +39.6] |
| family recraft | 54 | 83.3 % | 88.9 % | +5.6 | [+0.0, +13.0] |
| family sd3 | 75 | 98.7 % | 94.7 % | -4.0 | [-8.8, +0.0] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [-11.4, +11.4] |
| family seedream4 | 80 | 57.5 % | 71.2 % | +13.8 | [+3.7, +24.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

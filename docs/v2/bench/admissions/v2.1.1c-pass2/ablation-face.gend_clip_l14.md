# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +3.0 pts, CI [+1.5, +4.7] |
| A3b | pass | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -1.8 pts, CI [-2.7, -1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 83.0 % | +0.7 |
| F2 | 70 | 41.4 % | 44.3 % | +2.9 |
| F3 | 57 | 15.8 % | 21.1 % | +5.3 |
| F4 | 118 | 47.5 % | 50.8 % | +3.4 |
| F5 | 80 | 63.7 % | 68.8 % | +5.0 |
| F6 | 106 | 37.7 % | 40.6 % | +2.8 |
| F7 | 106 | 67.0 % | 67.9 % | +0.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 50.8 %, with 53.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.3 % | 4.5 % | +1.2 | [-0.5, +2.7] |
| R2 | 612 | 4.1 % | 3.9 % | -0.2 | [-0.3, +0.0] |
| R3 | 589 | 5.4 % | 5.8 % | +0.3 | [-0.5, +1.2] |
| R4 | 493 | 6.7 % | 5.9 % | -0.8 | [-2.0, +0.4] |
| R5 | 548 | 13.5 % | 4.2 % | -9.3 | [-12.0, -6.6] |
| R6 | 28 | 17.9 % | 17.9 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 65.2 % | -0.6 | [-1.9, +0.0] |
| threat faceswap | 272 | 29.0 % | 71.3 % | +42.3 | [+29.6, +54.4] |
| threat gan_face | 153 | 44.4 % | 54.2 % | +9.8 | [+1.3, +18.3] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 66.5 % | 65.1 % | -1.4 | [-2.1, -0.8] |
| family faceswap-stills | 272 | 29.0 % | 71.3 % | +42.3 | [+29.6, +54.4] |
| family flux1 | 85 | 64.7 % | 62.4 % | -2.4 | [-6.1, +0.0] |
| family flux2 | 70 | 42.9 % | 40.0 % | -2.9 | [-7.4, +0.0] |
| family gan-faces | 153 | 44.4 % | 54.2 % | +9.8 | [+1.3, +18.3] |
| family gpt-4o-image-1 | 41 | 39.0 % | 36.6 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 86.0 % | 84.0 % | -2.0 | [-6.4, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 78.7 % | 76.6 % | -2.1 | [-6.7, +0.0] |
| family imagen3 | 53 | 54.7 % | 49.1 % | -5.7 | [-13.2, +0.0] |
| family imagen4 | 106 | 66.0 % | 62.3 % | -3.8 | [-7.8, -0.9] |
| family legacy | 145 | 90.3 % | 91.0 % | +0.7 | [+0.0, +2.2] |
| family long-tail | 186 | 69.9 % | 71.0 % | +1.1 | [+0.0, +2.9] |
| family midjourney | 153 | 81.0 % | 80.4 % | -0.7 | [-2.0, +0.0] |
| family modern-t2i-faces | 115 | 69.6 % | 66.1 % | -3.5 | [-6.8, -0.9] |
| family nano-banana | 118 | 43.2 % | 42.4 % | -0.8 | [-2.6, +0.0] |
| family partial-edits | 161 | 65.8 % | 65.2 % | -0.6 | [-1.9, +0.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 35.8 % | 33.0 % | -2.8 | [-6.4, +0.0] |
| family recraft | 54 | 85.2 % | 83.3 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 57.5 % | -3.8 | [-8.6, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

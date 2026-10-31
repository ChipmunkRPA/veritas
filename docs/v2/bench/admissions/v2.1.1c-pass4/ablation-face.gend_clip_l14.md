# Ablation: face.gend_clip_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,face.gend_clip_l14 GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=face.gend_clip_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (Celeb-DF v3, DeepSpeak v1.1, DeepSpeak v2, FFIW, FaceForensics++)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -4.7 pts, CI [-6.3, -3.1] |
| A3b | fail | family faceswap-stills, family gan-faces, threat faceswap, threat gan_face |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.8, +0.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 35.9 % | 28.8 % | -7.2 |
| F2 | 70 | 42.9 % | 38.6 % | -4.3 |
| F3 | 57 | 40.4 % | 35.1 % | -5.3 |
| F4 | 118 | 49.2 % | 45.8 % | -3.4 |
| F5 | 80 | 40.0 % | 35.0 % | -5.0 |
| F6 | 106 | 34.9 % | 30.2 % | -4.7 |
| F7 | 106 | 32.1 % | 29.2 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.3 %, with 34.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.4 % | 4.0 % | -1.4 | [-3.2, +0.9] |
| R2 | 612 | 3.9 % | 3.1 % | -0.8 | [-2.2, +0.0] |
| R3 | 589 | 10.4 % | 9.2 % | -1.2 | [-2.9, +0.5] |
| R4 | 495 | 3.8 % | 5.7 % | +1.8 | [+0.4, +3.4] |
| R5 | 549 | 1.5 % | 2.9 % | +1.5 | [+0.5, +2.4] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [-10.7, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 15.5 % | 17.4 % | +1.9 | [-1.9, +6.2] |
| threat faceswap | 272 | 0.4 % | 46.0 % | +45.6 | [+34.2, +54.9] |
| threat gan_face | 153 | 0.0 % | 16.3 % | +16.3 | [+10.5, +22.9] |
| threat personalised | 115 | 9.6 % | 12.2 % | +2.6 | [-1.7, +7.0] |
| threat t2i | 1603 | 36.1 % | 33.7 % | -2.4 | [-3.6, -1.3] |
| family faceswap-stills | 272 | 0.4 % | 46.0 % | +45.6 | [+34.2, +54.9] |
| family flux1 | 85 | 61.2 % | 48.2 % | -12.9 | [-20.5, -6.0] |
| family flux2 | 70 | 41.4 % | 41.4 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 0.0 % | 16.3 % | +16.3 | [+10.5, +22.9] |
| family gpt-4o-image-1 | 41 | 36.6 % | 34.1 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 35.1 % | -5.3 | [-12.3, +0.0] |
| family hidream | 50 | 60.0 % | 60.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 62.5 % | 54.2 % | -8.3 | [-20.8, +0.0] |
| family ideogram | 47 | 27.7 % | 25.5 % | -2.1 | [-6.8, +0.0] |
| family imagen3 | 53 | 18.9 % | 15.1 % | -3.8 | [-9.4, +0.0] |
| family imagen4 | 106 | 30.2 % | 31.1 % | +0.9 | [-1.9, +4.0] |
| family legacy | 148 | 37.8 % | 41.9 % | +4.1 | [-1.3, +9.9] |
| family long-tail | 186 | 40.3 % | 41.9 % | +1.6 | [-1.8, +5.1] |
| family midjourney | 153 | 34.6 % | 29.4 % | -5.2 | [-9.2, -1.4] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 47.5 % | 44.1 % | -3.4 | [-6.8, -0.8] |
| family partial-edits | 161 | 15.5 % | 17.4 % | +1.9 | [-1.9, +6.2] |
| family personalised | 115 | 9.6 % | 12.2 % | +2.6 | [-1.7, +7.0] |
| family qwen-image | 106 | 34.0 % | 30.2 % | -3.8 | [-7.8, -0.9] |
| family recraft | 54 | 27.8 % | 25.9 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 30.7 % | 25.3 % | -5.3 | [-11.1, -1.3] |
| family seedream3 | 35 | 28.6 % | 20.0 % | -8.6 | [-17.6, +0.0] |
| family seedream4 | 80 | 40.0 % | 35.0 % | -5.0 | [-10.1, -1.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

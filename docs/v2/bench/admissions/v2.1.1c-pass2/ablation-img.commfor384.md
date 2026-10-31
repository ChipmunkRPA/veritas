# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq,img.commfor384 GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -1.3 pts, CI [-3.9, +1.1] |
| A3b | fail | family gan-faces, family seedream4, threat gan_face |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.8, +0.8] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 84.3 % | +1.3 |
| F2 | 70 | 44.3 % | 38.6 % | -5.7 |
| F3 | 57 | 21.1 % | 17.5 % | -3.5 |
| F4 | 118 | 50.8 % | 44.1 % | -6.8 |
| F5 | 80 | 68.8 % | 78.8 % | +10.0 |
| F6 | 106 | 40.6 % | 41.5 % | +0.9 |
| F7 | 106 | 67.9 % | 62.3 % | -5.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.8 %, with 52.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 5.7 % | +1.2 | [-2.3, +3.4] |
| R2 | 612 | 3.9 % | 3.8 % | -0.2 | [-1.1, +1.4] |
| R3 | 589 | 5.8 % | 5.6 % | -0.2 | [-1.7, +1.4] |
| R4 | 493 | 5.9 % | 4.9 % | -1.0 | [-2.6, +0.6] |
| R5 | 548 | 4.2 % | 4.6 % | +0.4 | [-0.7, +1.6] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.2 % | +0.0 | [-4.3, +4.3] |
| threat faceswap | 272 | 71.3 % | 78.7 % | +7.4 | [+4.0, +12.1] |
| threat gan_face | 153 | 54.2 % | 75.8 % | +21.6 | [+15.0, +27.5] |
| threat personalised | 115 | 96.5 % | 98.3 % | +1.7 | [-2.6, +6.1] |
| threat t2i | 1600 | 65.1 % | 66.2 % | +1.2 | [-0.4, +2.7] |
| family faceswap-stills | 272 | 71.3 % | 78.7 % | +7.4 | [+4.0, +12.1] |
| family flux1 | 85 | 62.4 % | 63.5 % | +1.2 | [-4.8, +7.1] |
| family flux2 | 70 | 40.0 % | 35.7 % | -4.3 | [-11.6, +2.9] |
| family gan-faces | 153 | 54.2 % | 75.8 % | +21.6 | [+15.0, +27.5] |
| family gpt-4o-image-1 | 41 | 36.6 % | 19.5 % | -17.1 | [-29.3, -7.3] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 15.8 % | -1.8 | [-8.8, +5.3] |
| family hidream | 50 | 84.0 % | 86.0 % | +2.0 | [-4.1, +9.3] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 74.5 % | -2.1 | [-13.6, +8.9] |
| family imagen3 | 53 | 49.1 % | 50.9 % | +1.9 | [-9.4, +13.2] |
| family imagen4 | 106 | 62.3 % | 57.5 % | -4.7 | [-12.2, +2.7] |
| family legacy | 145 | 91.0 % | 95.2 % | +4.1 | [+1.3, +7.6] |
| family long-tail | 186 | 71.0 % | 71.0 % | +0.0 | [-3.4, +3.3] |
| family midjourney | 153 | 80.4 % | 83.7 % | +3.3 | [+0.0, +7.3] |
| family modern-t2i-faces | 115 | 66.1 % | 64.3 % | -1.7 | [-8.2, +4.3] |
| family nano-banana | 118 | 42.4 % | 40.7 % | -1.7 | [-8.4, +4.3] |
| family partial-edits | 161 | 65.2 % | 65.2 % | +0.0 | [-4.3, +4.3] |
| family personalised | 115 | 96.5 % | 98.3 % | +1.7 | [-2.6, +6.1] |
| family qwen-image | 106 | 33.0 % | 37.7 % | +4.7 | [+0.0, +9.8] |
| family recraft | 54 | 83.3 % | 88.9 % | +5.6 | [+0.0, +13.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 82.9 % | +2.9 | [-5.9, +14.3] |
| family seedream4 | 80 | 57.5 % | 78.8 % | +21.3 | [+12.5, +31.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

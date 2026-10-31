# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7,img.commfor384 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.7 pts, CI [-0.6, +3.9] |
| A3b | pass | family faceswap-stills, family gan-faces, family seedream4, threat faceswap, threat gan_face |
| A4 | fail | pooled reals: Δ FPR +0.6 pts, CI [+0.0, +1.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 85.0 % | +0.0 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 19.3 % | 17.5 % | -1.8 |
| F4 | 118 | 51.7 % | 49.2 % | -2.5 |
| F5 | 80 | 63.7 % | 78.8 % | +15.0 |
| F6 | 106 | 40.6 % | 44.3 % | +3.8 |
| F7 | 106 | 70.8 % | 67.9 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.4 %, with 55.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 3.5 % | +1.2 | [-0.9, +2.4] |
| R2 | 612 | 2.5 % | 3.8 % | +1.3 | [+0.0, +2.3] |
| R3 | 589 | 4.9 % | 4.9 % | +0.0 | [-1.4, +1.4] |
| R4 | 493 | 4.5 % | 5.5 % | +1.0 | [-0.4, +2.4] |
| R5 | 548 | 1.3 % | 1.3 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.8 % | +1.2 | [-1.9, +4.4] |
| threat faceswap | 272 | 29.8 % | 51.5 % | +21.7 | [+15.0, +29.0] |
| threat gan_face | 153 | 9.8 % | 43.1 % | +33.3 | [+26.1, +40.5] |
| threat personalised | 115 | 91.3 % | 93.9 % | +2.6 | [-0.9, +7.0] |
| threat t2i | 1600 | 63.7 % | 66.9 % | +3.1 | [+1.7, +4.5] |
| family faceswap-stills | 272 | 29.8 % | 51.5 % | +21.7 | [+15.0, +29.0] |
| family flux1 | 85 | 62.4 % | 67.1 % | +4.7 | [+0.0, +10.3] |
| family flux2 | 70 | 38.6 % | 35.7 % | -2.9 | [-10.0, +4.2] |
| family gan-faces | 153 | 9.8 % | 43.1 % | +33.3 | [+26.1, +40.5] |
| family gpt-4o-image-1 | 41 | 31.7 % | 22.0 % | -9.8 | [-19.5, -2.4] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [-3.5, +7.0] |
| family hidream | 50 | 84.0 % | 86.0 % | +2.0 | [-4.3, +8.5] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 74.5 % | +4.3 | [-4.3, +12.8] |
| family imagen3 | 53 | 50.9 % | 58.5 % | +7.5 | [-1.9, +17.0] |
| family imagen4 | 106 | 60.4 % | 63.2 % | +2.8 | [-2.9, +8.3] |
| family legacy | 145 | 91.7 % | 95.2 % | +3.4 | [+0.7, +6.7] |
| family long-tail | 186 | 67.7 % | 69.4 % | +1.6 | [-2.3, +5.8] |
| family midjourney | 153 | 79.7 % | 83.7 % | +3.9 | [+0.6, +7.6] |
| family modern-t2i-faces | 115 | 65.2 % | 65.2 % | +0.0 | [-5.0, +5.4] |
| family nano-banana | 118 | 40.7 % | 41.5 % | +0.8 | [-4.2, +5.9] |
| family partial-edits | 161 | 64.6 % | 65.8 % | +1.2 | [-1.9, +4.4] |
| family personalised | 115 | 91.3 % | 93.9 % | +2.6 | [-0.9, +7.0] |
| family qwen-image | 106 | 33.0 % | 38.7 % | +5.7 | [+0.0, +11.7] |
| family recraft | 54 | 79.6 % | 85.2 % | +5.6 | [+0.0, +13.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 82.9 % | +5.7 | [+0.0, +14.7] |
| family seedream4 | 80 | 56.2 % | 75.0 % | +18.8 | [+10.1, +28.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

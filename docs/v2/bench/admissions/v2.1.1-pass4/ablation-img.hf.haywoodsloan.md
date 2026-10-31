# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.hf.haywoodsloan CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +18.8 pts, CI [+15.4, +22.3] |
| A3b | pass | family flux1, family gan-faces, family gpt-4o-image-1, family gpt-image-1.5-2, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family seedream4, threat gan_face, threat t2i |
| A4 | fail | pooled reals: Δ FPR +1.3 pts, CI [+0.1, +2.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 96.7 % | +11.8 |
| F2 | 70 | 42.9 % | 48.6 % | +5.7 |
| F3 | 57 | 22.8 % | 40.4 % | +17.5 |
| F4 | 118 | 52.5 % | 77.1 % | +24.6 |
| F5 | 80 | 63.7 % | 81.2 % | +17.5 |
| F6 | 106 | 39.6 % | 74.5 % | +34.9 |
| F7 | 106 | 72.6 % | 92.5 % | +19.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.2 %, with 73.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 5.0 % | +2.6 | [-2.6, +6.6] |
| R2 | 612 | 2.8 % | 2.0 % | -0.8 | [-2.2, +1.3] |
| R3 | 589 | 5.1 % | 4.8 % | -0.3 | [-2.2, +1.6] |
| R4 | 493 | 4.7 % | 2.8 % | -1.8 | [-3.7, -0.2] |
| R5 | 548 | 0.5 % | 8.0 % | +7.5 | [+5.3, +9.9] |
| R6 | 28 | 14.3 % | 7.1 % | -7.1 | [-19.2, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 69.6 % | +5.0 | [+0.0, +9.9] |
| threat faceswap | 272 | 5.9 % | 9.9 % | +4.0 | [+1.8, +7.1] |
| threat gan_face | 153 | 14.4 % | 25.5 % | +11.1 | [+5.9, +17.0] |
| threat personalised | 115 | 91.3 % | 93.0 % | +1.7 | [-1.7, +5.2] |
| threat t2i | 1600 | 65.4 % | 81.4 % | +15.9 | [+13.9, +18.1] |
| family faceswap-stills | 272 | 5.9 % | 9.9 % | +4.0 | [+1.8, +7.1] |
| family flux1 | 85 | 62.4 % | 83.5 % | +21.2 | [+12.8, +31.0] |
| family flux2 | 70 | 38.6 % | 47.1 % | +8.6 | [+1.3, +17.3] |
| family gan-faces | 153 | 14.4 % | 25.5 % | +11.1 | [+5.9, +17.0] |
| family gpt-4o-image-1 | 41 | 31.7 % | 70.7 % | +39.0 | [+24.4, +53.7] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 38.6 % | +22.8 | [+12.3, +33.3] |
| family hidream | 50 | 88.0 % | 92.0 % | +4.0 | [-4.1, +14.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 87.2 % | +17.0 | [+6.5, +28.3] |
| family imagen3 | 53 | 52.8 % | 66.0 % | +13.2 | [+5.7, +22.6] |
| family imagen4 | 106 | 65.1 % | 87.7 % | +22.6 | [+13.9, +31.7] |
| family legacy | 145 | 90.3 % | 90.3 % | +0.0 | [-4.3, +4.3] |
| family long-tail | 186 | 67.2 % | 81.7 % | +14.5 | [+8.2, +21.5] |
| family midjourney | 153 | 82.4 % | 96.7 % | +14.4 | [+9.0, +20.0] |
| family modern-t2i-faces | 115 | 73.0 % | 95.7 % | +22.6 | [+14.9, +30.9] |
| family nano-banana | 118 | 44.9 % | 68.6 % | +23.7 | [+14.3, +33.3] |
| family partial-edits | 161 | 64.6 % | 69.6 % | +5.0 | [+0.0, +9.9] |
| family personalised | 115 | 91.3 % | 93.0 % | +1.7 | [-1.7, +5.2] |
| family qwen-image | 106 | 34.9 % | 69.8 % | +34.9 | [+23.6, +45.3] |
| family recraft | 54 | 81.5 % | 90.7 % | +9.3 | [+0.0, +19.2] |
| family sd3 | 75 | 98.7 % | 97.3 % | -1.3 | [-4.2, +0.0] |
| family seedream3 | 35 | 77.1 % | 82.9 % | +5.7 | [+0.0, +14.7] |
| family seedream4 | 80 | 57.5 % | 76.2 % | +18.8 | [+9.9, +28.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.hf.haywoodsloan

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7,img.hf.haywoodsloan GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.haywoodsloan ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +18.6 pts, CI [+15.2, +22.0] |
| A3b | pass | family flux1, family gan-faces, family gpt-4o-image-1, family gpt-image-1.5-2, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream4, threat gan_face, threat t2i |
| A4 | fail | pooled reals: Δ FPR +1.2 pts, CI [-0.1, +2.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: R1 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 96.1 % | +11.1 |
| F2 | 70 | 42.9 % | 48.6 % | +5.7 |
| F3 | 57 | 19.3 % | 43.9 % | +24.6 |
| F4 | 118 | 51.7 % | 74.6 % | +22.9 |
| F5 | 80 | 63.7 % | 78.8 % | +15.0 |
| F6 | 106 | 40.6 % | 72.6 % | +32.1 |
| F7 | 106 | 70.8 % | 89.6 % | +18.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.4 %, with 72.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 6.4 % | +4.0 | [-2.2, +9.2] |
| R2 | 612 | 2.5 % | 1.8 % | -0.7 | [-1.8, +0.8] |
| R3 | 589 | 4.9 % | 4.4 % | -0.5 | [-2.4, +1.2] |
| R4 | 493 | 4.5 % | 3.2 % | -1.2 | [-3.0, +0.6] |
| R5 | 548 | 1.3 % | 6.8 % | +5.5 | [+3.5, +7.7] |
| R6 | 28 | 14.3 % | 7.1 % | -7.1 | [-19.2, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 68.9 % | +4.3 | [+0.0, +8.7] |
| threat faceswap | 272 | 29.8 % | 37.9 % | +8.1 | [-2.1, +19.9] |
| threat gan_face | 153 | 9.8 % | 20.9 % | +11.1 | [+5.2, +17.0] |
| threat personalised | 115 | 91.3 % | 93.0 % | +1.7 | [-2.6, +7.0] |
| threat t2i | 1600 | 63.7 % | 81.4 % | +17.6 | [+15.7, +19.7] |
| family faceswap-stills | 272 | 29.8 % | 37.9 % | +8.1 | [-2.1, +19.9] |
| family flux1 | 85 | 62.4 % | 83.5 % | +21.2 | [+12.8, +31.0] |
| family flux2 | 70 | 38.6 % | 45.7 % | +7.1 | [-1.5, +16.7] |
| family gan-faces | 153 | 9.8 % | 20.9 % | +11.1 | [+5.2, +17.0] |
| family gpt-4o-image-1 | 41 | 31.7 % | 70.7 % | +39.0 | [+24.4, +53.7] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 40.4 % | +24.6 | [+14.0, +35.1] |
| family hidream | 50 | 84.0 % | 92.0 % | +8.0 | [-2.0, +19.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 85.1 % | +14.9 | [+5.8, +26.0] |
| family imagen3 | 53 | 50.9 % | 62.3 % | +11.3 | [+3.8, +20.8] |
| family imagen4 | 106 | 60.4 % | 87.7 % | +27.4 | [+18.4, +36.5] |
| family legacy | 145 | 91.7 % | 93.1 % | +1.4 | [-2.8, +5.7] |
| family long-tail | 186 | 67.7 % | 84.9 % | +17.2 | [+11.3, +24.0] |
| family midjourney | 153 | 79.7 % | 96.1 % | +16.3 | [+10.7, +22.0] |
| family modern-t2i-faces | 115 | 65.2 % | 90.4 % | +25.2 | [+16.8, +33.6] |
| family nano-banana | 118 | 40.7 % | 71.2 % | +30.5 | [+21.2, +40.2] |
| family partial-edits | 161 | 64.6 % | 68.9 % | +4.3 | [+0.0, +8.7] |
| family personalised | 115 | 91.3 % | 93.0 % | +1.7 | [-2.6, +7.0] |
| family qwen-image | 106 | 33.0 % | 67.0 % | +34.0 | [+23.6, +44.1] |
| family recraft | 54 | 79.6 % | 90.7 % | +11.1 | [+3.7, +21.2] |
| family sd3 | 75 | 98.7 % | 97.3 % | -1.3 | [-3.9, +0.0] |
| family seedream3 | 35 | 77.1 % | 85.7 % | +8.6 | [+0.0, +19.4] |
| family seedream4 | 80 | 56.2 % | 75.0 % | +18.8 | [+8.9, +29.1] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

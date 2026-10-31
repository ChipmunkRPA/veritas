# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.veritas_probe.pe_core_l14 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +28.6 pts, CI [+25.1, +32.2] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat t2i |
| A4 | fail | pooled reals: Δ FPR +1.5 pts, CI [-0.0, +3.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3; not established: R1, R2, R4, R5, R6; point above +2.0 without established harm: R4, R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 95.4 % | +13.1 |
| F2 | 70 | 27.1 % | 48.6 % | +21.4 |
| F3 | 57 | 14.0 % | 40.4 % | +26.3 |
| F4 | 118 | 38.1 % | 73.7 % | +35.6 |
| F5 | 80 | 75.0 % | 93.8 % | +18.8 |
| F6 | 106 | 28.3 % | 77.4 % | +49.1 |
| F7 | 106 | 50.9 % | 86.8 % | +35.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.1 %, with 73.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.9 % | 1.2 % | -0.7 | [-2.3, +0.6] |
| R2 | 612 | 5.6 % | 0.2 % | -5.4 | [-7.7, -1.9] |
| R3 | 589 | 3.9 % | 10.5 % | +6.6 | [+4.1, +9.2] |
| R4 | 493 | 8.9 % | 13.2 % | +4.3 | [+0.8, +7.7] |
| R5 | 548 | 1.3 % | 4.7 % | +3.5 | [+1.5, +5.5] |
| R6 | 28 | 10.7 % | 0.0 % | -10.7 | [-25.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 71.4 % | +6.8 | [+2.5, +11.8] |
| threat faceswap | 272 | 59.2 % | 55.9 % | -3.3 | [-8.0, +1.2] |
| threat gan_face | 153 | 51.6 % | 48.4 % | -3.3 | [-8.5, +1.3] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-1.7, +3.5] |
| threat t2i | 1600 | 61.6 % | 87.6 % | +26.0 | [+23.7, +28.5] |
| family faceswap-stills | 272 | 59.2 % | 55.9 % | -3.3 | [-8.0, +1.2] |
| family flux1 | 85 | 61.2 % | 77.6 % | +16.5 | [+8.3, +25.3] |
| family flux2 | 70 | 28.6 % | 51.4 % | +22.9 | [+12.5, +32.9] |
| family gan-faces | 153 | 51.6 % | 48.4 % | -3.3 | [-8.5, +1.3] |
| family gpt-4o-image-1 | 41 | 17.1 % | 78.0 % | +61.0 | [+46.3, +75.6] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 49.1 % | +35.1 | [+22.8, +47.4] |
| family hidream | 50 | 80.0 % | 96.0 % | +16.0 | [+4.1, +28.6] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 100.0 % | +23.4 | [+11.1, +36.4] |
| family imagen3 | 53 | 49.1 % | 92.5 % | +43.4 | [+30.2, +58.5] |
| family imagen4 | 106 | 52.8 % | 90.6 % | +37.7 | [+27.9, +48.0] |
| family legacy | 145 | 86.9 % | 94.5 % | +7.6 | [+3.0, +12.1] |
| family long-tail | 186 | 65.1 % | 89.2 % | +24.2 | [+17.6, +31.4] |
| family midjourney | 153 | 82.4 % | 95.4 % | +13.1 | [+7.8, +18.6] |
| family modern-t2i-faces | 115 | 54.8 % | 97.4 % | +42.6 | [+33.8, +51.1] |
| family nano-banana | 118 | 37.3 % | 79.7 % | +42.4 | [+33.3, +52.1] |
| family partial-edits | 161 | 64.6 % | 71.4 % | +6.8 | [+2.5, +11.8] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [-1.7, +3.5] |
| family qwen-image | 106 | 28.3 % | 80.2 % | +51.9 | [+42.9, +61.2] |
| family recraft | 54 | 87.0 % | 100.0 % | +13.0 | [+5.2, +22.2] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 77.1 % | 94.3 % | +17.1 | [+2.9, +31.4] |
| family seedream4 | 80 | 75.0 % | 93.8 % | +18.8 | [+10.1, +27.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

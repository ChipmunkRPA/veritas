# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7,img.veritas_probe.pe_core_l14 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +22.5 pts, CI [+18.8, +26.1] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family gan-faces, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat faceswap, threat gan_face, threat t2i |
| A4 | fail | pooled reals: Δ FPR +2.7 pts, CI [+1.6, +4.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3, R5; not established: R1, R2, R4, R6; point above +2.0 without established harm: R4 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 96.7 % | +11.8 |
| F2 | 70 | 42.9 % | 54.3 % | +11.4 |
| F3 | 57 | 19.3 % | 43.9 % | +24.6 |
| F4 | 118 | 51.7 % | 80.5 % | +28.8 |
| F5 | 80 | 63.7 % | 85.0 % | +21.3 |
| F6 | 106 | 40.6 % | 79.2 % | +38.7 |
| F7 | 106 | 70.8 % | 91.5 % | +20.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.4 %, with 75.9 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 1.4 % | -0.9 | [-2.1, +1.1] |
| R2 | 612 | 2.5 % | 0.0 % | -2.5 | [-3.9, -0.8] |
| R3 | 589 | 4.9 % | 12.4 % | +7.5 | [+4.9, +10.1] |
| R4 | 493 | 4.5 % | 9.1 % | +4.7 | [+1.8, +7.5] |
| R5 | 548 | 1.3 % | 6.8 % | +5.5 | [+3.1, +7.7] |
| R6 | 28 | 14.3 % | 0.0 % | -14.3 | [-29.2, -3.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 72.7 % | +8.1 | [+3.1, +13.0] |
| threat faceswap | 272 | 29.8 % | 52.6 % | +22.8 | [+11.5, +33.6] |
| threat gan_face | 153 | 9.8 % | 21.6 % | +11.8 | [+6.5, +17.6] |
| threat personalised | 115 | 91.3 % | 96.5 % | +5.2 | [+0.9, +10.4] |
| threat t2i | 1600 | 63.7 % | 89.7 % | +25.9 | [+23.7, +28.4] |
| family faceswap-stills | 272 | 29.8 % | 52.6 % | +22.8 | [+11.5, +33.6] |
| family flux1 | 85 | 62.4 % | 82.4 % | +20.0 | [+11.9, +29.1] |
| family flux2 | 70 | 38.6 % | 60.0 % | +21.4 | [+11.8, +32.0] |
| family gan-faces | 153 | 9.8 % | 21.6 % | +11.8 | [+6.5, +17.6] |
| family gpt-4o-image-1 | 41 | 31.7 % | 92.7 % | +61.0 | [+46.3, +75.6] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 56.1 % | +40.4 | [+28.1, +52.6] |
| family hidream | 50 | 84.0 % | 98.0 % | +14.0 | [+4.1, +24.5] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 100.0 % | +29.8 | [+16.7, +43.8] |
| family imagen3 | 53 | 50.9 % | 84.9 % | +34.0 | [+20.8, +47.2] |
| family imagen4 | 106 | 60.4 % | 96.2 % | +35.8 | [+26.5, +45.0] |
| family legacy | 145 | 91.7 % | 93.1 % | +1.4 | [-2.2, +5.3] |
| family long-tail | 186 | 67.7 % | 91.4 % | +23.7 | [+17.1, +31.2] |
| family midjourney | 153 | 79.7 % | 97.4 % | +17.6 | [+11.8, +23.8] |
| family modern-t2i-faces | 115 | 65.2 % | 97.4 % | +32.2 | [+23.0, +41.7] |
| family nano-banana | 118 | 40.7 % | 86.4 % | +45.8 | [+36.8, +55.5] |
| family partial-edits | 161 | 64.6 % | 72.7 % | +8.1 | [+3.1, +13.0] |
| family personalised | 115 | 91.3 % | 96.5 % | +5.2 | [+0.9, +10.4] |
| family qwen-image | 106 | 33.0 % | 82.1 % | +49.1 | [+38.7, +59.0] |
| family recraft | 54 | 79.6 % | 96.3 % | +16.7 | [+7.4, +26.9] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [-4.1, +4.0] |
| family seedream3 | 35 | 77.1 % | 97.1 % | +20.0 | [+8.6, +33.3] |
| family seedream4 | 80 | 56.2 % | 88.8 % | +32.5 | [+22.8, +43.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

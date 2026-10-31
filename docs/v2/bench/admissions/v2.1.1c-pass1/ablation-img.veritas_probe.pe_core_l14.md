# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.veritas_probe.pe_core_l14 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +26.8 pts, CI [+23.4, +30.3] |
| A3b | pass | family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat t2i |
| A4 | pass | pooled reals: Δ FPR -0.3 pts, CI [-1.5, +1.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R3, R5; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 95.4 % | +14.4 |
| F2 | 70 | 25.7 % | 47.1 % | +21.4 |
| F3 | 57 | 14.0 % | 33.3 % | +19.3 |
| F4 | 118 | 33.1 % | 69.5 % | +36.4 |
| F5 | 80 | 75.0 % | 93.8 % | +18.8 |
| F6 | 106 | 29.2 % | 73.6 % | +44.3 |
| F7 | 106 | 50.9 % | 84.0 % | +33.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 71.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 0.7 % | -0.7 | [-2.1, +0.4] |
| R2 | 612 | 4.7 % | 0.2 % | -4.6 | [-6.5, -1.7] |
| R3 | 589 | 3.7 % | 6.3 % | +2.5 | [+0.5, +4.6] |
| R4 | 493 | 15.0 % | 13.6 % | -1.4 | [-4.5, +1.8] |
| R5 | 548 | 1.5 % | 4.4 % | +2.9 | [+0.9, +4.9] |
| R6 | 28 | 3.6 % | 0.0 % | -3.6 | [-11.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 70.2 % | +6.8 | [+1.9, +11.8] |
| threat faceswap | 272 | 62.9 % | 47.8 % | -15.1 | [-24.8, -7.4] |
| threat gan_face | 153 | 53.6 % | 45.1 % | -8.5 | [-14.4, -3.3] |
| threat personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-3.5, +1.7] |
| threat t2i | 1600 | 61.1 % | 83.6 % | +22.4 | [+20.2, +25.0] |
| family faceswap-stills | 272 | 62.9 % | 47.8 % | -15.1 | [-24.8, -7.4] |
| family flux1 | 85 | 61.2 % | 72.9 % | +11.8 | [+4.7, +19.8] |
| family flux2 | 70 | 27.1 % | 47.1 % | +20.0 | [+10.6, +29.6] |
| family gan-faces | 153 | 53.6 % | 45.1 % | -8.5 | [-14.4, -3.3] |
| family gpt-4o-image-1 | 41 | 7.3 % | 61.0 % | +53.7 | [+39.0, +68.3] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 38.6 % | +24.6 | [+14.0, +36.8] |
| family hidream | 50 | 80.0 % | 96.0 % | +16.0 | [+4.1, +28.6] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 89.4 % | +14.9 | [+5.8, +26.1] |
| family imagen3 | 53 | 41.5 % | 83.0 % | +41.5 | [+28.3, +56.6] |
| family imagen4 | 106 | 50.9 % | 85.8 % | +34.9 | [+25.0, +45.0] |
| family legacy | 145 | 96.6 % | 95.2 % | -1.4 | [-5.6, +2.8] |
| family long-tail | 186 | 63.4 % | 84.4 % | +21.0 | [+14.6, +28.0] |
| family midjourney | 153 | 81.7 % | 94.8 % | +13.1 | [+8.2, +18.4] |
| family modern-t2i-faces | 115 | 54.8 % | 94.8 % | +40.0 | [+31.0, +48.8] |
| family nano-banana | 118 | 32.2 % | 71.2 % | +39.0 | [+29.5, +48.3] |
| family partial-edits | 161 | 63.4 % | 70.2 % | +6.8 | [+1.9, +11.8] |
| family personalised | 115 | 96.5 % | 95.7 % | -0.9 | [-3.5, +1.7] |
| family qwen-image | 106 | 29.2 % | 73.6 % | +44.3 | [+35.6, +53.2] |
| family recraft | 54 | 87.0 % | 98.1 % | +11.1 | [+3.6, +20.4] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 94.3 % | +20.0 | [+5.9, +35.1] |
| family seedream4 | 80 | 75.0 % | 93.8 % | +18.8 | [+10.1, +27.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

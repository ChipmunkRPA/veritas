# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.veritas_probe.pe_core_l14 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +25.9 pts, CI [+22.2, +29.6] |
| A3b | pass | family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat t2i |
| A4 | fail | pooled reals: Δ FPR -0.6 pts, CI [-2.5, +1.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3; not established: R1, R2, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 96.7 % | +15.0 |
| F2 | 70 | 30.0 % | 52.9 % | +22.9 |
| F3 | 57 | 15.8 % | 33.3 % | +17.5 |
| F4 | 118 | 44.9 % | 78.0 % | +33.1 |
| F5 | 80 | 61.3 % | 82.5 % | +21.2 |
| F6 | 106 | 27.4 % | 69.8 % | +42.5 |
| F7 | 106 | 60.4 % | 89.6 % | +29.2 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 71.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 1.4 % | -1.4 | [-3.6, +1.5] |
| R2 | 612 | 8.8 % | 0.0 % | -8.8 | [-13.7, -2.3] |
| R3 | 589 | 4.9 % | 9.7 % | +4.8 | [+2.3, +7.3] |
| R4 | 493 | 8.7 % | 9.9 % | +1.2 | [-1.6, +4.1] |
| R5 | 548 | 1.5 % | 4.2 % | +2.7 | [+0.7, +4.7] |
| R6 | 28 | 14.3 % | 0.0 % | -14.3 | [-30.8, -3.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 68.9 % | +2.5 | [-3.7, +8.1] |
| threat faceswap | 272 | 55.5 % | 37.9 % | -17.6 | [-28.0, -9.2] |
| threat gan_face | 153 | 25.5 % | 15.7 % | -9.8 | [-16.4, -3.3] |
| threat personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-1.7, +3.5] |
| threat t2i | 1600 | 63.8 % | 84.2 % | +20.4 | [+18.3, +22.8] |
| family faceswap-stills | 272 | 55.5 % | 37.9 % | -17.6 | [-28.0, -9.2] |
| family flux1 | 85 | 61.2 % | 70.6 % | +9.4 | [+3.6, +16.0] |
| family flux2 | 70 | 30.0 % | 51.4 % | +21.4 | [+10.4, +32.4] |
| family gan-faces | 153 | 25.5 % | 15.7 % | -9.8 | [-16.4, -3.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 75.6 % | +41.5 | [+26.8, +56.1] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 35.1 % | +19.3 | [+10.5, +29.8] |
| family hidream | 50 | 82.0 % | 96.0 % | +14.0 | [+2.0, +26.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 95.7 % | +19.1 | [+8.2, +31.3] |
| family imagen3 | 53 | 52.8 % | 84.9 % | +32.1 | [+18.9, +45.3] |
| family imagen4 | 106 | 59.4 % | 89.6 % | +30.2 | [+21.2, +39.6] |
| family legacy | 145 | 89.0 % | 92.4 % | +3.4 | [-1.5, +8.1] |
| family long-tail | 186 | 65.6 % | 83.3 % | +17.7 | [+12.6, +23.8] |
| family midjourney | 153 | 81.7 % | 96.7 % | +15.0 | [+9.6, +20.6] |
| family modern-t2i-faces | 115 | 66.1 % | 95.7 % | +29.6 | [+20.4, +38.9] |
| family nano-banana | 118 | 44.9 % | 77.1 % | +32.2 | [+23.9, +41.0] |
| family partial-edits | 161 | 66.5 % | 68.9 % | +2.5 | [-3.7, +8.1] |
| family personalised | 115 | 95.7 % | 96.5 % | +0.9 | [-1.7, +3.5] |
| family qwen-image | 106 | 28.3 % | 75.5 % | +47.2 | [+38.1, +56.8] |
| family recraft | 54 | 87.0 % | 98.1 % | +11.1 | [+3.6, +20.4] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 82.9 % | 94.3 % | +11.4 | [+2.9, +22.9] |
| family seedream4 | 80 | 61.3 % | 82.5 % | +21.2 | [+12.3, +31.2] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq,img.veritas_probe.pe_core_l14 CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +23.6 pts, CI [+20.2, +27.0] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family hidream, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat faceswap, threat t2i |
| A4 | fail | pooled reals: Δ FPR +2.1 pts, CI [+0.7, +3.7] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3, R4, R5; not established: R1, R2, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 96.1 % | +12.4 |
| F2 | 70 | 37.1 % | 52.9 % | +15.7 |
| F3 | 57 | 15.8 % | 43.9 % | +28.1 |
| F4 | 118 | 48.3 % | 77.1 % | +28.8 |
| F5 | 80 | 76.2 % | 91.2 % | +15.0 |
| F6 | 106 | 39.6 % | 79.2 % | +39.6 |
| F7 | 106 | 66.0 % | 91.5 % | +25.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 52.4 %, with 76.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.2 % | 1.7 % | -2.6 | [-4.9, +1.1] |
| R2 | 612 | 3.8 % | 0.2 % | -3.6 | [-5.4, -1.6] |
| R3 | 589 | 4.6 % | 11.0 % | +6.5 | [+4.1, +9.1] |
| R4 | 493 | 4.5 % | 10.5 % | +6.1 | [+3.2, +8.9] |
| R5 | 548 | 2.0 % | 6.4 % | +4.4 | [+2.4, +6.6] |
| R6 | 28 | 10.7 % | 0.0 % | -10.7 | [-25.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 72.7 % | +7.5 | [+2.5, +13.0] |
| threat faceswap | 272 | 20.6 % | 46.3 % | +25.7 | [+11.7, +38.5] |
| threat gan_face | 153 | 43.8 % | 46.4 % | +2.6 | [-1.3, +6.5] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 67.6 % | 90.5 % | +22.9 | [+20.8, +25.3] |
| family faceswap-stills | 272 | 20.6 % | 46.3 % | +25.7 | [+11.7, +38.5] |
| family flux1 | 85 | 67.1 % | 83.5 % | +16.5 | [+9.2, +24.7] |
| family flux2 | 70 | 37.1 % | 54.3 % | +17.1 | [+8.7, +26.7] |
| family gan-faces | 153 | 43.8 % | 46.4 % | +2.6 | [-1.3, +6.5] |
| family gpt-4o-image-1 | 41 | 22.0 % | 90.2 % | +68.3 | [+53.7, +82.9] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 59.6 % | +42.1 | [+29.8, +54.4] |
| family hidream | 50 | 86.0 % | 98.0 % | +12.0 | [+2.1, +23.1] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 100.0 % | +25.5 | [+13.0, +39.1] |
| family imagen3 | 53 | 52.8 % | 88.7 % | +35.8 | [+22.6, +49.1] |
| family imagen4 | 106 | 65.1 % | 96.2 % | +31.1 | [+21.9, +40.8] |
| family legacy | 145 | 95.2 % | 95.9 % | +0.7 | [-2.9, +4.1] |
| family long-tail | 186 | 69.9 % | 93.5 % | +23.7 | [+16.9, +31.1] |
| family midjourney | 153 | 83.7 % | 98.0 % | +14.4 | [+9.0, +20.3] |
| family modern-t2i-faces | 115 | 72.2 % | 98.3 % | +26.1 | [+18.1, +34.3] |
| family nano-banana | 118 | 44.1 % | 84.7 % | +40.7 | [+32.2, +49.6] |
| family partial-edits | 161 | 65.2 % | 72.7 % | +7.5 | [+2.5, +13.0] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 37.7 % | 82.1 % | +44.3 | [+34.3, +54.5] |
| family recraft | 54 | 85.2 % | 96.3 % | +11.1 | [+3.6, +20.4] |
| family sd3 | 75 | 98.7 % | 100.0 % | +1.3 | [+0.0, +4.2] |
| family seedream3 | 35 | 82.9 % | 97.1 % | +14.3 | [+2.9, +25.7] |
| family seedream4 | 80 | 75.0 % | 93.8 % | +18.8 | [+10.8, +27.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

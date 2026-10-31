# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.veritas_probe.pe_core_l14 CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +23.1 pts, CI [+19.5, +26.6] |
| A3b | pass | family faceswap-stills, family flux1, family flux2, family gpt-4o-image-1, family gpt-image-1.5-2, family ideogram, family imagen3, family imagen4, family long-tail, family midjourney, family modern-t2i-faces, family nano-banana, family qwen-image, family recraft, family seedream3, family seedream4, threat faceswap, threat t2i |
| A4 | fail | pooled reals: Δ FPR +2.7 pts, CI [+1.5, +4.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R3, R5; not established: R1, R2, R4, R6; point above +2.0 without established harm: R4 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 97.4 % | +12.4 |
| F2 | 70 | 42.9 % | 57.1 % | +14.3 |
| F3 | 57 | 22.8 % | 45.6 % | +22.8 |
| F4 | 118 | 52.5 % | 82.2 % | +29.7 |
| F5 | 80 | 63.7 % | 83.8 % | +20.0 |
| F6 | 106 | 39.6 % | 80.2 % | +40.6 |
| F7 | 106 | 72.6 % | 94.3 % | +21.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.2 %, with 77.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 2.1 % | -0.2 | [-1.5, +2.7] |
| R2 | 612 | 2.8 % | 0.0 % | -2.8 | [-4.3, -0.9] |
| R3 | 589 | 5.1 % | 11.7 % | +6.6 | [+4.2, +9.2] |
| R4 | 493 | 4.7 % | 8.7 % | +4.1 | [+1.2, +7.1] |
| R5 | 548 | 0.5 % | 7.1 % | +6.6 | [+4.6, +8.8] |
| R6 | 28 | 14.3 % | 0.0 % | -14.3 | [-29.2, -3.3] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 72.7 % | +8.1 | [+3.1, +13.0] |
| threat faceswap | 272 | 5.9 % | 38.6 % | +32.7 | [+19.9, +44.3] |
| threat gan_face | 153 | 14.4 % | 20.3 % | +5.9 | [+1.3, +10.5] |
| threat personalised | 115 | 91.3 % | 97.4 % | +6.1 | [+1.7, +11.3] |
| threat t2i | 1600 | 65.4 % | 89.9 % | +24.4 | [+22.2, +26.9] |
| family faceswap-stills | 272 | 5.9 % | 38.6 % | +32.7 | [+19.9, +44.3] |
| family flux1 | 85 | 62.4 % | 82.4 % | +20.0 | [+11.9, +29.1] |
| family flux2 | 70 | 38.6 % | 58.6 % | +20.0 | [+10.3, +30.6] |
| family gan-faces | 153 | 14.4 % | 20.3 % | +5.9 | [+1.3, +10.5] |
| family gpt-4o-image-1 | 41 | 31.7 % | 90.2 % | +58.5 | [+43.9, +73.2] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 56.1 % | +40.4 | [+28.1, +52.6] |
| family hidream | 50 | 88.0 % | 98.0 % | +10.0 | [+2.0, +20.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 100.0 % | +29.8 | [+16.7, +43.8] |
| family imagen3 | 53 | 52.8 % | 86.8 % | +34.0 | [+20.8, +49.1] |
| family imagen4 | 106 | 65.1 % | 96.2 % | +31.1 | [+21.3, +40.8] |
| family legacy | 145 | 90.3 % | 93.1 % | +2.8 | [-1.4, +6.9] |
| family long-tail | 186 | 67.2 % | 93.0 % | +25.8 | [+18.8, +33.7] |
| family midjourney | 153 | 82.4 % | 97.4 % | +15.0 | [+9.7, +20.8] |
| family modern-t2i-faces | 115 | 73.0 % | 97.4 % | +24.3 | [+17.1, +32.4] |
| family nano-banana | 118 | 44.9 % | 87.3 % | +42.4 | [+33.3, +51.3] |
| family partial-edits | 161 | 64.6 % | 72.7 % | +8.1 | [+3.1, +13.0] |
| family personalised | 115 | 91.3 % | 97.4 % | +6.1 | [+1.7, +11.3] |
| family qwen-image | 106 | 34.9 % | 83.0 % | +48.1 | [+38.0, +57.9] |
| family recraft | 54 | 81.5 % | 96.3 % | +14.8 | [+5.6, +25.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [-4.1, +4.0] |
| family seedream3 | 35 | 77.1 % | 97.1 % | +20.0 | [+8.6, +33.3] |
| family seedream4 | 80 | 57.5 % | 87.5 % | +30.0 | [+20.0, +40.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

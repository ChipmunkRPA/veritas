# Ablation: img.veritas_probe.pe_core_l14

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.veritas_probe.pe_core_l14 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.veritas_probe.pe_core_l14 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +8.1 pts, CI [+5.5, +10.6] |
| A3b | pass | family gpt-4o-image-1, family gpt-image-1.5-2, family seedream4 |
| A4 | fail | pooled reals: Δ FPR -0.0 pts, CI [-1.0, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R4; not established: R1, R2, R3, R5, R6; point above +2.0 without established harm: R5 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 98.7 % | 98.7 % | +0.0 |
| F2 | 70 | 58.6 % | 62.9 % | +4.3 |
| F3 | 57 | 52.6 % | 64.9 % | +12.3 |
| F4 | 118 | 83.9 % | 92.4 % | +8.5 |
| F5 | 80 | 75.0 % | 95.0 % | +20.0 |
| F6 | 106 | 84.9 % | 92.5 % | +7.5 |
| F7 | 106 | 94.3 % | 98.1 % | +3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 78.3 %, with 86.3 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 9.7 % | 6.8 % | -2.8 | [-7.8, -0.6] |
| R2 | 612 | 0.3 % | 0.0 % | -0.3 | [-1.1, +0.0] |
| R3 | 589 | 14.8 % | 10.4 % | -4.4 | [-7.3, -1.5] |
| R4 | 495 | 1.2 % | 5.9 % | +4.6 | [+2.6, +6.9] |
| R5 | 549 | 3.6 % | 6.6 % | +2.9 | [+1.1, +4.9] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 84.5 % | 82.0 % | -2.5 | [-6.2, +1.2] |
| threat faceswap | 272 | 82.0 % | 81.6 % | -0.4 | [-4.0, +3.2] |
| threat gan_face | 153 | 64.7 % | 58.8 % | -5.9 | [-11.1, -0.7] |
| threat personalised | 115 | 97.4 % | 99.1 % | +1.7 | [+0.0, +4.3] |
| threat t2i | 1603 | 89.3 % | 93.9 % | +4.7 | [+3.5, +5.9] |
| family faceswap-stills | 272 | 82.0 % | 81.6 % | -0.4 | [-4.0, +3.2] |
| family flux1 | 85 | 87.1 % | 87.1 % | +0.0 | [-6.9, +6.0] |
| family flux2 | 70 | 60.0 % | 68.6 % | +8.6 | [+0.0, +17.3] |
| family gan-faces | 153 | 64.7 % | 58.8 % | -5.9 | [-11.1, -0.7] |
| family gpt-4o-image-1 | 41 | 78.0 % | 95.1 % | +17.1 | [+7.3, +29.3] |
| family gpt-image-1.5-2 | 57 | 57.9 % | 70.2 % | +12.3 | [+1.8, +22.8] |
| family hidream | 50 | 100.0 % | 98.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 97.9 % | 100.0 % | +2.1 | [+0.0, +6.8] |
| family imagen3 | 53 | 100.0 % | 98.1 % | -1.9 | [-5.7, +0.0] |
| family imagen4 | 106 | 97.2 % | 99.1 % | +1.9 | [+0.0, +4.9] |
| family legacy | 148 | 85.1 % | 88.5 % | +3.4 | [+0.7, +6.2] |
| family long-tail | 186 | 88.7 % | 96.8 % | +8.1 | [+4.3, +12.5] |
| family midjourney | 153 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 94.8 % | 97.4 % | +2.6 | [+0.0, +5.5] |
| family nano-banana | 118 | 86.4 % | 94.9 % | +8.5 | [+3.4, +13.7] |
| family partial-edits | 161 | 84.5 % | 82.0 % | -2.5 | [-6.2, +1.2] |
| family personalised | 115 | 97.4 % | 99.1 % | +1.7 | [+0.0, +4.3] |
| family qwen-image | 106 | 85.8 % | 94.3 % | +8.5 | [+1.9, +15.0] |
| family recraft | 54 | 98.1 % | 98.1 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 97.1 % | 100.0 % | +2.9 | [+0.0, +8.8] |
| family seedream4 | 80 | 85.0 % | 98.8 % | +13.8 | [+6.2, +21.5] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.forensic.copymove

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.forensic.copymove GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.copymove ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4993 (2692 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.7 pts, CI [+0.8, +2.8] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.4, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 81.7 % | +0.7 |
| F2 | 70 | 22.9 % | 25.7 % | +2.9 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 31.4 % | 34.7 % | +3.4 |
| F5 | 80 | 75.0 % | 76.2 % | +1.2 |
| F6 | 106 | 29.2 % | 30.2 % | +0.9 |
| F7 | 106 | 50.0 % | 52.8 % | +2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 43.4 %, with 45.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.2 % | 2.6 % | +1.4 | [+0.0, +3.3] |
| R2 | 611 | 4.9 % | 4.9 % | +0.0 | [+0.0, +0.0] |
| R3 | 588 | 3.6 % | 4.3 % | +0.7 | [+0.0, +1.5] |
| R4 | 493 | 14.2 % | 12.4 % | -1.8 | [-3.0, -0.8] |
| R5 | 548 | 1.1 % | 0.7 % | -0.4 | [-1.3, +0.4] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.0 % | 64.0 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 64.0 % | 45.2 % | -18.7 | [-32.1, -6.6] |
| threat gan_face | 153 | 54.2 % | 51.6 % | -2.6 | [-6.5, +0.7] |
| threat personalised | 115 | 95.7 % | 97.4 % | +1.7 | [+0.0, +4.3] |
| threat t2i | 1600 | 60.4 % | 61.8 % | +1.4 | [+0.8, +2.1] |
| family faceswap-stills | 272 | 64.0 % | 45.2 % | -18.7 | [-32.1, -6.6] |
| family flux1 | 85 | 61.2 % | 63.5 % | +2.4 | [+0.0, +6.0] |
| family flux2 | 70 | 24.3 % | 27.1 % | +2.9 | [+0.0, +7.2] |
| family gan-faces | 153 | 54.2 % | 51.6 % | -2.6 | [-6.5, +0.7] |
| family gpt-4o-image-1 | 41 | 7.3 % | 9.8 % | +2.4 | [+0.0, +7.3] |
| family gpt-image-1.5-2 | 57 | 10.5 % | 14.0 % | +3.5 | [+0.0, +8.8] |
| family hidream | 50 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 43.4 % | 47.2 % | +3.8 | [+0.0, +9.4] |
| family imagen4 | 106 | 50.9 % | 54.7 % | +3.8 | [+0.9, +7.6] |
| family legacy | 145 | 92.4 % | 90.3 % | -2.1 | [-5.1, +0.7] |
| family long-tail | 186 | 62.9 % | 65.1 % | +2.2 | [+0.5, +4.3] |
| family midjourney | 153 | 81.0 % | 81.0 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 55.7 % | 57.4 % | +1.7 | [+0.0, +3.9] |
| family nano-banana | 118 | 30.5 % | 33.1 % | +2.5 | [+0.0, +5.9] |
| family partial-edits | 161 | 64.0 % | 64.0 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 95.7 % | 97.4 % | +1.7 | [+0.0, +4.3] |
| family qwen-image | 106 | 29.2 % | 29.2 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 74.3 % | 80.0 % | +5.7 | [+0.0, +13.9] |
| family seedream4 | 80 | 75.0 % | 76.2 % | +1.2 | [+0.0, +3.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.forensic.copymove

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.forensic.copymove GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.copymove ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4993 (2692 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +7.2 pts, CI [+5.2, +9.1] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.7, +0.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 86.9 % | +5.2 |
| F2 | 70 | 30.0 % | 35.7 % | +5.7 |
| F3 | 57 | 15.8 % | 21.1 % | +5.3 |
| F4 | 118 | 44.9 % | 55.1 % | +10.2 |
| F5 | 80 | 61.3 % | 66.2 % | +5.0 |
| F6 | 106 | 27.4 % | 36.8 % | +9.4 |
| F7 | 106 | 60.4 % | 69.8 % | +9.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 53.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 3.5 % | +1.2 | [+0.0, +1.7] |
| R2 | 611 | 7.7 % | 9.5 % | +1.8 | [+0.3, +3.3] |
| R3 | 588 | 4.4 % | 5.1 % | +0.7 | [+0.0, +1.6] |
| R4 | 493 | 9.3 % | 5.3 % | -4.1 | [-6.1, -2.0] |
| R5 | 548 | 0.9 % | 0.9 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-11.6, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 66.5 % | +0.6 | [+0.0, +1.9] |
| threat faceswap | 272 | 54.0 % | 52.6 % | -1.5 | [-4.4, +1.6] |
| threat gan_face | 153 | 24.8 % | 24.8 % | +0.0 | [-3.3, +3.3] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 62.2 % | 64.1 % | +1.9 | [+1.2, +2.6] |
| family faceswap-stills | 272 | 54.0 % | 52.6 % | -1.5 | [-4.4, +1.6] |
| family flux1 | 85 | 61.2 % | 62.4 % | +1.2 | [+0.0, +3.7] |
| family flux2 | 70 | 30.0 % | 31.4 % | +1.4 | [+0.0, +4.5] |
| family gan-faces | 153 | 24.8 % | 24.8 % | +0.0 | [-3.3, +3.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 34.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 76.6 % | +6.4 | [+0.0, +14.6] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 58.5 % | 61.3 % | +2.8 | [+0.0, +6.6] |
| family legacy | 145 | 84.8 % | 83.4 % | -1.4 | [-4.5, +1.3] |
| family long-tail | 186 | 65.1 % | 67.2 % | +2.2 | [+0.5, +4.5] |
| family midjourney | 153 | 81.0 % | 82.4 % | +1.3 | [+0.0, +3.4] |
| family modern-t2i-faces | 115 | 65.2 % | 69.6 % | +4.3 | [+0.9, +8.7] |
| family nano-banana | 118 | 41.5 % | 44.1 % | +2.5 | [+0.0, +5.9] |
| family partial-edits | 161 | 65.8 % | 66.5 % | +0.6 | [+0.0, +1.9] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 27.4 % | 30.2 % | +2.8 | [+0.0, +6.7] |
| family recraft | 54 | 81.5 % | 87.0 % | +5.6 | [+0.0, +12.5] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 74.3 % | 80.0 % | +5.7 | [+0.0, +13.5] |
| family seedream4 | 80 | 58.8 % | 61.3 % | +2.5 | [+0.0, +6.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

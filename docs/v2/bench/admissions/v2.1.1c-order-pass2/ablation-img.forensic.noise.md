# Ablation: img.forensic.noise

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.forensic.noise GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.noise ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.4 pts, CI [+1.4, +3.6] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.3 pts, CI [-0.8, +0.2] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 84.3 % | +2.6 |
| F2 | 70 | 30.0 % | 32.9 % | +2.9 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 44.9 % | 44.9 % | +0.0 |
| F5 | 80 | 61.3 % | 62.5 % | +1.2 |
| F6 | 106 | 27.4 % | 33.0 % | +5.7 |
| F7 | 106 | 60.4 % | 65.1 % | +4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 48.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.1 % | -0.7 | [-1.5, +0.0] |
| R2 | 612 | 8.8 % | 8.8 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 4.9 % | 5.1 % | +0.2 | [+0.0, +0.5] |
| R4 | 493 | 8.7 % | 5.9 % | -2.8 | [-4.7, -1.0] |
| R5 | 548 | 1.5 % | 2.9 % | +1.5 | [+0.5, +2.6] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 67.1 % | +0.6 | [+0.0, +1.9] |
| threat faceswap | 272 | 55.5 % | 63.6 % | +8.1 | [+4.6, +12.3] |
| threat gan_face | 153 | 25.5 % | 26.8 % | +1.3 | [-1.3, +3.9] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 63.8 % | 63.7 % | -0.1 | [-0.6, +0.4] |
| family faceswap-stills | 272 | 55.5 % | 63.6 % | +8.1 | [+4.6, +12.3] |
| family flux1 | 85 | 61.2 % | 62.4 % | +1.2 | [+0.0, +3.7] |
| family flux2 | 70 | 30.0 % | 30.0 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 25.5 % | 26.8 % | +1.3 | [-1.3, +3.9] |
| family gpt-4o-image-1 | 41 | 34.1 % | 31.7 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 59.4 % | 57.5 % | -1.9 | [-4.9, +0.0] |
| family legacy | 145 | 89.0 % | 89.0 % | +0.0 | [-3.4, +3.4] |
| family long-tail | 186 | 65.6 % | 66.7 % | +1.1 | [+0.0, +2.7] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 66.1 % | 68.7 % | +2.6 | [+0.0, +5.7] |
| family nano-banana | 118 | 44.9 % | 43.2 % | -1.7 | [-4.3, +0.0] |
| family partial-edits | 161 | 66.5 % | 67.1 % | +0.6 | [+0.0, +1.9] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 28.3 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 85.2 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 80.0 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 61.3 % | 60.0 % | -1.3 | [-3.8, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

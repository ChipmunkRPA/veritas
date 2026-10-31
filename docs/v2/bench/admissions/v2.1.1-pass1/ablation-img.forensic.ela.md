# Ablation: img.forensic.ela

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.forensic.ela GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.ela ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.2 pts, CI [+1.0, +3.6] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.5 pts, CI [-1.1, +0.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 85.0 % | +2.6 |
| F2 | 70 | 27.1 % | 30.0 % | +2.9 |
| F3 | 57 | 14.0 % | 15.8 % | +1.8 |
| F4 | 118 | 38.1 % | 39.0 % | +0.8 |
| F5 | 80 | 75.0 % | 75.0 % | +0.0 |
| F6 | 106 | 28.3 % | 32.1 % | +3.8 |
| F7 | 106 | 50.9 % | 54.7 % | +3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.1 %, with 47.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.9 % | 1.9 % | +0.0 | [-0.9, +0.5] |
| R2 | 612 | 5.6 % | 4.7 % | -0.8 | [-1.7, +0.5] |
| R3 | 589 | 3.9 % | 4.6 % | +0.7 | [-0.2, +1.6] |
| R4 | 493 | 8.9 % | 7.1 % | -1.8 | [-4.3, +0.4] |
| R5 | 548 | 1.3 % | 0.7 % | -0.5 | [-1.3, +0.2] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-10.7, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.8 % | +1.2 | [+0.0, +3.1] |
| threat faceswap | 272 | 59.2 % | 58.8 % | -0.4 | [-3.5, +3.0] |
| threat gan_face | 153 | 51.6 % | 52.3 % | +0.7 | [-2.0, +3.9] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 61.6 % | 62.1 % | +0.4 | [-0.2, +1.1] |
| family faceswap-stills | 272 | 59.2 % | 58.8 % | -0.4 | [-3.5, +3.0] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 51.6 % | 52.3 % | +0.7 | [-2.0, +3.9] |
| family gpt-4o-image-1 | 41 | 17.1 % | 14.6 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 15.8 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 80.0 % | 78.0 % | -2.0 | [-10.0, +4.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 49.1 % | 50.9 % | +1.9 | [+0.0, +5.7] |
| family imagen4 | 106 | 52.8 % | 51.9 % | -0.9 | [-4.0, +1.9] |
| family legacy | 145 | 86.9 % | 89.7 % | +2.8 | [+0.0, +6.0] |
| family long-tail | 186 | 65.1 % | 65.6 % | +0.5 | [+0.0, +1.6] |
| family midjourney | 153 | 82.4 % | 82.4 % | +0.0 | [-1.9, +1.9] |
| family modern-t2i-faces | 115 | 54.8 % | 56.5 % | +1.7 | [+0.0, +4.5] |
| family nano-banana | 118 | 37.3 % | 38.1 % | +0.8 | [-1.7, +4.2] |
| family partial-edits | 161 | 64.6 % | 65.8 % | +1.2 | [+0.0, +3.1] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 29.2 % | +0.9 | [+0.0, +2.9] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 80.0 % | +2.9 | [+0.0, +8.8] |
| family seedream4 | 80 | 75.0 % | 72.5 % | -2.5 | [-6.3, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

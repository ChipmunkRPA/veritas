# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +3.2 pts, CI [+1.5, +5.1] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-1.1, +0.9] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 83.7 % | -1.3 |
| F2 | 70 | 31.4 % | 37.1 % | +5.7 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 43.2 % | 48.3 % | +5.1 |
| F5 | 80 | 76.2 % | 76.2 % | +0.0 |
| F6 | 106 | 32.1 % | 39.6 % | +7.5 |
| F7 | 106 | 60.4 % | 66.0 % | +5.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 49.2 %, with 52.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.6 % | 4.2 % | +1.7 | [-1.3, +3.2] |
| R2 | 612 | 7.0 % | 3.8 % | -3.3 | [-4.5, -0.8] |
| R3 | 589 | 4.1 % | 4.6 % | +0.5 | [-0.7, +1.8] |
| R4 | 493 | 4.3 % | 4.5 % | +0.2 | [-1.2, +1.6] |
| R5 | 548 | 0.5 % | 2.0 % | +1.5 | [+0.4, +2.7] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-20.8, +12.5] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 65.2 % | -0.6 | [-3.7, +2.5] |
| threat faceswap | 272 | 15.8 % | 20.6 % | +4.8 | [+2.5, +7.8] |
| threat gan_face | 153 | 44.4 % | 43.8 % | -0.7 | [-2.6, +1.3] |
| threat personalised | 115 | 99.1 % | 96.5 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 63.8 % | 67.6 % | +3.7 | [+2.5, +5.1] |
| family faceswap-stills | 272 | 15.8 % | 20.6 % | +4.8 | [+2.5, +7.8] |
| family flux1 | 85 | 62.4 % | 67.1 % | +4.7 | [+1.1, +9.6] |
| family flux2 | 70 | 31.4 % | 37.1 % | +5.7 | [+0.0, +12.5] |
| family gan-faces | 153 | 44.4 % | 43.8 % | -0.7 | [-2.6, +1.3] |
| family gpt-4o-image-1 | 41 | 17.1 % | 22.0 % | +4.9 | [-4.9, +14.6] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 80.0 % | 86.0 % | +6.0 | [-2.1, +15.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 74.5 % | -2.1 | [-9.8, +4.4] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [-9.4, +11.3] |
| family imagen4 | 106 | 58.5 % | 65.1 % | +6.6 | [+2.0, +11.5] |
| family legacy | 145 | 85.5 % | 95.2 % | +9.7 | [+5.0, +14.6] |
| family long-tail | 186 | 66.7 % | 69.9 % | +3.2 | [-0.6, +7.3] |
| family midjourney | 153 | 83.7 % | 83.7 % | +0.0 | [-3.1, +3.4] |
| family modern-t2i-faces | 115 | 65.2 % | 72.2 % | +7.0 | [+2.5, +12.3] |
| family nano-banana | 118 | 41.5 % | 44.1 % | +2.5 | [-3.4, +9.3] |
| family partial-edits | 161 | 65.8 % | 65.2 % | -0.6 | [-3.7, +2.5] |
| family personalised | 115 | 99.1 % | 96.5 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 30.2 % | 37.7 % | +7.5 | [+1.9, +14.2] |
| family recraft | 54 | 87.0 % | 85.2 % | -1.9 | [-7.7, +3.7] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 80.0 % | 82.9 % | +2.9 | [-5.9, +11.4] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [-3.7, +3.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

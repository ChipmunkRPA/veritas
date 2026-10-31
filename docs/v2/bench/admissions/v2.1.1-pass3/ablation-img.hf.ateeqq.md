# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +9.7 pts, CI [+6.7, +12.6] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.8 pts, CI [-2.3, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 79.7 % | 85.0 % | +5.2 |
| F2 | 70 | 30.0 % | 42.9 % | +12.9 |
| F3 | 57 | 15.8 % | 19.3 % | +3.5 |
| F4 | 118 | 41.5 % | 51.7 % | +10.2 |
| F5 | 80 | 57.5 % | 63.7 % | +6.2 |
| F6 | 106 | 25.5 % | 40.6 % | +15.1 |
| F7 | 106 | 55.7 % | 70.8 % | +15.1 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 43.7 %, with 53.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.7 % | 2.4 % | +0.7 | [-0.3, +2.2] |
| R2 | 612 | 7.7 % | 2.5 % | -5.2 | [-8.4, +0.0] |
| R3 | 589 | 4.2 % | 4.9 % | +0.7 | [-0.7, +2.1] |
| R4 | 493 | 4.5 % | 4.5 % | +0.0 | [-1.4, +1.4] |
| R5 | 548 | 0.9 % | 1.3 % | +0.4 | [-0.7, +1.5] |
| R6 | 28 | 10.7 % | 14.3 % | +3.6 | [-8.3, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 64.6 % | -0.6 | [-3.7, +2.5] |
| threat faceswap | 272 | 51.1 % | 29.8 % | -21.3 | [-33.0, -8.2] |
| threat gan_face | 153 | 20.9 % | 9.8 % | -11.1 | [-17.0, -5.9] |
| threat personalised | 115 | 95.7 % | 91.3 % | -4.3 | [-8.7, +0.0] |
| threat t2i | 1600 | 60.6 % | 63.7 % | +3.1 | [+1.7, +4.6] |
| family faceswap-stills | 272 | 51.1 % | 29.8 % | -21.3 | [-33.0, -8.2] |
| family flux1 | 85 | 61.2 % | 62.4 % | +1.2 | [-2.4, +5.7] |
| family flux2 | 70 | 30.0 % | 38.6 % | +8.6 | [-1.4, +18.8] |
| family gan-faces | 153 | 20.9 % | 9.8 % | -11.1 | [-17.0, -5.9] |
| family gpt-4o-image-1 | 41 | 31.7 % | 31.7 % | +0.0 | [-12.2, +14.6] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [-5.3, +5.3] |
| family hidream | 50 | 82.0 % | 84.0 % | +2.0 | [-4.3, +9.3] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 70.2 % | +0.0 | [-6.4, +6.5] |
| family imagen3 | 53 | 50.9 % | 50.9 % | +0.0 | [-9.4, +11.3] |
| family imagen4 | 106 | 56.6 % | 60.4 % | +3.8 | [+0.9, +7.5] |
| family legacy | 145 | 82.8 % | 91.7 % | +9.0 | [+3.6, +14.7] |
| family long-tail | 186 | 63.4 % | 67.7 % | +4.3 | [-0.6, +9.0] |
| family midjourney | 153 | 79.1 % | 79.7 % | +0.7 | [-2.0, +3.4] |
| family modern-t2i-faces | 115 | 58.3 % | 65.2 % | +7.0 | [+3.3, +11.3] |
| family nano-banana | 118 | 41.5 % | 40.7 % | -0.8 | [-7.6, +5.9] |
| family partial-edits | 161 | 65.2 % | 64.6 % | -0.6 | [-3.7, +2.5] |
| family personalised | 115 | 95.7 % | 91.3 % | -4.3 | [-8.7, +0.0] |
| family qwen-image | 106 | 24.5 % | 33.0 % | +8.5 | [+2.8, +15.1] |
| family recraft | 54 | 81.5 % | 79.6 % | -1.9 | [-7.7, +3.8] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 77.1 % | +2.9 | [-5.9, +11.4] |
| family seedream4 | 80 | 57.5 % | 56.2 % | -1.2 | [-6.3, +3.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

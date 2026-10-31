# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.hf.dima806 CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -3.3 pts, CI [-4.9, -1.9] |
| A3b | fail | family faceswap-stills, threat faceswap |
| A4 | fail | pooled reals: Δ FPR +0.7 pts, CI [+0.2, +1.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 81.7 % | -3.3 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 22.8 % | 17.5 % | -5.3 |
| F4 | 118 | 52.5 % | 47.5 % | -5.1 |
| F5 | 80 | 63.7 % | 58.8 % | -5.0 |
| F6 | 106 | 39.6 % | 39.6 % | +0.0 |
| F7 | 106 | 72.6 % | 67.9 % | -4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.2 %, with 50.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 2.1 % | -0.2 | [-0.5, +0.0] |
| R2 | 612 | 2.8 % | 2.6 % | -0.2 | [-1.0, +0.3] |
| R3 | 589 | 5.1 % | 4.8 % | -0.3 | [-1.4, +0.7] |
| R4 | 493 | 4.7 % | 4.5 % | -0.2 | [-1.0, +0.6] |
| R5 | 548 | 0.5 % | 5.1 % | +4.6 | [+2.7, +6.6] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-11.6, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.2 % | +0.6 | [-1.9, +3.7] |
| threat faceswap | 272 | 5.9 % | 20.2 % | +14.3 | [+7.4, +24.7] |
| threat gan_face | 153 | 14.4 % | 17.6 % | +3.3 | [+0.7, +6.5] |
| threat personalised | 115 | 91.3 % | 93.9 % | +2.6 | [+0.0, +6.1] |
| threat t2i | 1600 | 65.4 % | 66.2 % | +0.8 | [-0.1, +1.7] |
| family faceswap-stills | 272 | 5.9 % | 20.2 % | +14.3 | [+7.4, +24.7] |
| family flux1 | 85 | 62.4 % | 65.9 % | +3.5 | [+0.0, +8.3] |
| family flux2 | 70 | 38.6 % | 40.0 % | +1.4 | [-3.0, +5.9] |
| family gan-faces | 153 | 14.4 % | 17.6 % | +3.3 | [+0.7, +6.5] |
| family gpt-4o-image-1 | 41 | 31.7 % | 34.1 % | +2.4 | [-4.9, +12.2] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 88.0 % | 86.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 72.3 % | +2.1 | [+0.0, +6.7] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 65.1 % | 67.9 % | +2.8 | [-1.0, +7.3] |
| family legacy | 145 | 90.3 % | 89.0 % | -1.4 | [-4.2, +1.3] |
| family long-tail | 186 | 67.2 % | 69.4 % | +2.2 | [-0.5, +5.4] |
| family midjourney | 153 | 82.4 % | 81.7 % | -0.7 | [-3.2, +1.3] |
| family modern-t2i-faces | 115 | 73.0 % | 74.8 % | +1.7 | [-1.7, +5.3] |
| family nano-banana | 118 | 44.9 % | 42.4 % | -2.5 | [-7.6, +2.5] |
| family partial-edits | 161 | 64.6 % | 65.2 % | +0.6 | [-1.9, +3.7] |
| family personalised | 115 | 91.3 % | 93.9 % | +2.6 | [+0.0, +6.1] |
| family qwen-image | 106 | 34.9 % | 38.7 % | +3.8 | [+0.9, +7.7] |
| family recraft | 54 | 81.5 % | 83.3 % | +1.9 | [-3.8, +7.7] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 77.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 57.5 % | 56.2 % | -1.2 | [-3.8, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +7.8 pts, CI [+4.8, +10.6] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.8 pts, CI [-2.2, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 85.0 % | +2.6 |
| F2 | 70 | 31.4 % | 42.9 % | +11.4 |
| F3 | 57 | 15.8 % | 22.8 % | +7.0 |
| F4 | 118 | 44.1 % | 52.5 % | +8.5 |
| F5 | 80 | 61.3 % | 63.7 % | +2.5 |
| F6 | 106 | 27.4 % | 39.6 % | +12.3 |
| F7 | 106 | 62.3 % | 72.6 % | +10.4 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 46.4 %, with 54.2 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.7 % | 2.4 % | +0.7 | [-0.4, +1.7] |
| R2 | 612 | 7.8 % | 2.8 % | -5.1 | [-8.0, -0.3] |
| R3 | 589 | 4.2 % | 5.1 % | +0.8 | [-0.7, +2.4] |
| R4 | 493 | 4.3 % | 4.7 % | +0.4 | [-1.0, +1.8] |
| R5 | 548 | 0.9 % | 0.5 % | -0.4 | [-1.3, +0.4] |
| R6 | 28 | 10.7 % | 14.3 % | +3.6 | [-8.3, +16.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 64.6 % | -1.2 | [-4.3, +1.9] |
| threat faceswap | 272 | 8.1 % | 5.9 % | -2.2 | [-4.3, -0.6] |
| threat gan_face | 153 | 17.6 % | 14.4 % | -3.3 | [-6.5, +0.0] |
| threat personalised | 115 | 95.7 % | 91.3 % | -4.3 | [-8.7, +0.0] |
| threat t2i | 1600 | 61.8 % | 65.4 % | +3.6 | [+2.2, +5.1] |
| family faceswap-stills | 272 | 8.1 % | 5.9 % | -2.2 | [-4.3, -0.6] |
| family flux1 | 85 | 61.2 % | 62.4 % | +1.2 | [-2.4, +5.7] |
| family flux2 | 70 | 30.0 % | 38.6 % | +8.6 | [-1.4, +18.3] |
| family gan-faces | 153 | 17.6 % | 14.4 % | -3.3 | [-6.5, +0.0] |
| family gpt-4o-image-1 | 41 | 31.7 % | 31.7 % | +0.0 | [-12.2, +14.6] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [-5.3, +5.3] |
| family hidream | 50 | 82.0 % | 88.0 % | +6.0 | [-2.1, +14.8] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 72.3 % | 70.2 % | -2.1 | [-9.4, +4.5] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [-9.4, +9.4] |
| family imagen4 | 106 | 56.6 % | 65.1 % | +8.5 | [+3.8, +13.6] |
| family legacy | 145 | 82.8 % | 90.3 % | +7.6 | [+2.8, +12.7] |
| family long-tail | 186 | 63.4 % | 67.2 % | +3.8 | [-1.1, +8.2] |
| family midjourney | 153 | 80.4 % | 82.4 % | +2.0 | [-1.3, +5.5] |
| family modern-t2i-faces | 115 | 68.7 % | 73.0 % | +4.3 | [+1.5, +8.2] |
| family nano-banana | 118 | 42.4 % | 44.9 % | +2.5 | [-5.1, +10.3] |
| family partial-edits | 161 | 65.8 % | 64.6 % | -1.2 | [-4.3, +1.9] |
| family personalised | 115 | 95.7 % | 91.3 % | -4.3 | [-8.7, +0.0] |
| family qwen-image | 106 | 26.4 % | 34.9 % | +8.5 | [+2.8, +15.1] |
| family recraft | 54 | 81.5 % | 81.5 % | +0.0 | [-7.1, +7.4] |
| family sd3 | 75 | 97.3 % | 98.7 % | +1.3 | [+0.0, +4.1] |
| family seedream3 | 35 | 74.3 % | 77.1 % | +2.9 | [-5.9, +11.4] |
| family seedream4 | 80 | 57.5 % | 57.5 % | +0.0 | [-6.2, +6.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

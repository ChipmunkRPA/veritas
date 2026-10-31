# Ablation: pe.copymove.pairs

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq,pe.copymove.pairs CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=pe.copymove.pairs ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4993 (2692 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.1 pts, CI [-0.4, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.2 pts, CI [-0.0, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 83.7 % | +0.0 |
| F2 | 70 | 37.1 % | 37.1 % | +0.0 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 48.3 % | 48.3 % | +0.0 |
| F5 | 80 | 76.2 % | 76.2 % | +0.0 |
| F6 | 106 | 40.6 % | 39.6 % | -0.9 |
| F7 | 106 | 66.0 % | 66.0 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 52.5 %, with 52.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.6 % | -0.2 | [-1.0, +0.0] |
| R2 | 611 | 3.6 % | 3.9 % | +0.3 | [+0.0, +0.9] |
| R3 | 588 | 4.4 % | 4.3 % | -0.2 | [-0.5, +0.0] |
| R4 | 493 | 3.7 % | 3.9 % | +0.2 | [+0.0, +0.6] |
| R5 | 548 | 0.9 % | 1.8 % | +0.9 | [+0.0, +1.8] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 15.8 % | 18.4 % | +2.6 | [+0.8, +4.9] |
| threat gan_face | 153 | 43.1 % | 43.8 % | +0.7 | [+0.0, +2.0] |
| threat personalised | 115 | 97.4 % | 96.5 % | -0.9 | [-2.6, +0.0] |
| threat t2i | 1600 | 66.6 % | 67.1 % | +0.5 | [+0.2, +0.9] |
| family faceswap-stills | 272 | 15.8 % | 18.4 % | +2.6 | [+0.8, +4.9] |
| family flux1 | 85 | 64.7 % | 64.7 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 32.9 % | 34.3 % | +1.4 | [+0.0, +4.4] |
| family gan-faces | 153 | 43.1 % | 43.8 % | +0.7 | [+0.0, +2.0] |
| family gpt-4o-image-1 | 41 | 14.6 % | 14.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 86.0 % | 88.0 % | +2.0 | [+0.0, +6.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 54.7 % | 54.7 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 63.2 % | 63.2 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 94.5 % | 94.5 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 68.8 % | 69.4 % | +0.5 | [+0.0, +1.8] |
| family midjourney | 153 | 83.0 % | 83.0 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 71.3 % | 71.3 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 43.2 % | 44.9 % | +1.7 | [+0.0, +4.3] |
| family partial-edits | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 97.4 % | 96.5 % | -0.9 | [-2.6, +0.0] |
| family qwen-image | 106 | 38.7 % | 38.7 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 85.2 % | 88.9 % | +3.7 | [+0.0, +9.6] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 73.8 % | 75.0 % | +1.2 | [+0.0, +3.7] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

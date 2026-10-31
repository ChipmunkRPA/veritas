# Ablation: pe.copymove.pairs

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,pe.copymove.pairs GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=pe.copymove.pairs ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.1 pts, CI [+0.0, +0.4] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.1, +0.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 98.7 % | 98.7 % | +0.0 |
| F2 | 70 | 58.6 % | 58.6 % | +0.0 |
| F3 | 57 | 52.6 % | 52.6 % | +0.0 |
| F4 | 118 | 83.9 % | 84.7 % | +0.8 |
| F5 | 80 | 75.0 % | 75.0 % | +0.0 |
| F6 | 106 | 84.9 % | 84.9 % | +0.0 |
| F7 | 106 | 94.3 % | 94.3 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 78.3 %, with 78.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 9.7 % | 9.7 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 0.3 % | 0.3 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 14.8 % | 14.9 % | +0.2 | [+0.0, +0.5] |
| R4 | 495 | 1.2 % | 1.0 % | -0.2 | [-0.6, +0.0] |
| R5 | 549 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 84.5 % | 84.5 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 64.7 % | 64.7 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 97.4 % | 97.4 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1603 | 89.3 % | 89.5 % | +0.2 | [+0.1, +0.5] |
| family faceswap-stills | 272 | 82.0 % | 82.0 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 87.1 % | 87.1 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 60.0 % | 60.0 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 64.7 % | 64.7 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 78.0 % | 78.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 57.9 % | 59.6 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 97.9 % | 97.9 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 97.2 % | 97.2 % | +0.0 | [+0.0, +0.0] |
| family legacy | 148 | 85.1 % | 85.1 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 88.7 % | 89.2 % | +0.5 | [+0.0, +1.8] |
| family midjourney | 153 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 94.8 % | 94.8 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 86.4 % | 87.3 % | +0.8 | [+0.0, +2.6] |
| family partial-edits | 161 | 84.5 % | 84.5 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 97.4 % | 97.4 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 85.8 % | 86.8 % | +0.9 | [+0.0, +2.8] |
| family recraft | 54 | 98.1 % | 98.1 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 97.1 % | 97.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 85.0 % | 85.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

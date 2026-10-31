# Ablation: img.forensic.ela

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.forensic.ela CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.ela ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.7 pts, CI [-1.3, -0.1] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.3 pts, CI [+0.1, +0.6] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 85.0 % | +0.0 |
| F2 | 70 | 42.9 % | 41.4 % | -1.4 |
| F3 | 57 | 22.8 % | 22.8 % | +0.0 |
| F4 | 118 | 52.5 % | 52.5 % | +0.0 |
| F5 | 80 | 63.7 % | 62.5 % | -1.2 |
| F6 | 106 | 39.6 % | 38.7 % | -0.9 |
| F7 | 106 | 72.6 % | 71.7 % | -0.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.2 %, with 53.5 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 2.6 % | +0.2 | [+0.0, +0.5] |
| R2 | 612 | 2.8 % | 2.6 % | -0.2 | [-0.7, +0.0] |
| R3 | 589 | 5.1 % | 5.6 % | +0.5 | [+0.0, +1.2] |
| R4 | 493 | 4.7 % | 4.9 % | +0.2 | [+0.0, +0.6] |
| R5 | 548 | 0.5 % | 1.5 % | +0.9 | [+0.2, +1.8] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 66.5 % | +1.9 | [+0.0, +4.3] |
| threat faceswap | 272 | 5.9 % | 8.8 % | +2.9 | [+1.1, +5.4] |
| threat gan_face | 153 | 14.4 % | 15.7 % | +1.3 | [+0.0, +3.3] |
| threat personalised | 115 | 91.3 % | 91.3 % | +0.0 | [-2.6, +2.6] |
| threat t2i | 1600 | 65.4 % | 66.4 % | +1.0 | [+0.5, +1.5] |
| family faceswap-stills | 272 | 5.9 % | 8.8 % | +2.9 | [+1.1, +5.4] |
| family flux1 | 85 | 62.4 % | 63.5 % | +1.2 | [+0.0, +3.7] |
| family flux2 | 70 | 38.6 % | 38.6 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 14.4 % | 15.7 % | +1.3 | [+0.0, +3.3] |
| family gpt-4o-image-1 | 41 | 31.7 % | 34.1 % | +2.4 | [+0.0, +7.3] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 88.0 % | 88.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 70.2 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 65.1 % | 65.1 % | +0.0 | [-2.8, +2.8] |
| family legacy | 145 | 90.3 % | 91.7 % | +1.4 | [+0.0, +3.4] |
| family long-tail | 186 | 67.2 % | 68.8 % | +1.6 | [+0.0, +3.7] |
| family midjourney | 153 | 82.4 % | 82.4 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 73.0 % | 74.8 % | +1.7 | [+0.0, +4.6] |
| family nano-banana | 118 | 44.9 % | 47.5 % | +2.5 | [+0.0, +5.9] |
| family partial-edits | 161 | 64.6 % | 66.5 % | +1.9 | [+0.0, +4.3] |
| family personalised | 115 | 91.3 % | 91.3 % | +0.0 | [-2.6, +2.6] |
| family qwen-image | 106 | 34.9 % | 36.8 % | +1.9 | [+0.0, +4.8] |
| family recraft | 54 | 81.5 % | 83.3 % | +1.9 | [+0.0, +5.8] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 77.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 57.5 % | 57.5 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

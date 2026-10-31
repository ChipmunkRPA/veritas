# Ablation: img.forensic.jpeg_dq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.hf.ateeqq,img.forensic.jpeg_dq CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.jpeg_dq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.3 pts, CI [+0.0, +0.8] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.2, +0.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 83.7 % | +0.0 |
| F2 | 70 | 37.1 % | 37.1 % | +0.0 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 48.3 % | 48.3 % | +0.0 |
| F5 | 80 | 76.2 % | 77.5 % | +1.3 |
| F6 | 106 | 39.6 % | 39.6 % | +0.0 |
| F7 | 106 | 66.0 % | 67.0 % | +0.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 52.4 %, with 52.7 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.2 % | 4.0 % | -0.2 | [-1.0, +0.0] |
| R2 | 612 | 3.8 % | 3.9 % | +0.2 | [+0.0, +0.3] |
| R3 | 589 | 4.6 % | 4.4 % | -0.2 | [-0.5, +0.0] |
| R4 | 493 | 4.5 % | 4.5 % | +0.0 | [+0.0, +0.0] |
| R5 | 548 | 2.0 % | 2.2 % | +0.2 | [+0.0, +0.5] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.2 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 20.6 % | 21.0 % | +0.4 | [+0.0, +1.3] |
| threat gan_face | 153 | 43.8 % | 44.4 % | +0.7 | [+0.0, +2.0] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 67.6 % | 67.8 % | +0.2 | [+0.0, +0.4] |
| family faceswap-stills | 272 | 20.6 % | 21.0 % | +0.4 | [+0.0, +1.3] |
| family flux1 | 85 | 67.1 % | 67.1 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 37.1 % | 37.1 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 43.8 % | 44.4 % | +0.7 | [+0.0, +2.0] |
| family gpt-4o-image-1 | 41 | 22.0 % | 22.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 17.5 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 86.0 % | 86.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 65.1 % | 65.1 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 95.2 % | 95.2 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 69.9 % | 69.9 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 83.7 % | 84.3 % | +0.7 | [+0.0, +2.1] |
| family modern-t2i-faces | 115 | 72.2 % | 72.2 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 44.1 % | 44.9 % | +0.8 | [+0.0, +2.6] |
| family partial-edits | 161 | 65.2 % | 65.2 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 37.7 % | 37.7 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 85.2 % | 87.0 % | +1.9 | [+0.0, +5.8] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

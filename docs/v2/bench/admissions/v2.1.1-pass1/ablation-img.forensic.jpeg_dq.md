# Ablation: img.forensic.jpeg_dq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.forensic.jpeg_dq GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.jpeg_dq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.1 pts, CI [+0.0, +0.3] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 82.4 % | 82.4 % | +0.0 |
| F2 | 70 | 27.1 % | 27.1 % | +0.0 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 38.1 % | 38.1 % | +0.0 |
| F5 | 80 | 75.0 % | 75.0 % | +0.0 |
| F6 | 106 | 28.3 % | 28.3 % | +0.0 |
| F7 | 106 | 50.9 % | 50.9 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.1 %, with 45.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.9 % | 1.9 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 5.6 % | 5.6 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 3.9 % | 4.1 % | +0.2 | [+0.0, +0.5] |
| R4 | 493 | 8.9 % | 8.9 % | +0.0 | [+0.0, +0.0] |
| R5 | 548 | 1.3 % | 1.6 % | +0.4 | [+0.0, +0.9] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 59.2 % | 59.2 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 51.6 % | 51.6 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 61.6 % | 61.7 % | +0.1 | [+0.0, +0.2] |
| family faceswap-stills | 272 | 59.2 % | 59.2 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 51.6 % | 51.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 17.1 % | 17.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 49.1 % | 49.1 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 86.9 % | 87.6 % | +0.7 | [+0.0, +2.2] |
| family long-tail | 186 | 65.1 % | 65.1 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 82.4 % | 82.4 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 54.8 % | 54.8 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 37.3 % | 37.3 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 28.3 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 77.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

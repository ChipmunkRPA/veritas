# Ablation: img.meta.generator

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.meta.generator GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.meta.generator ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [+0.0, +0.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 81.0 % | +0.0 |
| F2 | 70 | 25.7 % | 25.7 % | +0.0 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 33.1 % | 33.1 % | +0.0 |
| F5 | 80 | 75.0 % | 75.0 % | +0.0 |
| F6 | 106 | 29.2 % | 29.2 % | +0.0 |
| F7 | 106 | 50.9 % | 50.9 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 44.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 1.4 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 4.7 % | 4.7 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 3.7 % | 3.7 % | +0.0 | [+0.0, +0.0] |
| R4 | 493 | 15.0 % | 15.0 % | +0.0 | [+0.0, +0.0] |
| R5 | 548 | 1.5 % | 1.5 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 63.4 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 62.9 % | 62.9 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 53.6 % | 53.6 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 61.1 % | 61.1 % | +0.0 | [+0.0, +0.0] |
| family faceswap-stills | 272 | 62.9 % | 62.9 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 27.1 % | 27.1 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 53.6 % | 53.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 7.3 % | 7.3 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 41.5 % | 41.5 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 50.9 % | 50.9 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 96.6 % | 96.6 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 63.4 % | 63.4 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 54.8 % | 54.8 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 32.2 % | 32.2 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 63.4 % | 63.4 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 29.2 % | 29.2 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 87.0 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 97.3 % | 97.3 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 74.3 % | 74.3 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 75.0 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

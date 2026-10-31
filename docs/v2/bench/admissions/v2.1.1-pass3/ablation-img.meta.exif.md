# Ablation: img.meta.exif

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.ateeqq,img.dfdc_b7,img.meta.exif GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.meta.exif ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.0 pts, CI [+0.0, +0.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [+0.0, +0.0] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 85.0 % | +0.0 |
| F2 | 70 | 42.9 % | 42.9 % | +0.0 |
| F3 | 57 | 19.3 % | 19.3 % | +0.0 |
| F4 | 118 | 51.7 % | 51.7 % | +0.0 |
| F5 | 80 | 63.7 % | 63.7 % | +0.0 |
| F6 | 106 | 40.6 % | 40.6 % | +0.0 |
| F7 | 106 | 70.8 % | 70.8 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.4 %, with 53.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 2.4 % | +0.0 | [+0.0, +0.0] |
| R2 | 612 | 2.5 % | 2.5 % | +0.0 | [+0.0, +0.0] |
| R3 | 589 | 4.9 % | 4.9 % | +0.0 | [+0.0, +0.0] |
| R4 | 493 | 4.5 % | 4.5 % | +0.0 | [+0.0, +0.0] |
| R5 | 548 | 1.3 % | 1.3 % | +0.0 | [+0.0, +0.0] |
| R6 | 28 | 14.3 % | 14.3 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 29.8 % | 29.8 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 9.8 % | 9.8 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 91.3 % | 91.3 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 63.7 % | 63.7 % | +0.0 | [+0.0, +0.0] |
| family faceswap-stills | 272 | 29.8 % | 29.8 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 62.4 % | 62.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 38.6 % | 38.6 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 9.8 % | 9.8 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 31.7 % | 31.7 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 15.8 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 84.0 % | 84.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 70.2 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 50.9 % | 50.9 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 60.4 % | 60.4 % | +0.0 | [+0.0, +0.0] |
| family legacy | 145 | 91.7 % | 91.7 % | +0.0 | [+0.0, +0.0] |
| family long-tail | 186 | 67.7 % | 67.7 % | +0.0 | [+0.0, +0.0] |
| family midjourney | 153 | 79.7 % | 79.7 % | +0.0 | [+0.0, +0.0] |
| family modern-t2i-faces | 115 | 65.2 % | 65.2 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 40.7 % | 40.7 % | +0.0 | [+0.0, +0.0] |
| family partial-edits | 161 | 64.6 % | 64.6 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 91.3 % | 91.3 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 33.0 % | 33.0 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 79.6 % | 79.6 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 77.1 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 56.2 % | 56.2 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

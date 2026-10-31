# Ablation: pe.mesorch.integrity

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq,pe.mesorch.integrity GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=pe.mesorch.integrity ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -3.2 pts, CI [-4.8, -1.7] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.5, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 80.4 % | -2.6 |
| F2 | 70 | 44.3 % | 41.4 % | -2.9 |
| F3 | 57 | 21.1 % | 19.3 % | -1.8 |
| F4 | 118 | 50.8 % | 44.9 % | -5.9 |
| F5 | 80 | 68.8 % | 65.0 % | -3.7 |
| F6 | 106 | 40.6 % | 38.7 % | -1.9 |
| F7 | 106 | 67.9 % | 64.2 % | -3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.8 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 4.2 % | -0.2 | [-1.1, +0.7] |
| R2 | 612 | 3.9 % | 3.6 % | -0.3 | [-1.4, +0.2] |
| R3 | 589 | 5.8 % | 5.3 % | -0.5 | [-1.3, +0.2] |
| R4 | 493 | 5.9 % | 5.5 % | -0.4 | [-1.6, +0.6] |
| R5 | 548 | 4.2 % | 5.8 % | +1.6 | [+0.2, +3.1] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-11.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.8 % | +0.6 | [-1.2, +3.1] |
| threat faceswap | 272 | 71.3 % | 71.0 % | -0.4 | [-2.7, +1.8] |
| threat gan_face | 153 | 54.2 % | 50.3 % | -3.9 | [-7.8, -0.7] |
| threat personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 65.1 % | 62.7 % | -2.3 | [-3.1, -1.5] |
| family faceswap-stills | 272 | 71.3 % | 71.0 % | -0.4 | [-2.7, +1.8] |
| family flux1 | 85 | 62.4 % | 60.0 % | -2.4 | [-6.0, +0.0] |
| family flux2 | 70 | 40.0 % | 35.7 % | -4.3 | [-9.6, +0.0] |
| family gan-faces | 153 | 54.2 % | 50.3 % | -3.9 | [-7.8, -0.7] |
| family gpt-4o-image-1 | 41 | 36.6 % | 36.6 % | +0.0 | [-9.8, +9.8] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 17.5 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 84.0 % | 82.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 72.3 % | -4.3 | [-10.9, +0.0] |
| family imagen3 | 53 | 49.1 % | 49.1 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 62.3 % | 59.4 % | -2.8 | [-6.1, +0.0] |
| family legacy | 145 | 91.0 % | 89.7 % | -1.4 | [-3.4, +0.0] |
| family long-tail | 186 | 71.0 % | 68.8 % | -2.2 | [-4.4, -0.5] |
| family midjourney | 153 | 80.4 % | 77.1 % | -3.3 | [-6.5, -0.7] |
| family modern-t2i-faces | 115 | 66.1 % | 60.9 % | -5.2 | [-8.5, -1.8] |
| family nano-banana | 118 | 42.4 % | 39.0 % | -3.4 | [-7.6, +0.8] |
| family partial-edits | 161 | 65.2 % | 65.8 % | +0.6 | [-1.2, +3.1] |
| family personalised | 115 | 96.5 % | 93.9 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 33.0 % | 30.2 % | -2.8 | [-6.5, +0.0] |
| family recraft | 54 | 83.3 % | 79.6 % | -3.7 | [-9.6, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 77.1 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 57.5 % | 58.8 % | +1.3 | [+0.0, +4.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq GATES='face.gend_clip_l14=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.4 pts, CI [-1.2, +2.2] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.4, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.0 % | 83.0 % | -2.0 |
| F2 | 70 | 42.9 % | 44.3 % | +1.4 |
| F3 | 57 | 21.1 % | 21.1 % | +0.0 |
| F4 | 118 | 52.5 % | 50.8 % | -1.7 |
| F5 | 80 | 66.2 % | 68.8 % | +2.5 |
| F6 | 106 | 37.7 % | 40.6 % | +2.8 |
| F7 | 106 | 67.9 % | 67.9 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 53.3 %, with 53.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.7 % | 4.5 % | -0.2 | [-1.0, +0.6] |
| R2 | 612 | 3.1 % | 3.9 % | +0.8 | [-0.3, +1.6] |
| R3 | 589 | 6.6 % | 5.8 % | -0.8 | [-2.1, +0.2] |
| R4 | 493 | 6.3 % | 5.9 % | -0.4 | [-1.2, +0.2] |
| R5 | 548 | 3.5 % | 4.2 % | +0.7 | [+0.0, +1.6] |
| R6 | 28 | 17.9 % | 17.9 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 65.2 % | -1.2 | [-3.1, +0.0] |
| threat faceswap | 272 | 69.5 % | 71.3 % | +1.8 | [+0.4, +3.9] |
| threat gan_face | 153 | 49.7 % | 54.2 % | +4.6 | [+1.3, +8.5] |
| threat personalised | 115 | 94.8 % | 96.5 % | +1.7 | [+0.0, +4.3] |
| threat t2i | 1600 | 65.0 % | 65.1 % | +0.1 | [-0.8, +0.9] |
| family faceswap-stills | 272 | 69.5 % | 71.3 % | +1.8 | [+0.4, +3.9] |
| family flux1 | 85 | 62.4 % | 62.4 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 38.6 % | 40.0 % | +1.4 | [-4.5, +7.2] |
| family gan-faces | 153 | 49.7 % | 54.2 % | +4.6 | [+1.3, +8.5] |
| family gpt-4o-image-1 | 41 | 41.5 % | 36.6 % | -4.9 | [-14.6, +4.9] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 17.5 % | +1.8 | [-3.5, +7.0] |
| family hidream | 50 | 86.0 % | 84.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 76.6 % | +6.4 | [+0.0, +14.9] |
| family imagen3 | 53 | 50.9 % | 49.1 % | -1.9 | [-5.7, +0.0] |
| family imagen4 | 106 | 60.4 % | 62.3 % | +1.9 | [+0.0, +4.7] |
| family legacy | 145 | 91.7 % | 91.0 % | -0.7 | [-2.2, +0.0] |
| family long-tail | 186 | 68.8 % | 71.0 % | +2.2 | [+0.0, +4.8] |
| family midjourney | 153 | 81.0 % | 80.4 % | -0.7 | [-3.2, +1.3] |
| family modern-t2i-faces | 115 | 66.1 % | 66.1 % | +0.0 | [-2.6, +2.6] |
| family nano-banana | 118 | 47.5 % | 42.4 % | -5.1 | [-9.4, -0.8] |
| family partial-edits | 161 | 66.5 % | 65.2 % | -1.2 | [-3.1, +0.0] |
| family personalised | 115 | 94.8 % | 96.5 % | +1.7 | [+0.0, +4.3] |
| family qwen-image | 106 | 32.1 % | 33.0 % | +0.9 | [+0.0, +2.9] |
| family recraft | 54 | 81.5 % | 83.3 % | +1.9 | [-3.8, +8.9] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 80.0 % | +2.9 | [+0.0, +8.8] |
| family seedream4 | 80 | 58.8 % | 57.5 % | -1.3 | [-5.9, +2.6] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

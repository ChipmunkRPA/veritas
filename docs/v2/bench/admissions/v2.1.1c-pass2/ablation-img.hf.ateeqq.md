# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.hf.dima806,face.gend_clip_l14,img.hf.ateeqq GATES='face.gend_clip_l14=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.6 pts, CI [+0.0, +5.1] |
| A3b | pass | family imagen4 |
| A4 | fail | pooled reals: Δ FPR -0.1 pts, CI [-1.3, +1.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R5; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 80.4 % | -0.7 |
| F2 | 70 | 31.4 % | 40.0 % | +8.6 |
| F3 | 57 | 14.0 % | 15.8 % | +1.8 |
| F4 | 118 | 43.2 % | 39.8 % | -3.4 |
| F5 | 80 | 63.7 % | 57.5 % | -6.2 |
| F6 | 106 | 28.3 % | 37.7 % | +9.4 |
| F7 | 106 | 56.6 % | 65.1 % | +8.5 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.5 %, with 48.0 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.0 % | 4.5 % | +0.5 | [-0.6, +1.1] |
| R2 | 612 | 6.5 % | 2.3 % | -4.2 | [-6.1, -0.8] |
| R3 | 589 | 4.4 % | 5.3 % | +0.8 | [-0.7, +2.3] |
| R4 | 493 | 4.1 % | 4.9 % | +0.8 | [-0.6, +2.2] |
| R5 | 548 | 5.5 % | 7.7 % | +2.2 | [-0.4, +4.4] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [-10.7, +11.5] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 63.4 % | -1.2 | [-5.0, +2.5] |
| threat faceswap | 272 | 73.5 % | 66.5 % | -7.0 | [-11.6, -2.5] |
| threat gan_face | 153 | 61.4 % | 41.2 % | -20.3 | [-27.5, -14.4] |
| threat personalised | 115 | 95.7 % | 92.2 % | -3.5 | [-7.0, -0.9] |
| threat t2i | 1600 | 60.6 % | 64.6 % | +4.0 | [+2.5, +5.5] |
| family faceswap-stills | 272 | 73.5 % | 66.5 % | -7.0 | [-11.6, -2.5] |
| family flux1 | 85 | 61.2 % | 63.5 % | +2.4 | [-1.2, +7.2] |
| family flux2 | 70 | 28.6 % | 38.6 % | +10.0 | [+0.0, +20.8] |
| family gan-faces | 153 | 61.4 % | 41.2 % | -20.3 | [-27.5, -14.4] |
| family gpt-4o-image-1 | 41 | 31.7 % | 31.7 % | +0.0 | [-12.2, +12.2] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 76.0 % | 86.0 % | +10.0 | [+2.1, +18.9] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 72.3 % | +2.1 | [-6.7, +12.2] |
| family imagen3 | 53 | 50.9 % | 50.9 % | +0.0 | [-9.4, +9.4] |
| family imagen4 | 106 | 50.9 % | 63.2 % | +12.3 | [+6.6, +18.6] |
| family legacy | 145 | 80.0 % | 89.0 % | +9.0 | [+3.8, +14.4] |
| family long-tail | 186 | 64.0 % | 69.9 % | +5.9 | [+1.5, +10.2] |
| family midjourney | 153 | 81.0 % | 80.4 % | -0.7 | [-3.9, +2.6] |
| family modern-t2i-faces | 115 | 69.6 % | 73.0 % | +3.5 | [+0.8, +7.3] |
| family nano-banana | 118 | 39.0 % | 37.3 % | -1.7 | [-8.5, +4.3] |
| family partial-edits | 161 | 64.6 % | 63.4 % | -1.2 | [-5.0, +2.5] |
| family personalised | 115 | 95.7 % | 92.2 % | -3.5 | [-7.0, -0.9] |
| family qwen-image | 106 | 26.4 % | 34.9 % | +8.5 | [+1.9, +15.7] |
| family recraft | 54 | 77.8 % | 77.8 % | +0.0 | [-7.7, +7.1] |
| family sd3 | 75 | 96.0 % | 98.7 % | +2.7 | [+0.0, +6.4] |
| family seedream3 | 35 | 74.3 % | 77.1 % | +2.9 | [-5.9, +11.4] |
| family seedream4 | 80 | 58.8 % | 57.5 % | -1.3 | [-8.8, +6.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

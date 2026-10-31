# Ablation: img.forensic.noise

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.forensic.noise GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.noise ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -4.4 pts, CI [-6.1, -3.0] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.6, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 28.1 % | -9.2 |
| F2 | 70 | 42.9 % | 38.6 % | -4.3 |
| F3 | 57 | 40.4 % | 38.6 % | -1.8 |
| F4 | 118 | 49.2 % | 45.8 % | -3.4 |
| F5 | 80 | 40.0 % | 35.0 % | -5.0 |
| F6 | 106 | 34.9 % | 30.2 % | -4.7 |
| F7 | 106 | 32.1 % | 29.2 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 35.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 3.5 % | -0.9 | [-2.1, +0.0] |
| R2 | 612 | 4.2 % | 3.1 % | -1.1 | [-2.7, -0.4] |
| R3 | 589 | 10.0 % | 10.2 % | +0.2 | [-1.1, +1.4] |
| R4 | 493 | 4.9 % | 6.7 % | +1.8 | [+0.4, +3.2] |
| R5 | 548 | 1.5 % | 1.5 % | +0.0 | [-0.5, +0.5] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [-10.7, +10.7] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.8 % | 15.5 % | -1.2 | [-3.7, +1.2] |
| threat faceswap | 272 | 0.4 % | 0.7 % | +0.4 | [+0.0, +1.2] |
| threat gan_face | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 36.4 % | 34.4 % | -1.9 | [-2.9, -1.0] |
| family faceswap-stills | 272 | 0.4 % | 0.7 % | +0.4 | [+0.0, +1.2] |
| family flux1 | 85 | 58.8 % | 52.9 % | -5.9 | [-11.8, -1.2] |
| family flux2 | 70 | 42.9 % | 42.9 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 41.5 % | 34.1 % | -7.3 | [-17.1, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 38.6 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 62.0 % | 62.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 62.5 % | 58.3 % | -4.2 | [-12.5, +0.0] |
| family ideogram | 47 | 27.7 % | 23.4 % | -4.3 | [-10.9, +0.0] |
| family imagen3 | 53 | 18.9 % | 17.0 % | -1.9 | [-5.7, +0.0] |
| family imagen4 | 106 | 31.1 % | 29.2 % | -1.9 | [-4.9, +0.0] |
| family legacy | 145 | 39.3 % | 40.7 % | +1.4 | [-3.9, +6.7] |
| family long-tail | 186 | 40.3 % | 41.4 % | +1.1 | [-1.2, +3.9] |
| family midjourney | 153 | 35.9 % | 30.7 % | -5.2 | [-9.0, -2.0] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 48.3 % | 45.8 % | -2.5 | [-5.9, +0.0] |
| family partial-edits | 161 | 16.8 % | 15.5 % | -1.2 | [-3.7, +1.2] |
| family personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 33.0 % | 30.2 % | -2.8 | [-6.6, +0.0] |
| family recraft | 54 | 27.8 % | 25.9 % | -1.9 | [-5.8, +0.0] |
| family sd3 | 75 | 29.3 % | 26.7 % | -2.7 | [-6.9, +0.0] |
| family seedream3 | 35 | 28.6 % | 25.7 % | -2.9 | [-8.1, +0.0] |
| family seedream4 | 80 | 38.8 % | 36.2 % | -2.5 | [-6.3, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

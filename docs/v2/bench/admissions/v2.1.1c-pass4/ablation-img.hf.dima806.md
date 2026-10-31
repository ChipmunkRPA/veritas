# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.hf.dima806 GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 5001 (2697 real, 2304 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -2.6 pts, CI [-3.9, -1.4] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.6, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 35.9 % | 31.4 % | -4.6 |
| F2 | 70 | 42.9 % | 40.0 % | -2.9 |
| F3 | 57 | 40.4 % | 40.4 % | +0.0 |
| F4 | 118 | 49.2 % | 44.9 % | -4.2 |
| F5 | 80 | 40.0 % | 37.5 % | -2.5 |
| F6 | 106 | 34.9 % | 34.9 % | +0.0 |
| F7 | 106 | 32.1 % | 28.3 % | -3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.3 %, with 36.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.4 % | 5.0 % | -0.5 | [-1.9, +1.2] |
| R2 | 612 | 3.9 % | 3.9 % | +0.0 | [-0.8, +0.7] |
| R3 | 589 | 10.4 % | 10.0 % | -0.3 | [-1.7, +1.0] |
| R4 | 495 | 3.8 % | 4.8 % | +1.0 | [-0.2, +2.2] |
| R5 | 549 | 1.5 % | 1.1 % | -0.4 | [-1.1, +0.2] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 15.5 % | 14.3 % | -1.2 | [-3.1, +0.0] |
| threat faceswap | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1603 | 36.1 % | 34.7 % | -1.4 | [-2.2, -0.6] |
| family faceswap-stills | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 61.2 % | 60.0 % | -1.2 | [-3.6, +0.0] |
| family flux2 | 70 | 41.4 % | 40.0 % | -1.4 | [-4.4, +0.0] |
| family gan-faces | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 36.6 % | 36.6 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 38.6 % | -1.8 | [-7.0, +3.5] |
| family hidream | 50 | 60.0 % | 64.0 % | +4.0 | [+0.0, +10.2] |
| family hunyuanimage | 24 | 62.5 % | 62.5 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 27.7 % | 25.5 % | -2.1 | [-6.7, +0.0] |
| family imagen3 | 53 | 18.9 % | 20.8 % | +1.9 | [+0.0, +5.7] |
| family imagen4 | 106 | 30.2 % | 29.2 % | -0.9 | [-3.0, +0.0] |
| family legacy | 148 | 37.8 % | 34.5 % | -3.4 | [-6.3, -0.7] |
| family long-tail | 186 | 40.3 % | 38.7 % | -1.6 | [-4.3, +0.0] |
| family midjourney | 153 | 34.6 % | 30.7 % | -3.9 | [-8.5, +0.0] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 47.5 % | 46.6 % | -0.8 | [-4.2, +1.7] |
| family partial-edits | 161 | 15.5 % | 14.3 % | -1.2 | [-3.1, +0.0] |
| family personalised | 115 | 9.6 % | 9.6 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 34.0 % | 34.0 % | +0.0 | [-2.8, +2.8] |
| family recraft | 54 | 27.8 % | 27.8 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 30.7 % | 26.7 % | -4.0 | [-9.3, +0.0] |
| family seedream3 | 35 | 28.6 % | 28.6 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 40.0 % | 37.5 % | -2.5 | [-6.3, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

# Ablation: img.forensic.ela

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.hf.ateeqq,img.forensic.ela GATES='img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.forensic.ela ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.7 pts, CI [-1.9, +0.5] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.7, +0.5] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 37.3 % | 36.6 % | -0.7 |
| F2 | 70 | 42.9 % | 40.0 % | -2.9 |
| F3 | 57 | 40.4 % | 42.1 % | +1.8 |
| F4 | 118 | 49.2 % | 48.3 % | -0.8 |
| F5 | 80 | 40.0 % | 38.8 % | -1.3 |
| F6 | 106 | 34.9 % | 34.0 % | -0.9 |
| F7 | 106 | 32.1 % | 32.1 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 39.5 %, with 38.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 5.7 % | +1.2 | [-1.0, +2.9] |
| R2 | 612 | 4.2 % | 2.1 % | -2.1 | [-4.7, -0.8] |
| R3 | 589 | 10.0 % | 11.5 % | +1.5 | [+0.3, +2.9] |
| R4 | 493 | 4.9 % | 4.3 % | -0.6 | [-1.8, +0.6] |
| R5 | 548 | 1.5 % | 1.5 % | +0.0 | [-0.7, +0.7] |
| R6 | 28 | 7.1 % | 7.1 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 16.8 % | 19.9 % | +3.1 | [+0.0, +6.8] |
| threat faceswap | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| threat gan_face | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| threat personalised | 115 | 9.6 % | 7.0 % | -2.6 | [-6.1, +0.0] |
| threat t2i | 1600 | 36.4 % | 37.4 % | +1.1 | [+0.0, +2.1] |
| family faceswap-stills | 272 | 0.4 % | 0.4 % | +0.0 | [+0.0, +0.0] |
| family flux1 | 85 | 58.8 % | 61.2 % | +2.4 | [-2.2, +7.1] |
| family flux2 | 70 | 42.9 % | 37.1 % | -5.7 | [-12.9, +0.0] |
| family gan-faces | 153 | 0.0 % | 0.0 % | +0.0 | [+0.0, +0.0] |
| family gpt-4o-image-1 | 41 | 41.5 % | 36.6 % | -4.9 | [-12.2, +0.0] |
| family gpt-image-1.5-2 | 57 | 40.4 % | 42.1 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 62.0 % | 62.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 62.5 % | 62.5 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 27.7 % | 34.0 % | +6.4 | [+0.0, +13.5] |
| family imagen3 | 53 | 18.9 % | 24.5 % | +5.7 | [+0.0, +11.3] |
| family imagen4 | 106 | 31.1 % | 31.1 % | +0.0 | [-3.8, +3.6] |
| family legacy | 145 | 39.3 % | 37.2 % | -2.1 | [-6.5, +2.2] |
| family long-tail | 186 | 40.3 % | 43.5 % | +3.2 | [+0.5, +6.5] |
| family midjourney | 153 | 35.9 % | 40.5 % | +4.6 | [+1.4, +7.9] |
| family modern-t2i-faces | 115 | 2.6 % | 2.6 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 48.3 % | 48.3 % | +0.0 | [-2.5, +2.5] |
| family partial-edits | 161 | 16.8 % | 19.9 % | +3.1 | [+0.0, +6.8] |
| family personalised | 115 | 9.6 % | 7.0 % | -2.6 | [-6.1, +0.0] |
| family qwen-image | 106 | 33.0 % | 34.0 % | +0.9 | [-1.9, +3.9] |
| family recraft | 54 | 27.8 % | 33.3 % | +5.6 | [+0.0, +12.5] |
| family sd3 | 75 | 29.3 % | 29.3 % | +0.0 | [-4.1, +3.9] |
| family seedream3 | 35 | 28.6 % | 31.4 % | +2.9 | [+0.0, +8.8] |
| family seedream4 | 80 | 38.8 % | 37.5 % | -1.3 | [-5.1, +2.5] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

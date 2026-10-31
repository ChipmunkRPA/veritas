# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.2 pts, CI [-0.4, +2.9] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.5 pts, CI [-1.1, +0.1] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.7 % | 83.0 % | +1.3 |
| F2 | 70 | 30.0 % | 32.9 % | +2.9 |
| F3 | 57 | 15.8 % | 15.8 % | +0.0 |
| F4 | 118 | 44.9 % | 44.9 % | +0.0 |
| F5 | 80 | 61.3 % | 63.7 % | +2.5 |
| F6 | 106 | 27.4 % | 31.1 % | +3.8 |
| F7 | 106 | 60.4 % | 58.5 % | -1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 45.9 %, with 47.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 2.1 % | -0.7 | [-1.5, +0.0] |
| R2 | 612 | 8.8 % | 7.8 % | -1.0 | [-2.0, -0.2] |
| R3 | 589 | 4.9 % | 4.8 % | -0.2 | [-0.9, +0.5] |
| R4 | 493 | 8.7 % | 7.7 % | -1.0 | [-3.2, +1.2] |
| R5 | 548 | 1.5 % | 2.0 % | +0.5 | [-0.4, +1.5] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 55.5 % | 59.6 % | +4.0 | [+1.2, +7.3] |
| threat gan_face | 153 | 25.5 % | 34.6 % | +9.2 | [+3.9, +15.0] |
| threat personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 63.8 % | 61.5 % | -2.3 | [-3.2, -1.5] |
| family faceswap-stills | 272 | 55.5 % | 59.6 % | +4.0 | [+1.2, +7.3] |
| family flux1 | 85 | 61.2 % | 61.2 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 30.0 % | 28.6 % | -1.4 | [-4.5, +0.0] |
| family gan-faces | 153 | 25.5 % | 34.6 % | +9.2 | [+3.9, +15.0] |
| family gpt-4o-image-1 | 41 | 34.1 % | 34.1 % | +0.0 | [+0.0, +0.0] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 14.0 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 82.0 % | 80.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 72.3 % | -4.3 | [-10.9, +0.0] |
| family imagen3 | 53 | 52.8 % | 52.8 % | +0.0 | [+0.0, +0.0] |
| family imagen4 | 106 | 59.4 % | 53.8 % | -5.7 | [-11.8, +0.0] |
| family legacy | 145 | 89.0 % | 82.1 % | -6.9 | [-11.6, -2.7] |
| family long-tail | 186 | 65.6 % | 63.4 % | -2.2 | [-4.8, +0.0] |
| family midjourney | 153 | 81.7 % | 80.4 % | -1.3 | [-4.0, +1.3] |
| family modern-t2i-faces | 115 | 66.1 % | 66.1 % | +0.0 | [-2.4, +2.6] |
| family nano-banana | 118 | 44.9 % | 42.4 % | -2.5 | [-5.9, +0.0] |
| family partial-edits | 161 | 66.5 % | 66.5 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 95.7 % | 95.7 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 28.3 % | 26.4 % | -1.9 | [-4.8, +0.0] |
| family recraft | 54 | 87.0 % | 83.3 % | -3.7 | [-9.6, +0.0] |
| family sd3 | 75 | 97.3 % | 96.0 % | -1.3 | [-3.9, +0.0] |
| family seedream3 | 35 | 82.9 % | 80.0 % | -2.9 | [-8.8, +0.0] |
| family seedream4 | 80 | 61.3 % | 60.0 % | -1.3 | [-3.8, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

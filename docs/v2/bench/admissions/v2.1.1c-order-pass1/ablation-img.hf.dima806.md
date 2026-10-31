# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +4.0 pts, CI [+2.4, +5.6] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.3 pts, CI [-1.0, +0.4] (upper bound ≤ +1.0); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 81.0 % | 84.3 % | +3.3 |
| F2 | 70 | 25.7 % | 32.9 % | +7.1 |
| F3 | 57 | 14.0 % | 15.8 % | +1.8 |
| F4 | 118 | 33.1 % | 37.3 % | +4.2 |
| F5 | 80 | 75.0 % | 78.8 % | +3.7 |
| F6 | 106 | 29.2 % | 34.0 % | +4.7 |
| F7 | 106 | 50.9 % | 53.8 % | +2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 44.1 %, with 48.1 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 1.4 % | 2.4 % | +0.9 | [-0.5, +2.1] |
| R2 | 612 | 4.7 % | 4.9 % | +0.2 | [+0.0, +0.5] |
| R3 | 589 | 3.7 % | 4.6 % | +0.8 | [-0.3, +2.1] |
| R4 | 493 | 15.0 % | 11.8 % | -3.2 | [-6.1, -0.4] |
| R5 | 548 | 1.5 % | 1.3 % | -0.2 | [-1.1, +0.9] |
| R6 | 28 | 3.6 % | 3.6 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 63.4 % | 64.6 % | +1.2 | [+0.0, +3.1] |
| threat faceswap | 272 | 62.9 % | 67.6 % | +4.8 | [+2.5, +7.8] |
| threat gan_face | 153 | 53.6 % | 60.1 % | +6.5 | [+2.6, +10.5] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 61.1 % | 61.4 % | +0.3 | [-0.6, +1.0] |
| family faceswap-stills | 272 | 62.9 % | 67.6 % | +4.8 | [+2.5, +7.8] |
| family flux1 | 85 | 61.2 % | 63.5 % | +2.4 | [+0.0, +6.1] |
| family flux2 | 70 | 27.1 % | 28.6 % | +1.4 | [+0.0, +4.5] |
| family gan-faces | 153 | 53.6 % | 60.1 % | +6.5 | [+2.6, +10.5] |
| family gpt-4o-image-1 | 41 | 7.3 % | 9.8 % | +2.4 | [+0.0, +7.3] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [+0.0, +0.0] |
| family hidream | 50 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 74.5 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 41.5 % | 49.1 % | +7.5 | [+1.9, +15.1] |
| family imagen4 | 106 | 50.9 % | 49.1 % | -1.9 | [-7.1, +2.9] |
| family legacy | 145 | 96.6 % | 90.3 % | -6.2 | [-10.2, -2.8] |
| family long-tail | 186 | 63.4 % | 62.9 % | -0.5 | [-2.6, +1.1] |
| family midjourney | 153 | 81.7 % | 81.7 % | +0.0 | [-1.9, +1.9] |
| family modern-t2i-faces | 115 | 54.8 % | 58.3 % | +3.5 | [+0.0, +7.5] |
| family nano-banana | 118 | 32.2 % | 32.2 % | +0.0 | [-4.2, +4.2] |
| family partial-edits | 161 | 63.4 % | 64.6 % | +1.2 | [+0.0, +3.1] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 29.2 % | 29.2 % | +0.0 | [+0.0, +0.0] |
| family recraft | 54 | 87.0 % | 88.9 % | +1.9 | [+0.0, +5.8] |
| family sd3 | 75 | 97.3 % | 96.0 % | -1.3 | [-3.9, +0.0] |
| family seedream3 | 35 | 74.3 % | 82.9 % | +8.6 | [+0.0, +19.4] |
| family seedream4 | 80 | 75.0 % | 76.2 % | +1.2 | [+0.0, +3.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

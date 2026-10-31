# Ablation: img.hf.ateeqq

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.ateeqq ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.230), with 5.0 % (q 0.317)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +3.7 pts, CI [+1.6, +6.0] |
| A3b | pass | family flux2, family imagen4, family qwen-image |
| A4 | fail | pooled reals: Δ FPR +0.1 pts, CI [-0.8, +1.1] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): R5; not established: R1, R2, R3, R4, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 84.3 % | 81.7 % | -2.6 |
| F2 | 70 | 32.9 % | 38.6 % | +5.7 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 37.3 % | 40.7 % | +3.4 |
| F5 | 80 | 78.8 % | 77.5 % | -1.2 |
| F6 | 106 | 34.0 % | 45.3 % | +11.3 |
| F7 | 106 | 53.8 % | 65.1 % | +11.3 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 48.1 %, with 51.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.4 % | 3.8 % | +1.4 | [-1.7, +2.9] |
| R2 | 612 | 4.9 % | 3.6 % | -1.3 | [-2.3, +0.3] |
| R3 | 589 | 4.6 % | 5.4 % | +0.8 | [-0.5, +2.2] |
| R4 | 493 | 11.6 % | 4.5 % | -7.1 | [-9.9, -4.4] |
| R5 | 548 | 1.3 % | 7.1 % | +5.8 | [+3.8, +7.8] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.8 % | +1.2 | [-1.9, +4.3] |
| threat faceswap | 272 | 67.6 % | 63.6 % | -4.0 | [-15.0, +7.2] |
| threat gan_face | 153 | 60.1 % | 51.6 % | -8.5 | [-14.4, -3.3] |
| threat personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-5.2, +1.7] |
| threat t2i | 1600 | 61.4 % | 66.9 % | +5.6 | [+4.1, +7.0] |
| family faceswap-stills | 272 | 67.6 % | 63.6 % | -4.0 | [-15.0, +7.2] |
| family flux1 | 85 | 63.5 % | 67.1 % | +3.5 | [+0.0, +8.0] |
| family flux2 | 70 | 28.6 % | 38.6 % | +10.0 | [+4.1, +17.4] |
| family gan-faces | 153 | 60.1 % | 51.6 % | -8.5 | [-14.4, -3.3] |
| family gpt-4o-image-1 | 41 | 9.8 % | 22.0 % | +12.2 | [+0.0, +24.4] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 14.0 % | +0.0 | [-5.3, +5.3] |
| family hidream | 50 | 80.0 % | 88.0 % | +8.0 | [+0.0, +17.6] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 72.3 % | -2.1 | [-10.0, +4.4] |
| family imagen3 | 53 | 49.1 % | 56.6 % | +7.5 | [-3.8, +18.9] |
| family imagen4 | 106 | 49.1 % | 61.3 % | +12.3 | [+5.6, +19.2] |
| family legacy | 145 | 90.3 % | 94.5 % | +4.1 | [-1.3, +9.7] |
| family long-tail | 186 | 62.9 % | 70.4 % | +7.5 | [+3.8, +11.6] |
| family midjourney | 153 | 81.7 % | 82.4 % | +0.7 | [-2.6, +4.1] |
| family modern-t2i-faces | 115 | 58.3 % | 67.8 % | +9.6 | [+5.3, +14.6] |
| family nano-banana | 118 | 32.2 % | 40.7 % | +8.5 | [+3.4, +14.3] |
| family partial-edits | 161 | 64.6 % | 65.8 % | +1.2 | [-1.9, +4.3] |
| family personalised | 115 | 96.5 % | 94.8 % | -1.7 | [-5.2, +1.7] |
| family qwen-image | 106 | 29.2 % | 40.6 % | +11.3 | [+5.6, +17.8] |
| family recraft | 54 | 88.9 % | 83.3 % | -5.6 | [-13.5, +1.9] |
| family sd3 | 75 | 96.0 % | 98.7 % | +2.7 | [+0.0, +6.4] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [-8.6, +8.6] |
| family seedream4 | 80 | 76.2 % | 77.5 % | +1.3 | [-3.8, +7.3] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

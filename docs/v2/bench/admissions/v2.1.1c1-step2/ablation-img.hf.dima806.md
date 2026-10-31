# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 4.9 % (q 0.855), with 5.0 % (q 0.317)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.8 pts, CI [+0.4, +3.4] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.4, +0.4] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 81.7 % | -1.3 |
| F2 | 70 | 31.4 % | 38.6 % | +7.1 |
| F3 | 57 | 12.3 % | 14.0 % | +1.8 |
| F4 | 118 | 41.5 % | 40.7 % | -0.8 |
| F5 | 80 | 76.2 % | 77.5 % | +1.3 |
| F6 | 106 | 40.6 % | 45.3 % | +4.7 |
| F7 | 106 | 65.1 % | 65.1 % | +0.0 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 50.0 %, with 51.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.5 % | 3.8 % | +0.2 | [+0.0, +0.6] |
| R2 | 612 | 3.8 % | 3.6 % | -0.2 | [-0.7, +0.0] |
| R3 | 589 | 4.6 % | 5.4 % | +0.8 | [+0.0, +1.8] |
| R4 | 493 | 5.1 % | 4.5 % | -0.5 | [-1.5, +0.2] |
| R5 | 548 | 7.5 % | 7.1 % | -0.3 | [-1.8, +1.0] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.8 % | +0.6 | [+0.0, +1.9] |
| threat faceswap | 272 | 62.5 % | 63.6 % | +1.1 | [-1.2, +3.2] |
| threat gan_face | 153 | 45.8 % | 51.6 % | +5.9 | [+2.0, +10.5] |
| threat personalised | 115 | 95.7 % | 94.8 % | -0.9 | [-2.6, +0.0] |
| threat t2i | 1600 | 67.2 % | 66.9 % | -0.2 | [-0.9, +0.4] |
| family faceswap-stills | 272 | 62.5 % | 63.6 % | +1.1 | [-1.2, +3.2] |
| family flux1 | 85 | 64.7 % | 67.1 % | +2.4 | [+0.0, +6.1] |
| family flux2 | 70 | 37.1 % | 38.6 % | +1.4 | [+0.0, +4.5] |
| family gan-faces | 153 | 45.8 % | 51.6 % | +5.9 | [+2.0, +10.5] |
| family gpt-4o-image-1 | 41 | 19.5 % | 22.0 % | +2.4 | [+0.0, +7.3] |
| family gpt-image-1.5-2 | 57 | 15.8 % | 14.0 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 86.0 % | 88.0 % | +2.0 | [+0.0, +6.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 72.3 % | -2.1 | [-6.8, +0.0] |
| family imagen3 | 53 | 54.7 % | 56.6 % | +1.9 | [-3.8, +7.6] |
| family imagen4 | 106 | 64.2 % | 61.3 % | -2.8 | [-7.0, +1.0] |
| family legacy | 145 | 95.2 % | 94.5 % | -0.7 | [-2.2, +0.0] |
| family long-tail | 186 | 71.0 % | 70.4 % | -0.5 | [-1.9, +0.0] |
| family midjourney | 153 | 82.4 % | 82.4 % | +0.0 | [-1.9, +1.9] |
| family modern-t2i-faces | 115 | 67.8 % | 67.8 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 42.4 % | 40.7 % | -1.7 | [-4.3, +0.0] |
| family partial-edits | 161 | 65.2 % | 65.8 % | +0.6 | [+0.0, +1.9] |
| family personalised | 115 | 95.7 % | 94.8 % | -0.9 | [-2.6, +0.0] |
| family qwen-image | 106 | 40.6 % | 40.6 % | +0.0 | [-2.8, +2.8] |
| family recraft | 54 | 87.0 % | 83.3 % | -3.7 | [-9.3, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 76.2 % | 77.5 % | +1.3 | [+0.0, +4.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

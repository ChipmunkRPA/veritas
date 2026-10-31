# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.145), with 5.0 % (q 0.365)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +0.8 pts, CI [-0.2, +1.9] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.0 pts, CI [-0.4, +0.4] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 86.3 % | 87.6 % | +1.3 |
| F2 | 70 | 38.6 % | 38.6 % | +0.0 |
| F3 | 57 | 19.3 % | 19.3 % | +0.0 |
| F4 | 118 | 48.3 % | 46.6 % | -1.7 |
| F5 | 80 | 78.8 % | 81.2 % | +2.5 |
| F6 | 106 | 42.5 % | 44.3 % | +1.9 |
| F7 | 106 | 64.2 % | 66.0 % | +1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 54.0 %, with 54.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.5 % | 4.5 % | -0.0 | [-0.1, +0.0] |
| R2 | 612 | 5.2 % | 5.2 % | +0.0 | [-0.8, +1.1] |
| R3 | 589 | 6.3 % | 5.8 % | -0.5 | [-1.4, +0.3] |
| R4 | 493 | 6.5 % | 7.2 % | +0.7 | [-0.7, +2.1] |
| R5 | 548 | 1.6 % | 1.6 % | +0.0 | [-0.7, +0.7] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.8 % | 65.8 % | +0.0 | [+0.0, +0.0] |
| threat faceswap | 272 | 66.9 % | 68.4 % | +1.5 | [+0.3, +3.0] |
| threat gan_face | 153 | 56.9 % | 62.1 % | +5.2 | [+2.0, +8.5] |
| threat personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| threat t2i | 1600 | 68.2 % | 67.8 % | -0.4 | [-1.1, +0.3] |
| family faceswap-stills | 272 | 66.9 % | 68.4 % | +1.5 | [+0.3, +3.0] |
| family flux1 | 85 | 68.2 % | 69.4 % | +1.2 | [+0.0, +3.7] |
| family flux2 | 70 | 38.6 % | 35.7 % | -2.9 | [-7.2, +0.0] |
| family gan-faces | 153 | 56.9 % | 62.1 % | +5.2 | [+2.0, +8.5] |
| family gpt-4o-image-1 | 41 | 22.0 % | 26.8 % | +4.9 | [+0.0, +12.2] |
| family gpt-image-1.5-2 | 57 | 19.3 % | 15.8 % | -3.5 | [-8.8, +0.0] |
| family hidream | 50 | 92.0 % | 90.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 76.6 % | +0.0 | [-6.5, +6.4] |
| family imagen3 | 53 | 54.7 % | 54.7 % | +0.0 | [-5.7, +5.7] |
| family imagen4 | 106 | 60.4 % | 62.3 % | +1.9 | [+0.0, +4.8] |
| family legacy | 145 | 95.2 % | 95.9 % | +0.7 | [+0.0, +2.3] |
| family long-tail | 186 | 73.1 % | 71.0 % | -2.2 | [-4.5, -0.5] |
| family midjourney | 153 | 85.6 % | 85.0 % | -0.7 | [-2.0, +0.0] |
| family modern-t2i-faces | 115 | 65.2 % | 64.3 % | -0.9 | [-2.9, +0.0] |
| family nano-banana | 118 | 44.1 % | 41.5 % | -2.5 | [-6.8, +1.7] |
| family partial-edits | 161 | 65.8 % | 65.8 % | +0.0 | [+0.0, +0.0] |
| family personalised | 115 | 96.5 % | 97.4 % | +0.9 | [+0.0, +2.6] |
| family qwen-image | 106 | 39.6 % | 39.6 % | +0.0 | [-2.8, +2.9] |
| family recraft | 54 | 90.7 % | 92.6 % | +1.9 | [-3.8, +7.7] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 77.5 % | 77.5 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

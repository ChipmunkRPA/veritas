# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.272), with 5.0 % (q 0.407)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.9 pts, CI [+0.0, +4.0] |
| A3b | pass | family gan-faces, threat gan_face |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.5, +0.5] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 85.6 % | 87.6 % | +2.0 |
| F2 | 70 | 45.7 % | 45.7 % | +0.0 |
| F3 | 57 | 28.1 % | 29.8 % | +1.8 |
| F4 | 118 | 56.8 % | 57.6 % | +0.8 |
| F5 | 80 | 66.2 % | 71.2 % | +5.0 |
| F6 | 106 | 42.5 % | 44.3 % | +1.9 |
| F7 | 106 | 70.8 % | 72.6 % | +1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 56.5 %, with 58.4 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 4.0 % | 3.8 % | -0.2 | [-0.5, +0.0] |
| R2 | 612 | 4.7 % | 5.1 % | +0.3 | [-0.5, +0.7] |
| R3 | 589 | 7.2 % | 6.5 % | -0.7 | [-2.0, +0.6] |
| R4 | 493 | 6.3 % | 5.9 % | -0.4 | [-1.4, +0.6] |
| R5 | 548 | 1.8 % | 3.0 % | +1.2 | [+0.0, +2.4] |
| R6 | 28 | 21.4 % | 17.9 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 67.1 % | 67.1 % | +0.0 | [-1.9, +1.9] |
| threat faceswap | 272 | 53.3 % | 57.7 % | +4.4 | [+1.9, +7.2] |
| threat gan_face | 153 | 24.2 % | 39.9 % | +15.7 | [+10.5, +21.6] |
| threat personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 67.0 % | 67.2 % | +0.2 | [-0.6, +1.1] |
| family faceswap-stills | 272 | 53.3 % | 57.7 % | +4.4 | [+1.9, +7.2] |
| family flux1 | 85 | 63.5 % | 67.1 % | +3.5 | [+0.0, +8.2] |
| family flux2 | 70 | 40.0 % | 42.9 % | +2.9 | [+0.0, +7.0] |
| family gan-faces | 153 | 24.2 % | 39.9 % | +15.7 | [+10.5, +21.6] |
| family gpt-4o-image-1 | 41 | 41.5 % | 41.5 % | +0.0 | [-9.8, +9.8] |
| family gpt-image-1.5-2 | 57 | 19.3 % | 19.3 % | +0.0 | [-5.3, +5.3] |
| family hidream | 50 | 88.0 % | 86.0 % | -2.0 | [-6.2, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 76.6 % | 78.7 % | +2.1 | [+0.0, +6.7] |
| family imagen3 | 53 | 52.8 % | 56.6 % | +3.8 | [+0.0, +9.4] |
| family imagen4 | 106 | 64.2 % | 64.2 % | +0.0 | [-3.8, +3.8] |
| family legacy | 145 | 91.7 % | 91.0 % | -0.7 | [-3.0, +1.5] |
| family long-tail | 186 | 69.9 % | 71.0 % | +1.1 | [-1.0, +3.3] |
| family midjourney | 153 | 83.0 % | 81.7 % | -1.3 | [-4.0, +1.3] |
| family modern-t2i-faces | 115 | 68.7 % | 67.8 % | -0.9 | [-3.8, +1.9] |
| family nano-banana | 118 | 49.2 % | 45.8 % | -3.4 | [-8.4, +0.9] |
| family partial-edits | 161 | 67.1 % | 67.1 % | +0.0 | [-1.9, +1.9] |
| family personalised | 115 | 96.5 % | 96.5 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 36.8 % | 37.7 % | +0.9 | [-1.9, +4.6] |
| family recraft | 54 | 83.3 % | 87.0 % | +3.7 | [+0.0, +9.6] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 80.0 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 61.3 % | 61.3 % | +0.0 | [-3.7, +3.8] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

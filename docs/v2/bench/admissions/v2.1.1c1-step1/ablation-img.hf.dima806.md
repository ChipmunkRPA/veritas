# Ablation: img.hf.dima806

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.hf.dima806 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.commfor384, img.dda, img.dfdc_b7, img.hf.ateeqq
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.069), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro -0.6 pts, CI [-2.0, +0.9] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.3, +0.5] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.0 % | 81.0 % | -2.0 |
| F2 | 70 | 34.3 % | 38.6 % | +4.3 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 42.4 % | 40.7 % | -1.7 |
| F5 | 80 | 76.2 % | 76.2 % | +0.0 |
| F6 | 106 | 41.5 % | 40.6 % | -0.9 |
| F7 | 106 | 65.1 % | 63.2 % | -1.9 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 51.2 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 8.0 % | 7.3 % | -0.7 | [-1.5, +0.0] |
| R2 | 612 | 3.4 % | 3.4 % | +0.0 | [-0.5, +0.4] |
| R3 | 589 | 4.4 % | 4.4 % | +0.0 | [-0.5, +0.5] |
| R4 | 493 | 4.9 % | 4.1 % | -0.8 | [-1.6, -0.2] |
| R5 | 548 | 3.6 % | 5.2 % | +1.5 | [+0.3, +2.8] |
| R6 | 28 | 10.7 % | 10.7 % | +0.0 | [+0.0, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.6 % | 65.2 % | +0.6 | [+0.0, +1.9] |
| threat faceswap | 272 | 78.3 % | 79.8 % | +1.5 | [+0.0, +3.7] |
| threat gan_face | 153 | 58.8 % | 59.5 % | +0.7 | [+0.0, +2.0] |
| threat personalised | 115 | 93.9 % | 93.9 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 66.1 % | 65.6 % | -0.6 | [-1.2, +0.1] |
| family faceswap-stills | 272 | 78.3 % | 79.8 % | +1.5 | [+0.0, +3.7] |
| family flux1 | 85 | 64.7 % | 64.7 % | +0.0 | [+0.0, +0.0] |
| family flux2 | 70 | 34.3 % | 37.1 % | +2.9 | [-2.9, +8.6] |
| family gan-faces | 153 | 58.8 % | 59.5 % | +0.7 | [+0.0, +2.0] |
| family gpt-4o-image-1 | 41 | 19.5 % | 17.1 % | -2.4 | [-7.3, +0.0] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 12.3 % | -1.8 | [-5.3, +0.0] |
| family hidream | 50 | 86.0 % | 86.0 % | +0.0 | [+0.0, +0.0] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 72.3 % | -2.1 | [-6.8, +0.0] |
| family imagen3 | 53 | 50.9 % | 50.9 % | +0.0 | [-5.7, +5.7] |
| family imagen4 | 106 | 62.3 % | 61.3 % | -0.9 | [-3.0, +0.0] |
| family legacy | 145 | 95.2 % | 94.5 % | -0.7 | [-2.2, +0.0] |
| family long-tail | 186 | 70.4 % | 70.4 % | +0.0 | [-2.9, +3.2] |
| family midjourney | 153 | 82.4 % | 81.7 % | -0.7 | [-2.0, +0.0] |
| family modern-t2i-faces | 115 | 64.3 % | 64.3 % | +0.0 | [+0.0, +0.0] |
| family nano-banana | 118 | 42.4 % | 40.7 % | -1.7 | [-4.3, +0.0] |
| family partial-edits | 161 | 64.6 % | 65.2 % | +0.6 | [+0.0, +1.9] |
| family personalised | 115 | 93.9 % | 93.9 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 37.7 % | 37.7 % | +0.0 | [-2.8, +2.9] |
| family recraft | 54 | 85.2 % | 81.5 % | -3.7 | [-9.3, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 75.0 % | 73.8 % | -1.2 | [-3.8, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

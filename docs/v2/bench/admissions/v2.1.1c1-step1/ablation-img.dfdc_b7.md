# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.commfor384, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.624), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.9 pts, CI [+1.6, +4.4] |
| A3b | fail | no family or threat gains ≥ 10 pts |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.9, +0.7] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R1; above +2.0 on fewer than 150 reals (no term): R6 |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 79.7 % | 81.0 % | +1.3 |
| F2 | 70 | 34.3 % | 38.6 % | +4.3 |
| F3 | 57 | 12.3 % | 14.0 % | +1.8 |
| F4 | 118 | 39.8 % | 40.7 % | +0.8 |
| F5 | 80 | 72.5 % | 76.2 % | +3.7 |
| F6 | 106 | 36.8 % | 40.6 % | +3.8 |
| F7 | 106 | 58.5 % | 63.2 % | +4.7 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 47.7 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.1 % | 7.3 % | +2.2 | [+0.0, +4.0] |
| R2 | 612 | 2.8 % | 3.4 % | +0.7 | [+0.0, +1.1] |
| R3 | 589 | 4.2 % | 4.4 % | +0.2 | [+0.0, +0.5] |
| R4 | 493 | 3.7 % | 4.1 % | +0.4 | [-0.4, +1.2] |
| R5 | 548 | 8.9 % | 5.2 % | -3.8 | [-6.0, -1.6] |
| R6 | 28 | 3.6 % | 10.7 % | +7.1 | [+0.0, +19.2] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.0 % | 65.2 % | +1.2 | [+0.0, +3.1] |
| threat faceswap | 272 | 73.5 % | 79.8 % | +6.2 | [+2.4, +10.6] |
| threat gan_face | 153 | 64.1 % | 59.5 % | -4.6 | [-9.2, +0.0] |
| threat personalised | 115 | 94.8 % | 93.9 % | -0.9 | [-2.6, +0.0] |
| threat t2i | 1600 | 64.1 % | 65.6 % | +1.4 | [+0.6, +2.3] |
| family faceswap-stills | 272 | 73.5 % | 79.8 % | +6.2 | [+2.4, +10.6] |
| family flux1 | 85 | 62.4 % | 64.7 % | +2.4 | [+0.0, +6.1] |
| family flux2 | 70 | 34.3 % | 37.1 % | +2.9 | [-2.9, +8.8] |
| family gan-faces | 153 | 64.1 % | 59.5 % | -4.6 | [-9.2, +0.0] |
| family gpt-4o-image-1 | 41 | 12.2 % | 17.1 % | +4.9 | [+0.0, +12.2] |
| family gpt-image-1.5-2 | 57 | 10.5 % | 12.3 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 84.0 % | 86.0 % | +2.0 | [+0.0, +6.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 72.3 % | 72.3 % | +0.0 | [+0.0, +0.0] |
| family imagen3 | 53 | 43.4 % | 50.9 % | +7.5 | [+1.9, +15.1] |
| family imagen4 | 106 | 58.5 % | 61.3 % | +2.8 | [+0.0, +6.6] |
| family legacy | 145 | 93.8 % | 94.5 % | +0.7 | [+0.0, +2.2] |
| family long-tail | 186 | 68.8 % | 70.4 % | +1.6 | [-1.3, +5.0] |
| family midjourney | 153 | 80.4 % | 81.7 % | +1.3 | [+0.0, +3.4] |
| family modern-t2i-faces | 115 | 67.8 % | 64.3 % | -3.5 | [-8.2, +0.9] |
| family nano-banana | 118 | 39.8 % | 40.7 % | +0.8 | [+0.0, +2.6] |
| family partial-edits | 161 | 64.0 % | 65.2 % | +1.2 | [+0.0, +3.1] |
| family personalised | 115 | 94.8 % | 93.9 % | -0.9 | [-2.6, +0.0] |
| family qwen-image | 106 | 33.0 % | 37.7 % | +4.7 | [+0.9, +10.1] |
| family recraft | 54 | 81.5 % | 81.5 % | +0.0 | [+0.0, +0.0] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 73.8 % | 73.8 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

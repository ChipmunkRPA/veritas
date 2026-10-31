# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=face.gend_clip_l14,img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='face.gend_clip_l14=has_face;img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: face.gend_clip_l14, img.dda, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.221), with 5.0 % (q 0.413)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | fail | Δ LGO macro +1.4 pts, CI [-0.8, +3.6] |
| A3b | pass | family gan-faces, family seedream4, threat gan_face |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.8, +0.7] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 80.4 % | 81.0 % | +0.7 |
| F2 | 70 | 42.9 % | 38.6 % | -4.3 |
| F3 | 57 | 14.0 % | 14.0 % | +0.0 |
| F4 | 118 | 40.7 % | 40.7 % | +0.0 |
| F5 | 80 | 58.8 % | 76.2 % | +17.5 |
| F6 | 106 | 41.5 % | 40.6 % | -0.9 |
| F7 | 106 | 66.0 % | 63.2 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 49.2 %, with 50.6 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 5.9 % | 7.3 % | +1.4 | [-1.7, +3.5] |
| R2 | 612 | 2.8 % | 3.4 % | +0.6 | [+0.1, +1.4] |
| R3 | 589 | 5.8 % | 4.4 % | -1.4 | [-2.9, +0.2] |
| R4 | 493 | 5.1 % | 4.1 % | -1.0 | [-2.6, +0.4] |
| R5 | 548 | 4.7 % | 5.2 % | +0.4 | [-0.9, +1.8] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 64.0 % | 65.2 % | +1.2 | [-2.5, +5.0] |
| threat faceswap | 272 | 71.7 % | 79.8 % | +8.1 | [+4.0, +12.8] |
| threat gan_face | 153 | 35.9 % | 59.5 % | +23.5 | [+15.7, +31.4] |
| threat personalised | 115 | 93.0 % | 93.9 % | +0.9 | [-4.3, +6.1] |
| threat t2i | 1600 | 65.0 % | 65.6 % | +0.6 | [-0.9, +2.1] |
| family faceswap-stills | 272 | 71.7 % | 79.8 % | +8.1 | [+4.0, +12.8] |
| family flux1 | 85 | 63.5 % | 64.7 % | +1.2 | [-4.9, +7.1] |
| family flux2 | 70 | 40.0 % | 37.1 % | -2.9 | [-10.0, +4.2] |
| family gan-faces | 153 | 35.9 % | 59.5 % | +23.5 | [+15.7, +31.4] |
| family gpt-4o-image-1 | 41 | 31.7 % | 17.1 % | -14.6 | [-26.8, -4.9] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 12.3 % | -1.8 | [-7.1, +3.5] |
| family hidream | 50 | 86.0 % | 86.0 % | +0.0 | [-6.0, +5.9] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 70.2 % | 72.3 % | +2.1 | [-8.0, +11.1] |
| family imagen3 | 53 | 52.8 % | 50.9 % | -1.9 | [-13.2, +9.4] |
| family imagen4 | 106 | 63.2 % | 61.3 % | -1.9 | [-7.9, +4.6] |
| family legacy | 145 | 89.7 % | 94.5 % | +4.8 | [+1.9, +8.7] |
| family long-tail | 186 | 71.0 % | 70.4 % | -0.5 | [-4.7, +3.8] |
| family midjourney | 153 | 79.7 % | 81.7 % | +2.0 | [-1.3, +5.3] |
| family modern-t2i-faces | 115 | 71.3 % | 64.3 % | -7.0 | [-14.3, +0.0] |
| family nano-banana | 118 | 39.0 % | 40.7 % | +1.7 | [-4.2, +7.7] |
| family partial-edits | 161 | 64.0 % | 65.2 % | +1.2 | [-2.5, +5.0] |
| family personalised | 115 | 93.0 % | 93.9 % | +0.9 | [-4.3, +6.1] |
| family qwen-image | 106 | 38.7 % | 37.7 % | -0.9 | [-6.7, +4.8] |
| family recraft | 54 | 77.8 % | 81.5 % | +3.7 | [+0.0, +9.6] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 77.1 % | 82.9 % | +5.7 | [+0.0, +14.7] |
| family seedream4 | 80 | 57.5 % | 73.8 % | +16.3 | [+8.8, +25.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

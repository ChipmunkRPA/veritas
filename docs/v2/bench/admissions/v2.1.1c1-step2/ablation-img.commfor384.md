# Ablation: img.commfor384

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.commfor384 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.dda, img.dfdc_b7, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (OwensLab/CommunityForensics-Eval)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.0 % (q 0.177), with 5.0 % (q 0.317)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +2.6 pts, CI [+0.2, +5.0] |
| A3b | pass | family faceswap-stills, family gan-faces, family seedream4, threat faceswap, threat gan_face |
| A4 | pass | pooled reals: Δ FPR +0.0 pts, CI [-0.7, +0.6] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: none; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 80.4 % | 81.7 % | +1.3 |
| F2 | 70 | 41.4 % | 38.6 % | -2.9 |
| F3 | 57 | 15.8 % | 14.0 % | -1.8 |
| F4 | 118 | 41.5 % | 40.7 % | -0.8 |
| F5 | 80 | 58.8 % | 77.5 % | +18.8 |
| F6 | 106 | 38.7 % | 45.3 % | +6.6 |
| F7 | 106 | 67.9 % | 65.1 % | -2.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 49.2 %, with 51.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 3.3 % | 3.8 % | +0.5 | [-2.5, +1.9] |
| R2 | 612 | 3.1 % | 3.6 % | +0.5 | [-0.4, +1.6] |
| R3 | 589 | 5.4 % | 5.4 % | +0.0 | [-1.4, +1.4] |
| R4 | 493 | 5.5 % | 4.5 % | -0.9 | [-2.4, +0.5] |
| R5 | 548 | 7.0 % | 7.1 % | +0.2 | [-1.2, +1.5] |
| R6 | 28 | 14.3 % | 10.7 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.2 % | 65.8 % | +0.6 | [-3.1, +5.0] |
| threat faceswap | 272 | 44.1 % | 63.6 % | +19.5 | [+11.2, +30.1] |
| threat gan_face | 153 | 23.5 % | 51.6 % | +28.1 | [+20.3, +35.3] |
| threat personalised | 115 | 94.8 % | 94.8 % | +0.0 | [-4.3, +4.3] |
| threat t2i | 1600 | 65.9 % | 66.9 % | +1.1 | [-0.4, +2.6] |
| family faceswap-stills | 272 | 44.1 % | 63.6 % | +19.5 | [+11.2, +30.1] |
| family flux1 | 85 | 63.5 % | 67.1 % | +3.5 | [-2.4, +9.5] |
| family flux2 | 70 | 41.4 % | 38.6 % | -2.9 | [-8.7, +2.8] |
| family gan-faces | 153 | 23.5 % | 51.6 % | +28.1 | [+20.3, +35.3] |
| family gpt-4o-image-1 | 41 | 34.1 % | 22.0 % | -12.2 | [-22.0, -2.4] |
| family gpt-image-1.5-2 | 57 | 17.5 % | 14.0 % | -3.5 | [-12.3, +5.3] |
| family hidream | 50 | 86.0 % | 88.0 % | +2.0 | [+0.0, +6.4] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 72.3 % | -2.1 | [-12.8, +6.5] |
| family imagen3 | 53 | 54.7 % | 56.6 % | +1.9 | [-9.4, +13.2] |
| family imagen4 | 106 | 64.2 % | 61.3 % | -2.8 | [-9.5, +3.9] |
| family legacy | 145 | 89.7 % | 94.5 % | +4.8 | [+1.9, +8.7] |
| family long-tail | 186 | 69.9 % | 70.4 % | +0.5 | [-3.5, +4.8] |
| family midjourney | 153 | 80.4 % | 82.4 % | +2.0 | [-1.3, +5.4] |
| family modern-t2i-faces | 115 | 73.0 % | 67.8 % | -5.2 | [-12.9, +1.8] |
| family nano-banana | 118 | 39.8 % | 40.7 % | +0.8 | [-5.1, +6.8] |
| family partial-edits | 161 | 65.2 % | 65.8 % | +0.6 | [-3.1, +5.0] |
| family personalised | 115 | 94.8 % | 94.8 % | +0.0 | [-4.3, +4.3] |
| family qwen-image | 106 | 38.7 % | 40.6 % | +1.9 | [-3.7, +7.3] |
| family recraft | 54 | 81.5 % | 83.3 % | +1.9 | [-3.8, +7.7] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 80.0 % | 82.9 % | +2.9 | [-5.9, +14.3] |
| family seedream4 | 80 | 58.8 % | 77.5 % | +18.8 | [+10.1, +28.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).

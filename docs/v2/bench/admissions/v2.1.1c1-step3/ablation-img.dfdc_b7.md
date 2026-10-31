# Ablation: img.dfdc_b7

- Command: `make calibrate MODALITY=image SEED=7 BENCH=vb-1.1 DETECTORS=img.commfor384,img.dda,img.dfdc_b7,img.hf.ateeqq,img.hf.dima806 GATES='img.dfdc_b7=has_face;img.hf.ateeqq=!has_face' CONTEXT=scored SCOPE=data/vbench/scope/vb-1.1.parquet ABLATE=img.dfdc_b7 ABLATE_BOOTSTRAP=2000`
- Manifest sha256: `8314119e487a9df3dd7505e55010577c32819e7c8440b21c8ecf325e51c9c3c3`
- Baseline detectors: img.commfor384, img.dda, img.hf.ateeqq, img.hf.dima806
- Items (fit ∪ cal): 4995 (2694 real, 2301 fake); contaminated dropped 0 (none listed)
- Paired grouped bootstrap B = 2000, seed 7; cross-fit 5 folds; τ at α = 0.05; weighting strata merged on the cal reals: R6 → R3
- Operating point (D-53): reals scoring exactly τ count as flagged with weight q (the rank rule's share); served rule (s ≥ τ) weighted FPR: without 5.6 % (q 0.678), with 5.0 % (q 0.365)

| Criterion | Result | Evidence |
|---|---|---|
| A3a | pass | Δ LGO macro +3.5 pts, CI [+2.1, +5.0] |
| A3b | pass | family faceswap-stills, threat faceswap |
| A4 | pass | pooled reals: Δ FPR -0.1 pts, CI [-0.8, +0.5] (blocks with a lower bound above +1.0, D-53); blocking strata (n ≥ 150, lower bound > +2.0): none; not established: R1, R2, R3, R4, R5, R6; point above +2.0 without established harm: R4; above +2.0 on fewer than 150 reals (no term): none |

## LGO folds (TPR@5%FPR on fit / cal fakes of F; refits without F)

| Fold | n fake | without | with | Δ pts |
|---|---|---|---|---|
| F1 | 153 | 83.7 % | 87.6 % | +3.9 |
| F2 | 70 | 37.1 % | 38.6 % | +1.4 |
| F3 | 57 | 15.8 % | 19.3 % | +3.5 |
| F4 | 118 | 43.2 % | 46.6 % | +3.4 |
| F5 | 80 | 77.5 % | 81.2 % | +3.7 |
| F6 | 106 | 39.6 % | 44.3 % | +4.7 |
| F7 | 106 | 62.3 % | 66.0 % | +3.8 |

Macro over F1, F2, F3, F4, F5, F6, F7: without 51.3 %, with 54.8 %. Missing: none.

## Real strata (out-of-fold FPR at τ_0.05)

| Stratum | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| R1 | 424 | 2.8 % | 4.5 % | +1.7 | [+0.0, +2.5] |
| R2 | 612 | 3.8 % | 5.2 % | +1.5 | [+0.7, +2.8] |
| R3 | 589 | 5.1 % | 5.8 % | +0.7 | [+0.2, +1.4] |
| R4 | 493 | 5.0 % | 7.2 % | +2.2 | [+0.7, +3.8] |
| R5 | 548 | 7.4 % | 1.6 % | -5.8 | [-7.6, -3.9] |
| R6 | 28 | 17.9 % | 14.3 % | -3.6 | [-12.5, +0.0] |

## Recall per threat and family (out-of-fold, τ_0.05)

| Slice | n | without | with | Δ pts | 95 % CI |
|---|---|---|---|---|---|
| threat edit | 161 | 65.6 % | 65.8 % | +0.2 | [+0.0, +0.6] |
| threat faceswap | 272 | 27.4 % | 68.4 % | +41.0 | [+27.2, +53.4] |
| threat gan_face | 153 | 55.2 % | 62.1 % | +6.8 | [+0.0, +14.5] |
| threat personalised | 115 | 97.4 % | 97.4 % | +0.0 | [+0.0, +0.0] |
| threat t2i | 1600 | 65.7 % | 67.8 % | +2.1 | [+1.4, +2.9] |
| family faceswap-stills | 272 | 27.4 % | 68.4 % | +41.0 | [+27.2, +53.4] |
| family flux1 | 85 | 64.7 % | 69.4 % | +4.7 | [+1.1, +9.6] |
| family flux2 | 70 | 35.7 % | 35.7 % | +0.0 | [+0.0, +0.0] |
| family gan-faces | 153 | 55.2 % | 62.1 % | +6.8 | [+0.0, +14.5] |
| family gpt-4o-image-1 | 41 | 17.1 % | 26.8 % | +9.8 | [+2.4, +19.5] |
| family gpt-image-1.5-2 | 57 | 14.0 % | 15.8 % | +1.8 | [+0.0, +5.3] |
| family hidream | 50 | 86.0 % | 90.0 % | +4.0 | [+0.0, +10.2] |
| family hunyuanimage | 24 | 100.0 % | 100.0 % | +0.0 | [+0.0, +0.0] |
| family ideogram | 47 | 74.5 % | 76.6 % | +2.1 | [+0.0, +6.8] |
| family imagen3 | 53 | 49.1 % | 54.7 % | +5.7 | [+0.0, +13.2] |
| family imagen4 | 106 | 57.5 % | 62.3 % | +4.7 | [+0.0, +9.6] |
| family legacy | 145 | 94.5 % | 95.9 % | +1.4 | [+0.0, +3.6] |
| family long-tail | 186 | 69.4 % | 71.0 % | +1.6 | [+0.0, +3.6] |
| family midjourney | 153 | 83.7 % | 85.0 % | +1.3 | [+0.0, +3.3] |
| family modern-t2i-faces | 115 | 64.3 % | 64.3 % | +0.0 | [-2.6, +2.6] |
| family nano-banana | 118 | 40.7 % | 41.5 % | +0.8 | [-1.7, +4.2] |
| family partial-edits | 161 | 65.6 % | 65.8 % | +0.2 | [+0.0, +0.6] |
| family personalised | 115 | 97.4 % | 97.4 % | +0.0 | [+0.0, +0.0] |
| family qwen-image | 106 | 36.8 % | 39.6 % | +2.8 | [+0.0, +6.3] |
| family recraft | 54 | 87.0 % | 92.6 % | +5.6 | [+0.0, +11.5] |
| family sd3 | 75 | 98.7 % | 98.7 % | +0.0 | [+0.0, +0.0] |
| family seedream3 | 35 | 82.9 % | 82.9 % | +0.0 | [+0.0, +0.0] |
| family seedream4 | 80 | 77.5 % | 77.5 % | +0.0 | [+0.0, +0.0] |

Not decided here: A1 (card + parity test); A2 (vbench-report); A5 (vbench-report); A6 (latency); A4 screenshot stratum (transform rows).
